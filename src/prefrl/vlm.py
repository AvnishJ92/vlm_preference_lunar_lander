"""Qwen2.5-VL as the rater via an OpenAI-compatible HTTP API (e.g. vLLM): show it two clips, read back which one it prefers."""

import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
import math
import os
import re
import tempfile
import urllib.request
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from prefrl.env import FPS

MU = {"1": 1.0, "2": 0.0, "TIE": 0.5}  # answer -> P(first clip preferred)


def parse(reply):
    """The last `ANSWER: x` of a reply; UNSURE when there is none."""
    found = re.findall(r"ANSWER:\s*`?(1|2|TIE|UNSURE)", reply.upper())
    return found[-1] if found else "UNSURE"


def combine(first, second=None, fail_as_tie=False):
    """mu = P(clip A preferred) from the answer to (A, B) and, if given, to the swapped (B, A). None = can't tell.

    The swap cancels position bias: a rater that just says "1" contradicts itself and yields no label at all,
    instead of a wrong one. In the paper's terms a contradiction is "can't tell" and is left out of the database.
    When `fail_as_tie=True`, contradictions or UNSURE yield 0.5 (TIE).
    """
    if first == "UNSURE" or second == "UNSURE":
        return 0.5 if fail_as_tie else None
    if second is not None and first != {"1": "2", "2": "1"}.get(second, second):
        return 0.5 if fail_as_tie else None
    return MU.get(first)


def clip_offsets(k, n, ends=()):
    """n evenly spaced step offsets into a k-step clip, nudged so the final frame of an attempt is always shown.

    That frame is the crash or the landing: the single most informative one, and easy to miss in sparse sampling.
    """
    offsets = np.linspace(0, k - 1, n).round().astype(int)
    for end in ends:
        offsets[np.abs(offsets - end).argmin()] = end
    return offsets


def _scale_frames(frames, scale):
    if scale == 1:
        return frames
    size = [round(s * scale) for s in frames[0].shape[1::-1]]
    return np.stack([np.asarray(Image.fromarray(f).resize(size, Image.BICUBIC)) for f in frames])


def _label_clip(frame, label):
    image = Image.fromarray(frame)
    font_size = max(14, round(frame.shape[0] * 0.085))
    ImageDraw.Draw(image).text(
        (10, 8), label, fill=(255, 230, 80), anchor="lt", font=ImageFont.load_default(size=font_size)
    )
    return np.asarray(image)


def _side_by_side_video(a, b, scale):
    """Scale two synchronized clips, label CLIP 1 / CLIP 2 on top-left, and stack horizontally (F, H, 2*W, 3)."""
    fa, fb = _scale_frames(a, scale), _scale_frames(b, scale)
    left = np.stack([_label_clip(f, "CLIP 1") for f in fa])
    right = np.stack([_label_clip(f, "CLIP 2") for f in fb])
    stacked = np.concatenate([left, right], axis=2)
    mid = left.shape[2]
    stacked[:, :, mid - 1 : mid + 1] = 120
    return stacked


def _encode_jpeg(frame, quality=85):
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _image_block(frame, quality):
    return {"type": "image_url", "image_url": {"url": _encode_jpeg(frame, quality)}}


def _encode_mp4(frames, fps):
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        with imageio.get_writer(tmp.name, fps=fps, codec="libx264", macro_block_size=1) as writer:
            for f in frames:
                writer.append_data(f)
        raw = Path(tmp.name).read_bytes()
    return "data:video/mp4;base64," + base64.b64encode(raw).decode("ascii")


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def extract_logprob_delta(choice):
    """Return logP('1') - logP('2') from choice logprobs (scans for the answer token position)."""
    content = (choice.get("logprobs") or {}).get("content") or []
    # Scan from the end so we hit the answer token after `ANSWER:` rather than any earlier digit.
    for pos in reversed(content):
        tops = pos.get("top_logprobs") or []
        lp1_vals = [t["logprob"] for t in tops if t["token"].strip() == "1"]
        lp2_vals = [t["logprob"] for t in tops if t["token"].strip() == "2"]
        if lp1_vals or lp2_vals:
            min_lp = min(t["logprob"] for t in tops) if tops else -20.0
            lp1 = max(lp1_vals) if lp1_vals else (min_lp - 1.0)
            lp2 = max(lp2_vals) if lp2_vals else (min_lp - 1.0)
            return float(lp1 - lp2)
    return None


class QwenJudge:
    """Calls an OpenAI-compatible `/v1/chat/completions` VLM server to judge clip pairs."""

    def __init__(self, cfg, prompts, segment_len):
        self.cfg = cfg
        env_urls = os.environ.get("VLM_API_URLS", "").strip()
        urls = [u.strip() for u in env_urls.split(",") if u.strip()] if env_urls else cfg.get("api_urls", [])
        if not urls:
            raise ValueError("vlm.api_urls (or VLM_API_URLS) must contain at least one API base URL")
        self.api_urls = [u.rstrip("/") for u in urls]
        self.system, self.question = prompts["system"], prompts["variants"][cfg["prompt"]]
        if cfg["mark_out_of_view"]:
            self.system += prompts["out_of_view_rule"]
        label_key = "video" if cfg["input_mode"] == "video_mp4" else cfg["input_mode"]
        self.clip_label = prompts["clip_label"][label_key]
        self.segment_len = segment_len
        self.dt = (segment_len - 1) / (cfg["frames_per_clip"] - 1) / FPS  # seconds between shown frames

    def messages(self, a, b):
        """Build OpenAI chat-completions `messages` for an ordered pair of clips (a, b)."""
        scale = self.cfg["frame_scale"]
        quality = self.cfg.get("jpeg_quality", 85)
        if self.cfg["input_mode"] == "video_mp4":
            frames = _side_by_side_video(a, b, scale)
            video_fps = float(self.cfg.get("video_fps", len(frames) / (self.segment_len / FPS)))
            content = []
            if self.clip_label.strip():
                content.append({"type": "text", "text": self.clip_label.format(frames=len(frames), dt=self.dt) + "\n"})
            content.append({"type": "video_url", "video_url": {"url": _encode_mp4(frames, video_fps)}})
            content.append({"type": "text", "text": self.question.strip()})
        elif self.cfg["input_mode"] == "video":
            frames = _side_by_side_video(a, b, scale)
            content = [
                {"type": "text", "text": self.clip_label.format(frames=len(frames), dt=self.dt) + "\n"},
                *[_image_block(f, quality) for f in frames],
                {"type": "text", "text": "\n" + self.question},
            ]
        else:
            content = []
            for n, frames in ((1, a), (2, b)):
                scaled = _scale_frames(frames, scale)
                content.append({"type": "text", "text": self.clip_label.format(n=n, frames=len(scaled), dt=self.dt) + "\n"})
                content += [_image_block(f, quality) for f in scaled]
                content.append({"type": "text", "text": "\n"})
            content.append({"type": "text", "text": self.question})
        msgs = []
        if self.system.strip():
            msgs.append({"role": "system", "content": self.system})
        msgs.append({"role": "user", "content": content})
        return msgs

    def payload(self, a, b):
        req = {
            "model": self.cfg["model"],
            "temperature": 0.0,
            "max_tokens": self.cfg["max_new_tokens"],
            "messages": self.messages(a, b),
        }
        if self.cfg["input_mode"] == "video_mp4":
            video_fps = float(self.cfg.get("video_fps", self.cfg["frames_per_clip"] / (self.segment_len / FPS)))
            req["media_io_kwargs"] = {"video": {"fps": video_fps}}
            req["mm_processor_kwargs"] = {"cap_pixels_per_frame": False}
        if self.cfg.get("logprob_debias", False):
            req["logprobs"] = True
            req["top_logprobs"] = int(self.cfg.get("top_logprobs", 10))
        return req

    def _post_choice(self, idx_and_pair):
        idx, (a, b) = idx_and_pair
        url = self.api_urls[idx % len(self.api_urls)] + "/chat/completions"
        body = json.dumps(self.payload(a, b)).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.cfg.get('api_key', 'EMPTY')}",
        }
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.cfg.get("timeout", 120)) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]

    def _post(self, idx_and_pair):
        return self._post_choice(idx_and_pair)["message"]["content"]

    def _choices(self, pairs):
        if not pairs:
            return []
        workers = min(len(pairs), max(1, int(self.cfg.get("concurrency", 16))))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(self._post_choice, enumerate(pairs)))

    def answer(self, pairs):
        """Raw replies, one per ordered pair, dispatched concurrently across `api_urls`."""
        return [c["message"]["content"] for c in self._choices(pairs)]

    def label(self, pairs):
        """[(mu or None, replies)] per pair. With swap_check every pair is asked in both orders."""
        if self.cfg.get("logprob_debias", False) and self.cfg["swap_check"]:
            choices = self._choices(pairs + [(b, a) for a, b in pairs])
            first, second = choices[: len(pairs)], choices[len(pairs) :]
            out = []
            for c_ab, c_ba in zip(first, second):
                r_ab, r_ba = c_ab["message"]["content"], c_ba["message"]["content"]
                d_ab, d_ba = extract_logprob_delta(c_ab), extract_logprob_delta(c_ba)
                if d_ab is not None and d_ba is not None:
                    delta = 0.5 * (d_ab - d_ba)
                    mu = _sigmoid(delta)
                else:
                    mu = combine(parse(r_ab), parse(r_ba))
                out.append((mu, [r_ab, r_ba]))
            return out
        fail_as_tie = bool(self.cfg.get("swap_fail_as_tie", False))
        if not self.cfg["swap_check"]:
            return [(combine(parse(r), fail_as_tie=fail_as_tie), [r]) for r in self.answer(pairs)]
        replies = self.answer(pairs + [(b, a) for a, b in pairs])
        first, second = replies[: len(pairs)], replies[len(pairs) :]
        return [(combine(parse(f), parse(s), fail_as_tie=fail_as_tie), [f, s]) for f, s in zip(first, second)]
