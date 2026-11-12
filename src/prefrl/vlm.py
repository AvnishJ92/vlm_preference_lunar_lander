"""Qwen2.5-VL as the rater: show it two clips, read back which one it prefers."""

import re

import numpy as np
import torch
from PIL import Image

from prefrl.env import FPS

MU = {"1": 1.0, "2": 0.0, "TIE": 0.5}  # answer -> P(first clip preferred)


def parse(reply):
    """The last `ANSWER: x` of a reply; UNSURE when there is none."""
    found = re.findall(r"ANSWER:\s*`?(1|2|TIE|UNSURE)", reply.upper())
    return found[-1] if found else "UNSURE"


def combine(first, second=None):
    """mu = P(clip A preferred) from the answer to (A, B) and, if given, to the swapped (B, A). None = can't tell.

    The swap cancels position bias: a rater that just says "1" contradicts itself and yields no label at all,
    instead of a wrong one. In the paper's terms a contradiction is "can't tell" and is left out of the database.
    """
    if second is not None and first != {"1": "2", "2": "1"}.get(second, second):
        return None
    return MU.get(first)


def clip_offsets(k, n, ends=()):
    """n evenly spaced step offsets into a k-step clip, nudged so the final frame of an attempt is always shown.

    That frame is the crash or the landing: the single most informative one, and easy to miss at 8 frames per clip.
    """
    offsets = np.linspace(0, k - 1, n).round().astype(int)
    for end in ends:
        offsets[np.abs(offsets - end).argmin()] = end
    return offsets


def load_qwen(cfg):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(cfg["model"], revision=cfg["revision"])
    processor.tokenizer.padding_side = "left"  # batched generation continues from the right edge
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg["model"], revision=cfg["revision"], dtype=torch.bfloat16, device_map="cuda"
    )
    return model.eval(), processor


class QwenJudge:
    """cfg: the `vlm` config block; prompts: configs/prompts.yaml; segment_len: clip length in env steps."""

    def __init__(self, model, processor, cfg, prompts, segment_len):
        self.model, self.processor, self.cfg = model, processor, cfg
        self.system, self.question = prompts["system"], prompts["variants"][cfg["prompt"]]
        self.clip_label = prompts["clip_label"][cfg["input_mode"]]
        self.dt = (segment_len - 1) / (cfg["frames_per_clip"] - 1) / FPS  # seconds between shown frames

    def _clip(self, frames):
        if self.cfg["frame_scale"] != 1:
            size = [round(s * self.cfg["frame_scale"]) for s in frames[0].shape[1::-1]]
            frames = [np.asarray(Image.fromarray(f).resize(size, Image.BICUBIC)) for f in frames]
        if self.cfg["input_mode"] == "video":
            return [{"type": "video", "video": list(frames)}]
        return [{"type": "image", "image": Image.fromarray(f)} for f in frames]

    def conversation(self, a, b):
        content = []
        for n, frames in ((1, a), (2, b)):
            content.append({"type": "text", "text": self.clip_label.format(n=n, frames=len(frames), dt=self.dt) + "\n"})
            content += self._clip(frames)
            content.append({"type": "text", "text": "\n"})
        content.append({"type": "text", "text": self.question})
        return [
            {"role": "system", "content": [{"type": "text", "text": self.system}]},
            {"role": "user", "content": content},
        ]

    def inputs(self, pairs):
        """Model inputs for a batch of ordered pairs [(frames_a, frames_b), ...], frames (F, H, W, 3) uint8."""
        kwargs = {"padding": True}
        if self.cfg["input_mode"] == "video":  # tell Qwen's time encoding how far apart the frames really are
            n = self.cfg["frames_per_clip"]
            meta = {"total_num_frames": n, "fps": 1 / self.dt, "frames_indices": list(range(n))}
            kwargs["video_metadata"] = [meta] * (2 * len(pairs))
        return self.processor.apply_chat_template(
            [self.conversation(a, b) for a, b in pairs], add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", processor_kwargs=kwargs,
        )  # fmt: skip

    @torch.inference_mode()
    def answer(self, pairs):
        """Raw replies, one per ordered pair."""
        replies, size = [], self.cfg["batch_size"]
        for i in range(0, len(pairs), size):
            inputs = self.inputs(pairs[i : i + size]).to(self.model.device)
            ids = self.model.generate(**inputs, max_new_tokens=self.cfg["max_new_tokens"], do_sample=False)
            replies += self.processor.batch_decode(ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        return replies

    def label(self, pairs):
        """[(mu or None, replies)] per pair. With swap_check every pair is asked in both orders."""
        if not self.cfg["swap_check"]:
            return [(combine(parse(r)), [r]) for r in self.answer(pairs)]
        replies = self.answer(pairs + [(b, a) for a, b in pairs])
        first, second = replies[: len(pairs)], replies[len(pairs) :]
        return [(combine(parse(f), parse(s)), [f, s]) for f, s in zip(first, second)]
