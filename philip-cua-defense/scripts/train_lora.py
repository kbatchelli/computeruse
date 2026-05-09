"""LoRA fine-tune Northstar-CUA-Fast on injection-resistant traces.

Generic over the training JSONL — each line:
  {"image_path": "data/images/foo.png",
   "instruction": "<user task text>",
   "target": "<tool_call>{...}</tool_call>"}

Design choices (see NOTES.md for context):
- PEFT LoRA, rank 16. Tzafon explicitly warned SFT can collapse RL-trained
  models, so keep rank low and epochs ≤ 1.
- Adapt language-model attn proj + MLP only. Vision encoder stays frozen.
- bf16 on A100. Batch size 1 with grad accumulation 8 → effective 8.
- Greedy LM loss on the target span only (label-mask the prompt).
- Cosine LR, 2e-4 peak, 50-step warmup.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)

from cua_shared import MODEL_DIR, QWEN_TOOLS, SYS_PROMPT, build_messages


# --------------------------------------------------------------------------- #
# Dataset                                                                     #
# --------------------------------------------------------------------------- #


@dataclass
class TrainExample:
    image_path: str
    instruction: str
    target: str


class InjectionTrainDataset(Dataset):
    """Reads JSONL and yields (messages, target_text, image)."""

    def __init__(self, jsonl_path: str | Path, root_dir: str | Path | None = None):
        self.root = Path(root_dir) if root_dir else Path(jsonl_path).parent
        self.examples: list[TrainExample] = []
        with open(jsonl_path) as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                row = json.loads(ln)
                self.examples.append(
                    TrainExample(
                        image_path=row["image_path"],
                        instruction=row["instruction"],
                        target=row["target"],
                    )
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        img_path = (self.root / ex.image_path).resolve()
        img = Image.open(img_path).convert("RGB")
        messages = build_messages(ex.instruction, with_system=True)
        return {"messages": messages, "target": ex.target, "image": img}


# --------------------------------------------------------------------------- #
# Collator                                                                    #
# --------------------------------------------------------------------------- #


class MultimodalCollator:
    """Builds a batch by tokenizing prompt + target, masking prompt tokens
    in labels so loss is only computed on the assistant's tool_call output."""

    def __init__(self, processor):
        self.processor = processor
        # End-of-message marker we expect from assistant turns.
        self.eos_id = processor.tokenizer.eos_token_id

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        prompt_texts: list[str] = []
        full_texts: list[str] = []
        images: list[Any] = []
        for ex in batch:
            prompt = self.processor.apply_chat_template(
                ex["messages"], tools=QWEN_TOOLS, add_generation_prompt=True, tokenize=False
            )
            full = prompt + ex["target"] + self.processor.tokenizer.eos_token
            prompt_texts.append(prompt)
            full_texts.append(full)
            images.append(ex["image"])

        # Encode the FULL conversation as the input.
        enc = self.processor(
            text=full_texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        # Encode prompts alone to know where to mask labels.
        prompt_enc = self.processor.tokenizer(
            prompt_texts, return_tensors="pt", padding=True, add_special_tokens=False
        )

        labels = enc["input_ids"].clone()
        for i in range(len(batch)):
            # Mask out the prompt span so we only learn the target.
            prompt_len = (prompt_enc["input_ids"][i] != self.processor.tokenizer.pad_token_id).sum().item()
            labels[i, :prompt_len] = -100
            # Also mask pad in the full sequence.
            pad_mask = enc["input_ids"][i] == self.processor.tokenizer.pad_token_id
            labels[i, pad_mask] = -100
        enc["labels"] = labels
        return dict(enc)


# --------------------------------------------------------------------------- #
# LoRA targeting                                                              #
# --------------------------------------------------------------------------- #


def find_lora_targets(model) -> list[str]:
    """Return language-model linear projection module names. Skip vision tower."""
    targets: set[str] = set()
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            if "visual" in name or "vision" in name:
                continue
            short = name.split(".")[-1]
            if short in {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}:
                targets.add(short)
    return sorted(targets)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Path to training JSONL")
    ap.add_argument("--root", default=None, help="Root dir for image_path resolution (default: dirname of --train)")
    ap.add_argument("--out", default="outputs/lora-r16")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--per-device-batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=-1)
    args = ap.parse_args()

    print(f"[train] loading model from {MODEL_DIR}", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_DIR)

    targets = find_lora_targets(model)
    print(f"[train] LoRA target modules: {targets}", flush=True)
    if not targets:
        raise RuntimeError("No LoRA target modules found - inspect model.named_modules()")

    lora_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        target_modules=targets,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    ds = InjectionTrainDataset(args.train, args.root)
    print(f"[train] dataset size: {len(ds)}", flush=True)

    collator = MultimodalCollator(processor)

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=False,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=collator,
    )

    print(f"[train] starting training. output -> {args.out}", flush=True)
    trainer.train()
    trainer.save_model(args.out)
    print(f"[train] done. adapter saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
