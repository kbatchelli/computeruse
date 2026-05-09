"""DPO fine-tune Northstar on cseval3-style prose preference pairs.

Differences from train_dpo.py:
- NO tools — model emits plain prose, matching cseval3 runtime mode.
- System prompt comes per-example from _meta.system_prompt (contains the secret).
- Chosen/rejected are plain prose strings (descriptions vs leaks).

Usage:
    python scripts/train_dpo_qa.py --train data/dpo_pairs_qa.jsonl --out outputs/dpo-qa-r16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from trl import DPOConfig, DPOTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cua_shared import MODEL_DIR  # noqa: E402

PCD = Path("/home/shadeform/computeruse/philip-cua-defense")


def build_qa_messages(system_text: str, user_text: str) -> list[dict]:
    """Cseval3-style prompt: system holds the secret, user asks about image."""
    return [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def assistant_message(text: str) -> list[dict]:
    return [{"role": "assistant", "content": [{"type": "text", "text": text}]}]


def load_pairs(jsonl_path: Path, root: Path) -> Dataset:
    rows = []
    for ln in open(jsonl_path):
        ln = ln.strip()
        if not ln:
            continue
        r = json.loads(ln)
        img_path = (root / r["image_path"]).resolve()
        rows.append({
            "images": [Image.open(img_path).convert("RGB")],
            "prompt": build_qa_messages(r["_meta"]["system_prompt"], r["prompt"]),
            "chosen": assistant_message(r["chosen"]),
            "rejected": assistant_message(r["rejected"]),
        })
    return Dataset.from_list(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--root", default=str(PCD / "data" / "dpo_qa"))
    ap.add_argument("--out", default=str(PCD / "outputs" / "dpo-qa-r16"))
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--per-device-batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--warmup-steps", type=int, default=20)
    ap.add_argument("--logging-steps", type=int, default=10)
    args = ap.parse_args()

    print(f"[dpo-qa] loading model from {MODEL_DIR}", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_DIR)

    print(f"[dpo-qa] loading pairs from {args.train}", flush=True)
    ds = load_pairs(Path(args.train), Path(args.root))
    print(f"[dpo-qa] dataset size: {len(ds)}", flush=True)

    lora_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    cfg = DPOConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        bf16=True,
        max_length=None,
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=cfg,
        train_dataset=ds,
        processing_class=processor,
        peft_config=lora_cfg,
    )

    print(f"[dpo-qa] starting training. output -> {args.out}", flush=True)
    trainer.train()
    trainer.save_model(args.out)
    print(f"[dpo-qa] done. adapter saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
