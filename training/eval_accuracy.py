# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause


"""
Evaluate SFT/GRPO checkpoint accuracy by generating responses and checking answers.

Usage:
    python eval_accuracy.py <checkpoint_path> <val_jsonl> [--num-samples 200] [--batch-size 8]

Loads a checkpoint, generates CoT responses on val data, extracts the final answer,
and computes exact-match accuracy against ground truth.
"""

import argparse
import json
import re
import sys
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image
from qwen_vl_utils import process_vision_info


def load_model(checkpoint_path, device="cuda"):
    """Load model from SFT checkpoint or HF model name."""
    print(f"Loading model from {checkpoint_path}...")
    processor = AutoProcessor.from_pretrained(checkpoint_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    print(f"Model loaded on {device}")
    return model, processor


def load_val_data(val_jsonl, num_samples=None):
    """Load val examples from JSONL."""
    examples = []
    with open(val_jsonl) as f:
        for line in f:
            examples.append(json.loads(line.strip()))
            if num_samples and len(examples) >= num_samples:
                break
    print(f"Loaded {len(examples)} val examples")
    return examples


def extract_final_answer(text):
    """Extract the final number from a CoT response.

    Looks for patterns like '= 31.' or '= 31' at the end.
    Falls back to extracting the last number in the text.
    """
    # Pattern: "= <number>." at end of CoT
    match = re.search(r'=\s*(\d+)\s*\.?\s*$', text.strip())
    if match:
        return match.group(1)
    # Fallback: last number in text
    numbers = re.findall(r'\d+', text)
    return numbers[-1] if numbers else ""


def evaluate(model, processor, examples, batch_size=4, max_new_tokens=512):
    """Generate responses and compute accuracy."""
    correct = 0
    total = 0
    results = []

    for i in range(0, len(examples), batch_size):
        batch = examples[i:i + batch_size]

        for ex in batch:
            image = Image.open(ex["image_path"]).convert("RGB")
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": ex["question"]},
                ]}
            ]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                )

            # Decode only the generated part
            generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
            response = processor.decode(generated_ids, skip_special_tokens=True)

            # Extract answer and compare
            predicted = extract_final_answer(response)
            # Ground truth: for CoT data, answer is the full CoT; extract task_main from it
            gt_answer = extract_final_answer(ex["answer"])
            if not gt_answer:
                # Non-CoT data: answer is just the number
                gt_answer = ex["answer"].strip()

            is_correct = predicted == gt_answer
            correct += int(is_correct)
            total += 1

            results.append({
                "predicted": predicted,
                "ground_truth": gt_answer,
                "correct": is_correct,
                "response_preview": response[:200],
            })

            if total <= 5 or (not is_correct and total <= 20):
                status = "✓" if is_correct else "✗"
                print(f"  [{status}] pred={predicted}, gt={gt_answer}, response={response[:100]}...")

        acc = correct / total if total > 0 else 0
        print(f"  Progress: {total}/{len(examples)}, accuracy: {acc:.1%}")

    accuracy = correct / total if total > 0 else 0
    print(f"\n{'='*50}")
    print(f"ACCURACY: {correct}/{total} = {accuracy:.1%}")
    print(f"{'='*50}")

    return accuracy, results


def main():
    parser = argparse.ArgumentParser(description="Evaluate SFT checkpoint accuracy")
    parser.add_argument("checkpoint_path", help="Path to SFT checkpoint (HF format) or model name")
    parser.add_argument("val_jsonl", help="Path to val.jsonl")
    parser.add_argument("--num-samples", type=int, default=200, help="Number of val samples to evaluate")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for generation")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    model, processor = load_model(args.checkpoint_path, args.device)
    examples = load_val_data(args.val_jsonl, args.num_samples)
    accuracy, results = evaluate(
        model, processor, examples,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    # Save results
    output_path = "eval_results.json"
    with open(output_path, "w") as f:
        json.dump({"accuracy": accuracy, "num_samples": len(results), "results": results}, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
