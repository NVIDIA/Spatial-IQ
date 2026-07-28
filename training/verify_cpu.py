# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause


"""
CPU-only validation of the SpatialIQ training pipeline.

Tests imports, dataset loading, processor logic, config parsing,
and tokenization — everything except actual GPU training.

Usage (inside NeMo RL container):
    PYTHONPATH=/path/to/training:$PYTHONPATH uv run python verify_cpu.py
"""

import json
import os
import sys
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VERIFY_DATA = os.path.join(SCRIPT_DIR, "_verify_data")
TRAIN_JSONL = os.path.join(VERIFY_DATA, "train.jsonl")
VAL_JSONL = os.path.join(VERIFY_DATA, "val.jsonl")


def step(name):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")


def test_data_preparation():
    step("1. Data preparation")
    assert os.path.exists(TRAIN_JSONL), f"Missing {TRAIN_JSONL} — run prepare_data.py first"
    with open(TRAIN_JSONL) as f:
        lines = [json.loads(l) for l in f if l.strip()]
    print(f"  Train JSONL: {len(lines)} examples")
    ex = lines[0]
    assert "image_path" in ex, "Missing image_path"
    assert "question" in ex, "Missing question"
    assert "answer" in ex, "Missing answer"
    assert os.path.exists(ex["image_path"]), f"Image not found: {ex['image_path']}"
    print(f"  Sample: image_path={ex['image_path']}, answer={ex['answer']}")
    print("  PASSED")


def test_imports():
    step("2. NeMo RL imports")
    from nemo_rl.data.datasets.raw_dataset import RawDataset
    from nemo_rl.data.datasets.utils import pil_to_base64
    from nemo_rl.data.interfaces import DatumSpec, TaskDataSpec
    from nemo_rl.data.datasets import AllTaskProcessedDataset, load_response_dataset
    from nemo_rl.data.llm_message_utils import get_formatted_message_log
    from nemo_rl.data.multimodal_utils import PackedTensor, get_multimodal_keys_from_processor
    from nemo_rl.algorithms.sft import setup, sft_train
    from nemo_rl.algorithms.grpo import grpo_train
    from nemo_rl.environments.vlm_environment import VLMEnvironment
    from nemo_rl.utils.config import load_config, parse_hydra_overrides
    print("  All NeMo RL imports OK")
    print("  PASSED")


def test_spatialiq_imports():
    step("3. SpatialIQ module imports")
    from spatialiq_dataset import SpatialIQDataset, format_spatialiq_example
    print("  SpatialIQDataset, format_spatialiq_example imported OK")
    print("  PASSED")


def test_dataset_loading():
    step("4. Dataset loading")
    from spatialiq_dataset import SpatialIQDataset
    ds = SpatialIQDataset(train_path=TRAIN_JSONL, val_path=VAL_JSONL)
    assert ds.task_name == "spatialiq"
    assert ds.formatted_ds["train"] is not None
    assert ds.formatted_ds["validation"] is not None
    print(f"  Train: {len(ds.formatted_ds['train'])} examples")
    print(f"  Val: {len(ds.formatted_ds['validation'])} examples")
    print(f"  Columns: {ds.formatted_ds['train'].column_names}")
    print("  PASSED")


def test_format_function():
    step("5. Format function (image loading + base64 + message structure)")
    from spatialiq_dataset import format_spatialiq_example
    with open(TRAIN_JSONL) as f:
        ex = json.loads(f.readline())

    formatted = format_spatialiq_example(ex)
    assert "messages" in formatted
    assert len(formatted["messages"]) == 2
    assert formatted["messages"][0]["role"] == "user"
    assert formatted["messages"][1]["role"] == "assistant"

    user_content = formatted["messages"][0]["content"]
    assert isinstance(user_content, list)
    assert user_content[0]["type"] == "image"
    assert user_content[0]["image"].startswith("data:image/png;base64,")
    assert user_content[1]["type"] == "text"
    assert "block" in user_content[1]["text"].lower()

    answer = formatted["messages"][1]["content"]
    assert answer == ex["answer"]
    print(f"  User content: image (base64, {len(user_content[0]['image'])} chars) + text")
    print(f"  Assistant content: '{answer}'")
    print(f"  task_name: {formatted['task_name']}")
    print("  PASSED")


def test_format_pil():
    step("6. Format function (return_pil=True for SFT)")
    from spatialiq_dataset import format_spatialiq_example
    from PIL import Image
    with open(TRAIN_JSONL) as f:
        ex = json.loads(f.readline())

    formatted = format_spatialiq_example(ex, return_pil=True)
    user_content = formatted["messages"][0]["content"]
    assert isinstance(user_content[0]["image"], Image.Image)
    print(f"  Image type: {type(user_content[0]['image'])}")
    print(f"  Image size: {user_content[0]['image'].size}")
    print("  PASSED")


def test_config_loading():
    step("7. Config loading and parsing")
    from omegaconf import OmegaConf
    from nemo_rl.utils.config import load_config, parse_hydra_overrides
    OmegaConf.register_new_resolver("mul", lambda a, b: a * b, replace=True)
    OmegaConf.register_new_resolver("max", lambda a, b: max(a, b), replace=True)

    sft_cfg = load_config(os.path.join(SCRIPT_DIR, "configs", "sft_spatialiq.yaml"))
    sft_cfg = parse_hydra_overrides(sft_cfg, [
        f"data.train_data_path={TRAIN_JSONL}",
        f"data.val_data_path={VAL_JSONL}",
    ])
    sft_dict = OmegaConf.to_container(sft_cfg, resolve=True)
    assert sft_dict["data"]["train_data_path"] == TRAIN_JSONL
    assert sft_dict["policy"]["model_name"] == "Qwen/Qwen2.5-VL-3B-Instruct"
    print(f"  SFT config: model={sft_dict['policy']['model_name']}, steps={sft_dict['sft']['max_num_steps']}")

    grpo_cfg = load_config(os.path.join(SCRIPT_DIR, "configs", "grpo_spatialiq.yaml"))
    grpo_cfg = parse_hydra_overrides(grpo_cfg, [
        f"data.train_data_path={TRAIN_JSONL}",
        f"data.val_data_path={VAL_JSONL}",
    ])
    grpo_dict = OmegaConf.to_container(grpo_cfg, resolve=True)
    assert grpo_dict["data"]["env_name"] == "spatialiq"
    assert grpo_dict["env"]["spatialiq"]["reward_functions"][1]["name"] == "exact_alnum"
    print(f"  GRPO config: env={grpo_dict['data']['env_name']}, reward={grpo_dict['env']['spatialiq']['reward_functions']}")
    print("  PASSED")


def test_task_spec():
    step("8. TaskDataSpec creation")
    from spatialiq_dataset import SpatialIQDataset
    from nemo_rl.data.interfaces import TaskDataSpec

    ds = SpatialIQDataset(train_path=TRAIN_JSONL, val_path=VAL_JSONL)
    ds.set_task_spec({
        "dataset_name": "spatialiq",
        "prompt_file": None,
        "system_prompt_file": None,
    })
    assert ds.task_spec is not None
    assert ds.task_spec.task_name == "spatialiq"
    print(f"  task_name: {ds.task_spec.task_name}")
    print(f"  prompt: {ds.task_spec.prompt}")
    print("  PASSED")


def test_tokenizer_loading():
    step("9. Tokenizer / Processor loading (Qwen2.5-VL-3B)")
    from nemo_rl.algorithms.utils import get_tokenizer
    processor = get_tokenizer({"name": "Qwen/Qwen2.5-VL-3B-Instruct"}, get_processor=True)
    tokenizer = processor.tokenizer
    print(f"  Processor type: {type(processor).__name__}")
    print(f"  Tokenizer type: {type(tokenizer).__name__}")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print("  PASSED")
    return processor


def test_sft_processor(processor):
    step("10. SFT preprocessor (tokenization of formatted data)")
    from functools import partial
    from spatialiq_dataset import format_spatialiq_example
    from nemo_rl.data.interfaces import TaskDataSpec
    from nemo_rl.data.llm_message_utils import get_formatted_message_log

    with open(TRAIN_JSONL) as f:
        ex = json.loads(f.readline())

    formatted = format_spatialiq_example(ex, return_pil=True)
    task_spec = TaskDataSpec(task_name="spatialiq")

    message_log = get_formatted_message_log(
        formatted["messages"],
        processor,
        task_spec,
        add_bos_token=True,
        add_eos_token=True,
        add_generation_prompt=False,
    )
    total_tokens = sum(len(m["token_ids"]) for m in message_log)
    print(f"  Messages: {len(message_log)} turns")
    print(f"  Total tokens: {total_tokens}")
    for i, m in enumerate(message_log):
        print(f"    Turn {i}: role={m['role']}, tokens={len(m['token_ids'])}")
    assert total_tokens > 0, "No tokens produced"
    print("  PASSED")


def test_grpo_processor(processor):
    step("11. GRPO VLM processor (hf_data_processor)")
    # Import our local hf_data_processor from run_grpo
    sys.path.insert(0, SCRIPT_DIR)
    import importlib
    run_grpo = importlib.import_module("run_grpo")

    with open(TRAIN_JSONL) as f:
        ex = json.loads(f.readline())

    from nemo_rl.data.interfaces import TaskDataSpec
    task_spec = TaskDataSpec(task_name="spatialiq")

    datum_spec = run_grpo.hf_data_processor(
        ex, task_spec, processor, max_seq_length=512, idx=0
    )

    assert "message_log" in datum_spec
    assert "extra_env_info" in datum_spec
    assert "vllm_content" in datum_spec
    assert "vllm_images" in datum_spec
    assert datum_spec["extra_env_info"]["ground_truth"] == ex["answer"]
    assert datum_spec["task_name"] == "spatialiq"
    user_turn = datum_spec["message_log"][0]
    assert user_turn["role"] == "user"

    expected_user_message = run_grpo.format_spatialiq_example(ex)["messages"][0]
    if hasattr(processor, "conversation_preprocessor"):
        expected_user_message = processor.conversation_preprocessor(expected_user_message)
    expected_tokenized = processor.apply_chat_template(
        [expected_user_message],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    if "mm_token_type_ids" in expected_tokenized:
        assert "mm_token_type_ids" in user_turn
        assert user_turn["mm_token_type_ids"].tolist() == (
            expected_tokenized["mm_token_type_ids"][0].tolist()
        )

    length = datum_spec["length"]
    n_images = len(datum_spec["vllm_images"])
    print(f"  Token length: {length}")
    print(f"  Images: {n_images}")
    print(f"  Ground truth: {datum_spec['extra_env_info']['ground_truth']}")
    print(f"  Loss multiplier: {datum_spec['loss_multiplier']}")
    print(f"  vllm_content length: {len(datum_spec['vllm_content']) if datum_spec['vllm_content'] else 0}")
    print("  PASSED")


def main():
    print("SpatialIQ Training Pipeline — CPU Validation")
    print(f"Script dir: {SCRIPT_DIR}")

    tests = [
        ("Data preparation", test_data_preparation),
        ("NeMo RL imports", test_imports),
        ("SpatialIQ imports", test_spatialiq_imports),
        ("Dataset loading", test_dataset_loading),
        ("Format function", test_format_function),
        ("Format PIL", test_format_pil),
        ("Config loading", test_config_loading),
        ("TaskDataSpec", test_task_spec),
    ]

    passed = 0
    failed = 0
    processor = None

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            failed += 1

    # Tokenizer-dependent tests (may fail if model download is slow)
    try:
        processor = test_tokenizer_loading()
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        failed += 1

    if processor is not None:
        for name, test_fn in [
            ("SFT preprocessor", lambda: test_sft_processor(processor)),
            ("GRPO processor", lambda: test_grpo_processor(processor)),
        ]:
            try:
                test_fn()
                passed += 1
            except Exception as e:
                print(f"  FAILED: {e}")
                traceback.print_exc()
                failed += 1

    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed out of {passed+failed}")
    print(f"{'='*60}")
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
