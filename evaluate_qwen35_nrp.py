#!/usr/bin/env python3
"""
Evaluate SAMA questions with Qwen3.5 through the NRP OpenAI-compatible API.

Input:
    resort_questions.json
    A directory containing the map images referenced by image_filename.

Outputs:
    <output>.jsonl            Successful model responses, one per question.
    <output>.errors.jsonl       Failed request attempts.
    <output>.run_config.json    Reproducibility metadata.

The script is resumable:
    - Questions already present in the successful JSONL file are skipped.
    - Failed questions are retried when the script is run again.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from openai import OpenAI


DEFAULT_BASE_URL = "https://ellm.nrp-nautilus.io/v1"
DEFAULT_MODEL = "Qwen/Qwen3.5-397B-A17B-FP8"

# This is the exact benchmark system prompt sent to the evaluated model.
SYSTEM_PROMPT = """You are a visual question-answering model evaluating attraction maps.

You will receive one map image and one natural-language question about that map. Answer the question using only information visible in the provided map image.

Rules:
- Provide a short and direct natural-language answer.
- Answer with only the final answer. Do not explain your reasoning.
- Do not mention uncertainty unless the answer cannot be confirmed from the map.
- Use map labels exactly when possible.
- For yes/no questions, begin with "Yes" or "No."
- For counting questions, answer with the number.
- If the requested information is not shown or cannot be confirmed from the map, answer: "Cannot be confirmed from the map alone."

Direction rules:
- For direction questions, use map-based directions such as north, south, east, west, northeast, northwest, southeast, or southwest when applicable.
- First check whether the map shows a north arrow, compass, or other orientation indicator.
- If an orientation indicator is shown, use that map-defined orientation.
- If no orientation indicator is shown, treat the top of the image as north.
- Do not assume real-world geographic north unless explicitly indicated by the map.

Route, nearest, and location rules:
- Use only visible or clearly indicated valid routes, such as roads, paths, hallways, bridges, stairs, entrances, exits, or connected indoor passages.
- Do not invent shortcuts, doors, tunnels, paths, or access points.
- Do not cross walls, barriers, rivers, fences, restricted areas, or disconnected spaces unless a valid crossing is shown.
- Entering a building and exiting from another side is valid only if the map clearly supports that connection.
- If the question asks for the nearest place, facility, icon, or landmark, judge nearest by the shortest visible valid route, not by straight-line distance.
- If something appears close but is not reachable by a visible valid route, do not treat it as the nearest reachable answer.
- For location-based answers, identify the place using visible map-grounded references when helpful, such as its name, nearby landmarks, adjacent paths, zone, region, or relative direction."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_line(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def read_completed_question_ids(path: Path) -> Set[str]:
    completed: Set[str] = set()

    if not path.exists():
        return completed

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}: {error}"
                ) from error

            question_id = record.get("question_id")
            if question_id:
                completed.add(str(question_id))

    return completed


def load_dataset(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)

    questions = dataset.get("questions")
    if not isinstance(questions, list):
        raise ValueError("Dataset must contain a top-level 'questions' list.")

    declared_count = dataset.get("question_count")
    if declared_count is not None and declared_count != len(questions):
        raise ValueError(
            f"question_count is {declared_count}, but the file contains "
            f"{len(questions)} question objects."
        )

    required_fields = {
        "question_id",
        "image_id",
        "image_filename",
        "question",
    }

    seen_ids: Set[str] = set()

    for index, question in enumerate(questions, start=1):
        if not isinstance(question, dict):
            raise ValueError(f"Question {index} is not a JSON object.")

        missing = [
            field
            for field in required_fields
            if not str(question.get(field, "")).strip()
        ]
        if missing:
            raise ValueError(
                f"Question {index} is missing required fields: {missing}"
            )

        question_id = str(question["question_id"])
        if question_id in seen_ids:
            raise ValueError(f"Duplicate question_id: {question_id}")
        seen_ids.add(question_id)

    return dataset


def image_to_data_url(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)

    if mime_type not in {
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/bmp",
        "image/tiff",
    }:
        raise ValueError(
            f"Unsupported or unknown image format for {path.name}: {mime_type}"
        )

    image_bytes = path.read_bytes()
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def text_content(value: Any) -> str:
    """
    Preserve a normal string response exactly.
    Convert unusual structured content to JSON rather than discarding it.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def usage_value(usage: Any, name: str) -> Optional[int]:
    if usage is None:
        return None
    return getattr(usage, name, None)


def verify_model_available(
    client: OpenAI,
    model_id: str,
) -> None:
    response = client.models.list()
    model_ids = sorted(str(model.id) for model in response.data)

    if model_id in model_ids:
        print(f"Model confirmed: {model_id}")
        return

    related = [
        candidate
        for candidate in model_ids
        if "qwen3.5" in candidate.lower() or "397b" in candidate.lower()
    ]

    message = [
        f"The exact model ID was not returned by /v1/models: {model_id}"
    ]

    if related:
        message.append("Related available model IDs:")
        message.extend(f"  - {candidate}" for candidate in related)
    else:
        message.append(
            "Run the following command to inspect available model IDs:\n"
            'curl -H "Authorization: Bearer $OPENAI_API_KEY" '
            f"{DEFAULT_BASE_URL}/models"
        )

    raise RuntimeError("\n".join(message))


def build_run_config(
    args: argparse.Namespace,
    dataset: Dict[str, Any],
    dataset_hash: str,
    cache_salt_enabled: bool,
) -> Dict[str, Any]:
    return {
        "run_id": args.run_id,
        "dataset": dataset.get("dataset"),
        "dataset_category": dataset.get("category"),
        "dataset_question_count": len(dataset["questions"]),
        "dataset_file": str(args.dataset.resolve()),
        "dataset_sha256": dataset_hash,
        "image_directory": str(args.image_dir.resolve()),
        "provider": "NRP Envoy AI Gateway",
        "base_url": args.base_url,
        "requested_model_id": args.model,
        "prompt_version": args.prompt_version,
        "system_prompt": SYSTEM_PROMPT,
        "system_prompt_sha256": sha256_bytes(
            SYSTEM_PROMPT.encode("utf-8")
        ),
        "user_prompt_template": "Question: {QUESTION}\\n\\nAnswer:",
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "thinking_enabled": False,
        "number_of_outputs": 1,
        "response_format": "plain text",
        "timeout_seconds": args.timeout,
        "maximum_attempts_per_execution": args.max_attempts,
        "retry_backoff_seconds": args.retry_backoff,
        "concurrency": 1,
        "image_preprocessing": "None; original image bytes submitted as a base64 data URL.",
        "cache_salt_enabled": cache_salt_enabled,
        "created_at_utc": utc_now(),
    }


def validate_or_create_run_config(
    path: Path,
    config: Dict[str, Any],
) -> None:
    """
    Prevent accidentally resuming the same output file with different
    benchmark settings.
    """
    fields_that_must_match = [
        "run_id",
        "dataset_sha256",
        "requested_model_id",
        "prompt_version",
        "system_prompt_sha256",
        "temperature",
        "top_p",
        "max_tokens",
        "seed",
        "thinking_enabled",
        "image_preprocessing",
    ]

    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)

        differences = []
        for field in fields_that_must_match:
            if existing.get(field) != config.get(field):
                differences.append(
                    f"{field}: existing={existing.get(field)!r}, "
                    f"current={config.get(field)!r}"
                )

        if differences:
            raise RuntimeError(
                "The existing run configuration does not match this execution. "
                "Use a new output filename or restore the original settings:\n"
                + "\n".join(f"- {item}" for item in differences)
            )

        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def evaluate_question(
    client: OpenAI,
    args: argparse.Namespace,
    question_record: Dict[str, Any],
    image_path: Path,
    image_data_url: str,
    image_hash: str,
    cache_salt: Optional[str],
    error_path: Path,
) -> Optional[Dict[str, Any]]:
    question_id = str(question_record["question_id"])
    question_text = str(question_record["question"])
    user_prompt = f"Question: {question_text}\n\nAnswer:"

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_url,
                    },
                },
                {
                    "type": "text",
                    "text": user_prompt,
                },
            ],
        },
    ]

    extra_body: Dict[str, Any] = {
        # Qwen3.5 thinks by default. This benchmark requests only the
        # direct final answer, so non-thinking mode is explicitly enabled.
        "chat_template_kwargs": {
            "enable_thinking": False,
        },
    }

    if cache_salt:
        extra_body["cache_salt"] = cache_salt

    for attempt in range(1, args.max_attempts + 1):
        started = time.perf_counter()
        request_timestamp = utc_now()

        try:
            request_kwargs: Dict[str, Any] = {
                "model": args.model,
                "messages": messages,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
                "n": 1,
                "extra_body": extra_body,
            }

            if args.seed is not None:
                request_kwargs["seed"] = args.seed

            response = client.chat.completions.create(**request_kwargs)
            latency_ms = round(
                (time.perf_counter() - started) * 1000,
                3,
            )

            if not response.choices:
                raise RuntimeError("The API returned no completion choices.")

            choice = response.choices[0]
            message = choice.message
            response_text = text_content(message.content)

            if response_text == "":
                raise RuntimeError(
                    "The API returned an empty final response."
                )

            reasoning_content = getattr(
                message,
                "reasoning_content",
                None,
            )

            return {
                "run_id": args.run_id,
                "question_id": question_id,
                "image_id": question_record["image_id"],
                "image_filename": question_record["image_filename"],
                "image_sha256": image_hash,
                "question": question_text,
                "model_response": response_text,
                "requested_model_id": args.model,
                "returned_model_id": getattr(
                    response,
                    "model",
                    None,
                ),
                "provider": "NRP Envoy AI Gateway",
                "status": "success",
                "finish_reason": getattr(
                    choice,
                    "finish_reason",
                    None,
                ),
                "attempts_used": attempt,
                "latency_ms": latency_ms,
                "prompt_tokens": usage_value(
                    response.usage,
                    "prompt_tokens",
                ),
                "completion_tokens": usage_value(
                    response.usage,
                    "completion_tokens",
                ),
                "total_tokens": usage_value(
                    response.usage,
                    "total_tokens",
                ),
                "reasoning_content": text_content(
                    reasoning_content
                ) if reasoning_content is not None else None,
                "api_response_id": getattr(
                    response,
                    "id",
                    None,
                ),
                "api_request_id": getattr(
                    response,
                    "_request_id",
                    None,
                ),
                "request_timestamp_utc": request_timestamp,
                "completed_timestamp_utc": utc_now(),
            }

        except Exception as error:
            latency_ms = round(
                (time.perf_counter() - started) * 1000,
                3,
            )

            error_record = {
                "run_id": args.run_id,
                "question_id": question_id,
                "image_id": question_record["image_id"],
                "image_filename": question_record["image_filename"],
                "requested_model_id": args.model,
                "status": "error",
                "attempt": attempt,
                "maximum_attempts": args.max_attempts,
                "latency_ms": latency_ms,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "timestamp_utc": utc_now(),
            }
            write_json_line(error_path, error_record)

            print(
                f"  Attempt {attempt}/{args.max_attempts} failed: "
                f"{type(error).__name__}: {error}",
                file=sys.stderr,
            )

            if attempt < args.max_attempts:
                delay = args.retry_backoff * (2 ** (attempt - 1))
                time.sleep(delay)

    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a SAMA JSON dataset with Qwen3.5 through the "
            "NRP OpenAI-compatible API."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("resort_questions.json"),
        help="SAMA category JSON file.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("resort_images"),
        help="Directory containing the map images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "qwen3_5_397b_a17b_fp8_resort_run01.jsonl"
        ),
        help="Successful response JSONL file.",
    )
    parser.add_argument(
        "--run-id",
        default="qwen3_5_397b_a17b_fp8_resort_run01",
        help="Stable identifier for this evaluation run.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Exact model ID returned by the NRP /v1/models endpoint.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
    )
    parser.add_argument(
        "--prompt-version",
        default="sama-evaluation-v1.0",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Fixed at 0 by the SAMA benchmark protocol.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum final-answer output tokens.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Requested sampling seed. Its determinism depends on the "
            "serving backend."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=5.0,
        help="Initial retry delay; later retries use exponential backoff.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only this many pending questions for testing.",
    )
    parser.add_argument(
        "--skip-model-check",
        action="store_true",
        help="Skip checking the exact model ID through /v1/models.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "Error: OPENAI_API_KEY is not set.",
            file=sys.stderr,
        )
        return 1

    cache_salt = os.environ.get("NRP_CACHE_SALT")
    if not cache_salt:
        print(
            "Warning: NRP_CACHE_SALT is not set. Requests will not use "
            "a private cache namespace.",
            file=sys.stderr,
        )

    if not args.dataset.is_file():
        print(
            f"Error: dataset file not found: {args.dataset}",
            file=sys.stderr,
        )
        return 1

    if not args.image_dir.is_dir():
        print(
            f"Error: image directory not found: {args.image_dir}",
            file=sys.stderr,
        )
        return 1

    if args.max_tokens <= 0:
        print("Error: --max-tokens must be positive.", file=sys.stderr)
        return 1

    if args.max_attempts <= 0:
        print("Error: --max-attempts must be positive.", file=sys.stderr)
        return 1

    dataset = load_dataset(args.dataset)
    dataset_hash = sha256_file(args.dataset)

    output_path = args.output
    error_path = output_path.with_suffix(".errors.jsonl")
    config_path = output_path.with_suffix(".run_config.json")

    run_config = build_run_config(
        args=args,
        dataset=dataset,
        dataset_hash=dataset_hash,
        cache_salt_enabled=cache_salt is not None,
    )
    validate_or_create_run_config(config_path, run_config)

    client = OpenAI(
        api_key=api_key,
        base_url=args.base_url,
        timeout=args.timeout,
        max_retries=0,  # Retries are logged explicitly by this script.
    )

    if not args.skip_model_check:
        verify_model_available(client, args.model)

    completed_ids = read_completed_question_ids(output_path)
    questions = [
        question
        for question in dataset["questions"]
        if str(question["question_id"]) not in completed_ids
    ]

    if args.limit is not None:
        questions = questions[: args.limit]

    print(f"Dataset questions: {len(dataset['questions'])}")
    print(f"Already completed: {len(completed_ids)}")
    print(f"Pending in this execution: {len(questions)}")
    print(f"Successful responses: {output_path}")
    print(f"Error log: {error_path}")
    print(f"Run configuration: {config_path}")

    if not questions:
        print("No pending questions.")
        return 0

    # Cache encoded images and hashes in memory because many questions
    # refer to the same map.
    image_data_urls: Dict[Path, str] = {}
    image_hashes: Dict[Path, str] = {}

    successful_this_execution = 0
    failed_this_execution = 0

    for position, question_record in enumerate(questions, start=1):
        question_id = str(question_record["question_id"])
        image_filename = str(question_record["image_filename"])
        image_path = args.image_dir / image_filename

        print(
            f"[{position}/{len(questions)}] {question_id} "
            f"({image_filename})"
        )

        if not image_path.is_file():
            error_record = {
                "run_id": args.run_id,
                "question_id": question_id,
                "image_id": question_record["image_id"],
                "image_filename": image_filename,
                "requested_model_id": args.model,
                "status": "error",
                "attempt": 0,
                "maximum_attempts": args.max_attempts,
                "latency_ms": None,
                "error_type": "FileNotFoundError",
                "error_message": (
                    f"Image file not found: {image_path.resolve()}"
                ),
                "timestamp_utc": utc_now(),
            }
            write_json_line(error_path, error_record)
            print(
                f"  Missing image: {image_path}",
                file=sys.stderr,
            )
            failed_this_execution += 1
            continue

        resolved_image_path = image_path.resolve()

        if resolved_image_path not in image_data_urls:
            image_data_urls[resolved_image_path] = image_to_data_url(
                resolved_image_path
            )
            image_hashes[resolved_image_path] = sha256_file(
                resolved_image_path
            )

        result = evaluate_question(
            client=client,
            args=args,
            question_record=question_record,
            image_path=resolved_image_path,
            image_data_url=image_data_urls[resolved_image_path],
            image_hash=image_hashes[resolved_image_path],
            cache_salt=cache_salt,
            error_path=error_path,
        )

        if result is None:
            print("  Failed after all attempts.", file=sys.stderr)
            failed_this_execution += 1
            continue

        write_json_line(output_path, result)
        successful_this_execution += 1
        print(f"  Response: {result['model_response']!r}")

    print()
    print(f"Successful in this execution: {successful_this_execution}")
    print(f"Failed in this execution: {failed_this_execution}")
    print(
        "Run the same command again to skip successful questions and "
        "retry unresolved failures."
    )

    return 0 if failed_this_execution == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())