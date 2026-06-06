#!/usr/bin/env python3
"""Quick sanity checks before training / evaluation (no GPU required)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def ok(msg: str) -> None:
    print(f"  OK  {msg}")


def fail(msg: str) -> None:
    print(f"FAIL  {msg}", file=sys.stderr)


def run_py(args: list[str]) -> int:
    return subprocess.call([sys.executable, *args], cwd=ROOT)


def main() -> int:
    errors = 0
    print("PersonaSteer smoke check\n")

    if run_py(["Datasets/generate_profile.py", "--help"]) != 0:
        fail("generate_profile.py --help")
        errors += 1
    else:
        ok("generate_profile.py --help")

    if run_py(["Eval/llm_judge_eval.py", "--help"]) != 0:
        fail("llm_judge_eval.py --help")
        errors += 1
    else:
        ok("llm_judge_eval.py --help")

    if run_py(["Eval/eval_profile_steer.py", "--help"]) != 0:
        fail("eval_profile_steer.py --help (pip install -e LlamaFactory if import errors)")
        errors += 1
    else:
        ok("eval_profile_steer.py --help")

    lf_src = ROOT / "LlamaFactory" / "src" / "llamafactory" / "train" / "persona_profile_steering"
    if not lf_src.is_dir():
        fail(f"missing {lf_src}")
        errors += 1
    else:
        ok("persona_profile_steering training code present")

    train_yaml = ROOT / "LlamaFactory" / "examples" / "train_profile_persona_steering" / "custom_persona_steering.yaml"
    if train_yaml.is_file():
        ok("custom_persona_steering.yaml present")
    else:
        fail("missing custom_persona_steering.yaml")
        errors += 1

    try:
        sys.path.insert(0, str(ROOT / "Eval"))
        from eval_text_metrics import evaluate_data  # noqa: WPS433

        sample = [{"generated_text": "hello world", "output": "hello there"}]
        evaluate_data(sample, "smoke")
        ok("eval_text_metrics.evaluate_data")
    except Exception as exc:
        fail(f"eval_text_metrics: {exc}")
        errors += 1

    print()
    if errors:
        print(f"Smoke check finished with {errors} error(s).")
        print("Install deps: pip install -r requirements.txt && pip install -r requirements-eval.txt")
        return 1
    print("Smoke check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
