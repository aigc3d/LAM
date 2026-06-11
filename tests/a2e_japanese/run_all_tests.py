"""
A2E + 日本語音声テスト: マスターテストランナー

全テストを順番に実行:
  Step 0: 環境チェック (setup_oac_env.py)
  Step 1: テスト音声生成 (generate_test_audio.py)
  Step 2: A2Eテスト (test_a2e_cpu.py)
  Step 3: ブレンドシェイプ分析 (analyze_blendshapes.py) ※推論結果がある場合

使い方:
    cd C:\Users\hamad\OpenAvatarChat
    conda activate oac
    python tests/a2e_japanese/run_all_tests.py

    または:
    python tests/a2e_japanese/run_all_tests.py --oac-dir C:\Users\hamad\OpenAvatarChat
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def run_step(step_name: str, script_path: str, extra_args: list = None):
    """テストステップを実行"""
    print(f"\n{'#' * 60}")
    print(f"# {step_name}")
    print(f"{'#' * 60}\n")

    if not os.path.exists(script_path):
        print(f"  ERROR: Script not found: {script_path}")
        return False

    cmd = [sys.executable, script_path] + (extra_args or [])
    t0 = time.time()

    try:
        result = subprocess.run(cmd, timeout=300)
        elapsed = time.time() - t0
        success = result.returncode == 0
        status = "PASSED" if success else "FAILED"
        print(f"\n  [{status}] {step_name} ({elapsed:.1f}s)")
        return success
    except subprocess.TimeoutExpired:
        print(f"\n  [TIMEOUT] {step_name} (>300s)")
        return False
    except Exception as e:
        print(f"\n  [ERROR] {step_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="A2E Japanese Audio Test Runner")
    parser.add_argument("--oac-dir", type=str, default=None,
                        help="Path to OpenAvatarChat directory")
    parser.add_argument("--skip-env-check", action="store_true",
                        help="Skip environment check")
    parser.add_argument("--skip-audio-gen", action="store_true",
                        help="Skip audio generation (use existing)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    oac_args = ["--oac-dir", args.oac_dir] if args.oac_dir else []

    print("=" * 60)
    print("A2E + Japanese Audio Test Suite - Master Runner")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    results = {}

    # Step 0: 環境チェック
    if not args.skip_env_check:
        results["env_check"] = run_step(
            "Step 0: Environment Check",
            str(script_dir / "setup_oac_env.py"),
            oac_args,
        )
    else:
        print("\n  [SKIP] Environment check")
        results["env_check"] = True

    # Step 1: テスト音声生成
    if not args.skip_audio_gen:
        results["audio_gen"] = run_step(
            "Step 1: Generate Test Audio",
            str(script_dir / "generate_test_audio.py"),
        )
    else:
        print("\n  [SKIP] Audio generation")
        results["audio_gen"] = True

    # Step 2: A2Eテスト
    results["a2e_test"] = run_step(
        "Step 2: A2E Inference Test",
        str(script_dir / "test_a2e_cpu.py"),
        oac_args,
    )

    # Step 3: ブレンドシェイプ分析
    output_dir = script_dir / "blendshape_outputs"
    if output_dir.exists() and list(output_dir.glob("*.npy")):
        results["analysis"] = run_step(
            "Step 3: Blendshape Analysis",
            str(script_dir / "analyze_blendshapes.py"),
            ["--input-dir", str(output_dir), "--export-csv", "--export-json"],
        )
    else:
        print(f"\n  [SKIP] Step 3: No blendshape outputs in {output_dir}")
        print("  Run full A2E inference and save outputs there first.")
        results["analysis"] = None

    # サマリー
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    for name, passed in results.items():
        if passed is None:
            status = "SKIP"
        elif passed:
            status = "PASS"
        else:
            status = "FAIL"
        print(f"  [{status}] {name}")

    failed = sum(1 for v in results.values() if v is False)
    if failed:
        print(f"\n  {failed} step(s) failed.")
        print("\n  Troubleshooting:")
        print("  1. Run setup_oac_env.py to check environment")
        print("  2. Ensure all models are downloaded")
        print("  3. For GPU errors, patch infer.py: .cuda() -> .cpu()")
        return 1
    else:
        print("\n  All steps completed!")
        print("\n  Next: Start OpenAvatarChat and test lip sync quality")
        print("    cd C:\\Users\\hamad\\OpenAvatarChat")
        print("    python src/demo.py --config config/chat_with_lam_jp.yaml")
        print("    Open https://localhost:8282 and speak Japanese")
        return 0


if __name__ == "__main__":
    sys.exit(main())
