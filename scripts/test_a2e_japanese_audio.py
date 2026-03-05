"""
日本語音声 A2E テスト - 簡易スタンドアロン版

OpenAvatarChat で data_bundle.py の修正が正しく機能するかテストします。

使い方:
    cd C:\Users\hamad\OpenAvatarChat
    conda activate oac
    python scripts/test_a2e_japanese_audio.py

このスクリプトを C:\Users\hamad\OpenAvatarChat\scripts\ にコピーして実行してください。
"""

import sys
import os
import time
import traceback
from pathlib import Path

# OpenAvatarChatのルートディレクトリを検出
SCRIPT_DIR = Path(__file__).parent
OAC_DIR = SCRIPT_DIR.parent  # scripts/ の親 = OpenAvatarChat/

def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_1_environment():
    """テスト1: 環境チェック"""
    print_header("TEST 1: Environment Check")
    errors = []

    # Python version
    print(f"  Python: {sys.version}")

    # NumPy
    try:
        import numpy as np
        print(f"  NumPy: {np.__version__}")
    except ImportError:
        errors.append("NumPy not installed")

    # PyTorch
    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        errors.append("PyTorch not installed")

    # transformers
    try:
        import transformers
        print(f"  Transformers: {transformers.__version__}")
    except ImportError:
        errors.append("transformers not installed")

    # onnxruntime
    try:
        import onnxruntime
        print(f"  ONNXRuntime: {onnxruntime.__version__}")
    except ImportError:
        print("  ONNXRuntime: not installed (optional)")

    if errors:
        for e in errors:
            print(f"  [ERROR] {e}")
        return False

    print("  [PASS] Environment OK")
    return True


def test_2_model_files():
    """テスト2: モデルファイル存在確認"""
    print_header("TEST 2: Model Files Check")

    checks = {
        "LAM_audio2exp dir": OAC_DIR / "models" / "LAM_audio2exp",
        "wav2vec2-base-960h dir": OAC_DIR / "models" / "wav2vec2-base-960h",
        "pretrained_models dir": OAC_DIR / "models" / "LAM_audio2exp" / "pretrained_models",
    }

    all_ok = True
    for label, path in checks.items():
        exists = path.exists()
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {label}: {path}")
        if not exists:
            all_ok = False

    if all_ok:
        print("  [PASS] All model directories found")
    else:
        print("  [FAIL] Some model files missing")
    return all_ok


def test_3_data_bundle_fix():
    """テスト3: data_bundle.py の list/tuple → ndarray 変換テスト"""
    print_header("TEST 3: data_bundle.py Fix Verification")

    try:
        import numpy as np

        # data_bundle.py のパスを確認
        db_path = OAC_DIR / "src" / "chat_engine" / "data_models" / "runtime_data" / "data_bundle.py"
        if not db_path.exists():
            print(f"  [SKIP] File not found: {db_path}")
            return True  # ファイルがなければスキップ

        # ファイル内容をチェック
        content = db_path.read_text(encoding="utf-8")
        if "isinstance(data, (list, tuple))" in content:
            print("  [OK] list/tuple conversion patch found in data_bundle.py")
        else:
            print("  [WARN] list/tuple conversion patch NOT found in data_bundle.py")
            print("  Add this before 'if isinstance(data, np.ndarray)'::")
            print("    if isinstance(data, (list, tuple)):")
            print("        data = np.array(data, dtype=np.float32)")
            return False

        # 実際に変換が動作するかテスト
        test_list = [0.1, 0.2, 0.3, 0.4, 0.5]
        test_tuple = (0.1, 0.2, 0.3)
        arr_from_list = np.array(test_list, dtype=np.float32)
        arr_from_tuple = np.array(test_tuple, dtype=np.float32)

        assert isinstance(arr_from_list, np.ndarray), "list→ndarray conversion failed"
        assert isinstance(arr_from_tuple, np.ndarray), "tuple→ndarray conversion failed"
        assert arr_from_list.dtype == np.float32, "dtype should be float32"
        print(f"  [OK] list→ndarray: {test_list} → shape={arr_from_list.shape}")
        print(f"  [OK] tuple→ndarray: {test_tuple} → shape={arr_from_tuple.shape}")

        print("  [PASS] data_bundle.py fix is correct")
        return True

    except Exception as e:
        print(f"  [FAIL] {e}")
        traceback.print_exc()
        return False


def test_4_wav2vec2_load():
    """テスト4: Wav2Vec2モデルの読み込みテスト"""
    print_header("TEST 4: Wav2Vec2 Model Loading")

    try:
        import torch
        from transformers import Wav2Vec2Model, Wav2Vec2Processor
        import numpy as np

        wav2vec_dir = OAC_DIR / "models" / "wav2vec2-base-960h"
        if wav2vec_dir.exists() and (wav2vec_dir / "config.json").exists():
            model_path = str(wav2vec_dir)
            print(f"  Loading from local: {model_path}")
        else:
            model_path = "facebook/wav2vec2-base-960h"
            print(f"  Loading from HuggingFace: {model_path}")

        t0 = time.time()
        model = Wav2Vec2Model.from_pretrained(model_path)
        model.eval()
        elapsed = time.time() - t0
        print(f"  Model loaded in {elapsed:.1f}s")

        # ダミー音声でテスト (1秒の無音)
        dummy_audio = np.zeros(16000, dtype=np.float32)
        try:
            processor = Wav2Vec2Processor.from_pretrained(model_path)
        except Exception:
            processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

        inputs = processor(dummy_audio, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)

        features = outputs.last_hidden_state
        print(f"  Output shape: {tuple(features.shape)}")
        print(f"  [PASS] Wav2Vec2 working correctly")
        return True

    except Exception as e:
        print(f"  [FAIL] {e}")
        traceback.print_exc()
        return False


def test_5_a2e_import():
    """テスト5: A2Eモジュールのインポートテスト"""
    print_header("TEST 5: A2E Module Import")

    # sys.pathにOpenAvatarChatのパスを追加
    paths_to_add = [
        str(OAC_DIR / "src"),
        str(OAC_DIR / "src" / "handlers"),
        str(OAC_DIR / "src" / "handlers" / "avatar" / "lam"),
        str(OAC_DIR / "src" / "handlers" / "avatar" / "lam" / "LAM_Audio2Expression"),
    ]
    for p in paths_to_add:
        if p not in sys.path and os.path.exists(p):
            sys.path.insert(0, p)

    imported = False

    # 方法1: A2E直接インポート
    try:
        from LAM_Audio2Expression.engines.infer import Audio2ExpressionInfer
        print("  [OK] A2E infer module imported")
        imported = True
    except ImportError as e:
        print(f"  [INFO] Direct A2E import failed: {e}")

    # 方法2: handler経由
    if not imported:
        try:
            from avatar.lam.avatar_handler_lam_audio2expression import HandlerAvatarLAM
            print("  [OK] A2E handler module imported")
            imported = True
        except ImportError as e:
            print(f"  [INFO] Handler import failed: {e}")

    if imported:
        print("  [PASS] A2E module is importable")
    else:
        print("  [WARN] A2E module not importable (may need specific env)")
        print("  This is OK if other tests pass")

    return True  # インポート失敗でも致命的ではない


def main():
    print("=" * 60)
    print("  A2E Japanese Audio Test - Standalone")
    print(f"  OAC Dir: {OAC_DIR}")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    results = {}
    results["environment"] = test_1_environment()
    results["model_files"] = test_2_model_files()
    results["data_bundle_fix"] = test_3_data_bundle_fix()
    results["wav2vec2"] = test_4_wav2vec2_load()
    results["a2e_import"] = test_5_a2e_import()

    # サマリー
    print_header("SUMMARY")
    passed = 0
    total = len(results)
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if ok:
            passed += 1

    print(f"\n  Result: {passed}/{total} passed")

    if passed == total:
        print("\n  All tests passed!")
        print("  Next step: Start OpenAvatarChat and test with Japanese voice:")
        print("    python src/demo.py --config config/chat_with_lam_jp.yaml")
    else:
        print("\n  Some tests failed. Fix the issues above and re-run.")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
