"""
ONNX RuntimeError 診断スクリプト

OpenAvatarChatで発生する以下のエラーの原因を特定する:
    RuntimeError: Input data type <class 'list'> is not supported.

このスクリプトは各ハンドラーのONNX関連処理を個別にテストし、
エラーの発生箇所を特定する。

使い方:
    cd C:\Users\hamad\OpenAvatarChat
    conda activate oac
    python tests/a2e_japanese/diagnose_onnx_error.py
"""

import os
import sys
import traceback
from pathlib import Path


def find_oac_dir() -> Path:
    candidates = [
        Path(r"C:\Users\hamad\OpenAvatarChat"),
        Path.home() / "OpenAvatarChat",
        Path.cwd(),
    ]
    for p in candidates:
        if (p / "src" / "handlers").exists():
            return p
    return Path.cwd()


def test_onnx_runtime_basic():
    """Test 1: ONNX Runtime の基本動作確認"""
    print("\n" + "=" * 60)
    print("TEST 1: ONNX Runtime Basic Check")
    print("=" * 60)

    try:
        import onnxruntime
        print(f"  onnxruntime version: {onnxruntime.__version__}")
        print(f"  Available providers: {onnxruntime.get_available_providers()}")
        print("  [PASS]")
        return True
    except ImportError:
        print("  [FAIL] onnxruntime not installed")
        return False


def test_silero_vad_onnx(oac_dir: Path):
    """Test 2: SileroVAD ONNX モデルのロードと推論テスト"""
    print("\n" + "=" * 60)
    print("TEST 2: SileroVAD ONNX Model")
    print("=" * 60)

    import onnxruntime
    import numpy as np

    # モデルファイルの検索
    model_candidates = [
        oac_dir / "src" / "handlers" / "vad" / "silerovad" / "silero_vad" / "src" / "silero_vad" / "data" / "silero_vad.onnx",
        oac_dir / "src" / "handlers" / "vad" / "silerovad" / "data" / "silero_vad.onnx",
    ]

    model_path = None
    for p in model_candidates:
        if p.exists():
            model_path = p
            break

    if model_path is None:
        # Recursive search
        for p in oac_dir.rglob("silero_vad.onnx"):
            model_path = p
            break

    if model_path is None:
        print("  [SKIP] silero_vad.onnx not found")
        return None

    print(f"  Model: {model_path}")

    # モデルロード
    try:
        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.log_severity_level = 4
        session = onnxruntime.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
            sess_options=options,
        )
        print("  Model loaded successfully")
    except Exception as e:
        print(f"  [FAIL] Model load error: {e}")
        return False

    # 入力/出力情報
    print("\n  Model inputs:")
    for inp in session.get_inputs():
        print(f"    {inp.name}: shape={inp.shape}, type={inp.type}")

    print("  Model outputs:")
    for out in session.get_outputs():
        print(f"    {out.name}: shape={out.shape}, type={out.type}")

    num_outputs = len(session.get_outputs())
    print(f"\n  Number of outputs: {num_outputs}")

    # テスト1: 正しい numpy 入力
    print("\n  --- Test 2a: Correct numpy inputs ---")
    try:
        clip = np.zeros((1, 512), dtype=np.float32)
        sr = np.array([16000], dtype=np.int64)
        state = np.zeros((2, 1, 128), dtype=np.float32)

        inputs = {"input": clip, "sr": sr, "state": state}
        print(f"    input: type={type(clip).__name__}, dtype={clip.dtype}, shape={clip.shape}")
        print(f"    sr: type={type(sr).__name__}, dtype={sr.dtype}, shape={sr.shape}")
        print(f"    state: type={type(state).__name__}, dtype={state.dtype}, shape={state.shape}")

        results = session.run(None, inputs)
        print(f"    Output count: {len(results)}")
        for i, r in enumerate(results):
            print(f"    output[{i}]: type={type(r).__name__}, dtype={r.dtype}, shape={r.shape}")

        # 出力数が2の場合のunpack確認
        if len(results) == 2:
            prob, new_state = results
            print(f"    Unpacked prob: type={type(prob).__name__}, value={prob}")
            print(f"    Unpacked state: type={type(new_state).__name__}, shape={new_state.shape}")
            print("    [PASS] 2-output unpack works correctly")
        elif len(results) == 3:
            print("    [WARN] Model has 3 outputs! VAD handler expects 2.")
            print("    This WILL cause 'too many values to unpack' error.")
            print("    FIX: Update _inference to handle 3 outputs")
        else:
            print(f"    [WARN] Unexpected output count: {len(results)}")

        # 2回目の推論（stateを再利用）
        if len(results) >= 2:
            new_state = results[1]
            inputs2 = {"input": clip, "sr": sr, "state": new_state}
            print(f"\n    Second inference with returned state:")
            print(f"    state type={type(new_state).__name__}, dtype={new_state.dtype}, shape={new_state.shape}")
            results2 = session.run(None, inputs2)
            print(f"    [PASS] Second inference succeeded")

    except Exception as e:
        print(f"    [FAIL] {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

    # テスト2: list 入力 → エラー再現
    print("\n  --- Test 2b: List input (reproduce error) ---")
    try:
        list_input = [0.0] * 512  # Python list instead of numpy array
        inputs_bad = {"input": list_input, "sr": sr, "state": state}
        results = session.run(None, inputs_bad)
        print("    [UNEXPECTED] No error with list input")
    except RuntimeError as e:
        if "list" in str(e).lower():
            print(f"    [CONFIRMED] Error reproduced: {e}")
            print("    This is the EXACT error from the logs.")
        else:
            print(f"    [FAIL] Different RuntimeError: {e}")
    except Exception as e:
        print(f"    [INFO] Different error type: {type(e).__name__}: {e}")

    # テスト3: state を list で渡す → エラー再現
    print("\n  --- Test 2c: State as list (reproduce error) ---")
    try:
        state_list = state.tolist()  # Convert numpy to nested list
        inputs_bad = {"input": clip, "sr": sr, "state": state_list}
        results = session.run(None, inputs_bad)
        print("    [UNEXPECTED] No error with list state")
    except RuntimeError as e:
        if "list" in str(e).lower():
            print(f"    [CONFIRMED] Error reproduced: {e}")
            print("    If model_state becomes a list, this error occurs.")
        else:
            print(f"    [FAIL] Different RuntimeError: {e}")
    except Exception as e:
        print(f"    [INFO] Different error type: {type(e).__name__}: {e}")

    print("\n  [PASS] SileroVAD ONNX diagnosis complete")
    return True


def test_sensevoice_funasr(oac_dir: Path):
    """Test 3: FunASR SenseVoice のロードテスト"""
    print("\n" + "=" * 60)
    print("TEST 3: FunASR SenseVoice Model Load")
    print("=" * 60)

    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA: {torch.cuda.is_available()}")
    except ImportError:
        print("  [FAIL] PyTorch not installed")
        return False

    try:
        from funasr import AutoModel
        print("  FunASR imported successfully")
    except ImportError:
        print("  [SKIP] FunASR not installed")
        return None

    model_name = "iic/SenseVoiceSmall"
    model_path = oac_dir / "models" / "iic" / "SenseVoiceSmall"
    if model_path.exists():
        model_name = str(model_path)

    print(f"  Loading model: {model_name}")

    try:
        model = AutoModel(model=model_name, disable_update=True)
        print("  [PASS] SenseVoice model loaded successfully")
    except RuntimeError as e:
        if "list" in str(e).lower():
            print(f"  [FAIL] ONNX list error during model load!")
            print(f"  Error: {e}")
            print("  >>> THIS is the source of the error! <<<")
            print("  FunASR's model loading triggers ONNX with list input.")
            return False
        else:
            print(f"  [FAIL] RuntimeError: {e}")
            return False
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

    # テスト推論
    print("\n  Testing inference with dummy audio...")
    try:
        import numpy as np
        dummy_audio = np.zeros(16000, dtype=np.float32)
        res = model.generate(input=dummy_audio, batch_size_s=10)
        print(f"  Result: {res}")
        print("  [PASS] SenseVoice inference succeeded")
    except RuntimeError as e:
        if "list" in str(e).lower():
            print(f"  [FAIL] ONNX list error during inference!")
            print(f"  Error: {e}")
            print("  >>> THIS is the source of the error! <<<")
            return False
        else:
            print(f"  [FAIL] RuntimeError: {e}")
            return False
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

    return True


def test_vad_handler_timestamp_bug():
    """Test 4: VAD handler の timestamp[0] バグ確認"""
    print("\n" + "=" * 60)
    print("TEST 4: VAD Handler timestamp[0] Bug Check")
    print("=" * 60)

    print("  In vad_handler_silero.py handle() method:")
    print("    timestamp = None")
    print("    if inputs.is_timestamp_valid():")
    print("        timestamp = inputs.timestamp")
    print("    ...")
    print("    context.slice_context.update_start_id(timestamp[0], ...)")
    print()
    print("  If is_timestamp_valid() returns False, timestamp stays None.")
    print("  Then timestamp[0] raises TypeError!")
    print()

    # Simulate the bug
    timestamp = None
    try:
        _ = timestamp[0]
        print("  [UNEXPECTED] No error")
    except TypeError as e:
        print(f"  [CONFIRMED] TypeError: {e}")
        print("  This crashes the handler BEFORE any ONNX call.")
        print("  The pipeline may then produce the RuntimeError downstream.")

    print()
    print("  FIX: Add null check before timestamp[0]:")
    print("    if timestamp is not None:")
    print("        context.slice_context.update_start_id(timestamp[0], ...)")
    print("    else:")
    print("        context.slice_context.update_start_id(0, ...)")

    return True


def test_audio_data_flow(oac_dir: Path):
    """Test 5: fastrtc -> handler のデータフロー確認"""
    print("\n" + "=" * 60)
    print("TEST 5: Audio Data Flow Check")
    print("=" * 60)

    try:
        sys.path.insert(0, str(oac_dir / "src"))
        from engine_utils.general_slicer import SliceContext, slice_data
        import numpy as np

        # SliceContext のテスト
        ctx = SliceContext.create_numpy_slice_context(slice_size=512, slice_axis=0)
        print("  SliceContext created successfully")

        # numpy audio → slice_data
        audio = np.random.randn(4096).astype(np.float32)
        slices = list(slice_data(ctx, audio))
        print(f"  slice_data: {len(slices)} slices from {audio.shape} audio")

        for i, s in enumerate(slices[:3]):
            print(f"    slice[{i}]: type={type(s).__name__}, dtype={s.dtype}, shape={s.shape}")

        all_numpy = all(isinstance(s, np.ndarray) for s in slices)
        if all_numpy:
            print("  [PASS] All slices are numpy arrays")
        else:
            print("  [FAIL] Some slices are NOT numpy arrays!")
            for i, s in enumerate(slices):
                if not isinstance(s, np.ndarray):
                    print(f"    slice[{i}]: type={type(s).__name__}")

        return all_numpy

    except ImportError as e:
        print(f"  [SKIP] Cannot import engine_utils: {e}")
        return None
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def main():
    oac_dir = find_oac_dir()

    print("=" * 60)
    print("ONNX RuntimeError Diagnostic Tool")
    print("=" * 60)
    print(f"OAC Directory: {oac_dir}")
    print(f"Python: {sys.version}")

    results = {}

    # Test 1: ONNX Runtime basic
    results["onnx_basic"] = test_onnx_runtime_basic()

    # Test 2: SileroVAD ONNX
    if results["onnx_basic"]:
        results["silero_vad"] = test_silero_vad_onnx(oac_dir)

    # Test 3: FunASR SenseVoice
    results["sensevoice"] = test_sensevoice_funasr(oac_dir)

    # Test 4: timestamp bug
    results["timestamp_bug"] = test_vad_handler_timestamp_bug()

    # Test 5: Audio data flow
    results["data_flow"] = test_audio_data_flow(oac_dir)

    # Summary
    print("\n" + "=" * 60)
    print("DIAGNOSIS SUMMARY")
    print("=" * 60)

    for name, passed in results.items():
        if passed is None:
            status = "SKIP"
        elif passed:
            status = "PASS"
        else:
            status = "FAIL"
        print(f"  [{status}] {name}")

    # Recommendations
    print("\n  RECOMMENDATIONS:")
    print("  1. Apply patch_vad_handler.py to add defensive type checking")
    print("  2. Fix timestamp[0] null check in vad_handler_silero.py")
    print("  3. If SenseVoice FAIL, check FunASR ONNX configuration")
    print("  4. Run OpenAvatarChat with ONNX_DEBUG=1 for detailed logging")

    return 0 if all(v is not False for v in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
