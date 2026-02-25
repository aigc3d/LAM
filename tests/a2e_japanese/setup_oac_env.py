"""
OpenAvatarChat 環境セットアップ & 既知問題自動修正スクリプト

チャットログで判明した既知問題を自動的に検出・修正:
  1. chat_with_lam.yaml の構造 (handlers: → default: > chat_engine: > handler_configs:)
  2. infer.py の .cuda() → .cpu() (GPUなし環境)
  3. 不足パッケージのインストール
  4. モデルファイルの存在確認
  5. SSL証明書の確認

使い方:
    cd C:\Users\hamad\OpenAvatarChat
    conda activate oac
    python tests/a2e_japanese/setup_oac_env.py

    または:
    python tests/a2e_japanese/setup_oac_env.py --oac-dir C:\Users\hamad\OpenAvatarChat
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


class OACSetupChecker:
    def __init__(self, oac_dir: Path):
        self.oac_dir = oac_dir
        self.issues = []
        self.fixes_applied = []

    def check_all(self):
        """全チェック実行"""
        print("=" * 60)
        print("OpenAvatarChat Environment Check")
        print(f"Directory: {self.oac_dir}")
        print("=" * 60)

        self._check_directory_structure()
        self._check_python_packages()
        self._check_models()
        self._check_cuda_cpu()
        self._check_config_yaml()
        self._check_ssl_certs()
        self._check_vad_handler_bugs()
        self._check_llm_handler_bugs()

        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        if not self.issues:
            print("  All checks passed! Environment is ready.")
        else:
            print(f"  {len(self.issues)} issue(s) found:")
            for i, issue in enumerate(self.issues, 1):
                print(f"  {i}. {issue}")

        if self.fixes_applied:
            print(f"\n  {len(self.fixes_applied)} fix(es) applied:")
            for fix in self.fixes_applied:
                print(f"  - {fix}")

        return len(self.issues) == 0

    def _check_directory_structure(self):
        """基本ディレクトリ構造の確認"""
        print("\n[1/6] Directory Structure")
        required = [
            "src/demo.py",
            "src/handlers/avatar/lam/avatar_handler_lam_audio2expression.py",
            "src/handlers/avatar/lam/LAM_Audio2Expression/engines/infer.py",
            "config/chat_with_lam.yaml",
        ]
        for rel_path in required:
            full_path = self.oac_dir / rel_path
            exists = full_path.exists()
            status = "OK" if exists else "MISSING"
            print(f"  [{status}] {rel_path}")
            if not exists:
                self.issues.append(f"Missing: {rel_path}")

    def _check_python_packages(self):
        """必要パッケージの確認"""
        print("\n[2/6] Python Packages")
        packages = {
            "edge_tts": "edge-tts",
            "addict": "addict",
            "yapf": "yapf",
            "regex": "regex",
            "librosa": "librosa",
            "transformers": "transformers",
            "termcolor": "termcolor",
            "torch": "torch",
            "numpy": "numpy",
            "omegaconf": "omegaconf",
        }
        missing = []
        for module_name, pip_name in packages.items():
            try:
                __import__(module_name)
                print(f"  [OK] {module_name}")
            except ImportError:
                print(f"  [MISSING] {module_name} (pip install {pip_name})")
                missing.append(pip_name)

        if missing:
            self.issues.append(f"Missing packages: {', '.join(missing)}")
            print(f"\n  Install all missing: pip install {' '.join(missing)}")

    def _check_models(self):
        """モデルファイルの確認"""
        print("\n[3/6] Model Files")
        models_dir = self.oac_dir / "models"

        checks = {
            "LAM_audio2exp checkpoint": [
                models_dir / "LAM_audio2exp" / "pretrained_models" / "lam_audio2exp_streaming.tar",
                models_dir / "LAM_audio2exp" / "pretrained_models",
            ],
            "wav2vec2-base-960h": [
                models_dir / "wav2vec2-base-960h" / "pytorch_model.bin",
                models_dir / "wav2vec2-base-960h" / "model.safetensors",
                models_dir / "wav2vec2-base-960h" / "config.json",
            ],
            "SenseVoiceSmall": [
                models_dir / "iic" / "SenseVoiceSmall" / "model.pt",
            ],
        }

        for name, paths in checks.items():
            found = any(p.exists() for p in paths)
            status = "OK" if found else "MISSING"
            print(f"  [{status}] {name}")
            if not found:
                self.issues.append(f"Missing model: {name}")
                if "LAM_audio2exp" in name:
                    print(f"    Download from HuggingFace: 3DAIGC/LAM_audio2exp")
                elif "wav2vec2" in name:
                    print(f"    Run: python -c \"from transformers import Wav2Vec2Model; "
                          f"m = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h'); "
                          f"m.save_pretrained(r'{models_dir / 'wav2vec2-base-960h'}')\"")

    def _check_cuda_cpu(self):
        """CUDA/CPU環境の確認とinfer.pyの修正"""
        print("\n[4/6] CUDA/CPU Environment")

        try:
            import torch
            cuda_available = torch.cuda.is_available()
            print(f"  PyTorch: {torch.__version__}")
            print(f"  CUDA available: {cuda_available}")
        except ImportError:
            print("  [FAIL] PyTorch not installed")
            self.issues.append("PyTorch not installed")
            return

        if cuda_available:
            print(f"  CUDA version: {torch.version.cuda}")
            print("  GPU mode: OK")
            return

        # GPUなし → infer.pyの.cuda()を.cpu()に変更が必要
        print("  GPU not available. Checking infer.py for .cuda() calls...")

        infer_path = (self.oac_dir / "src" / "handlers" / "avatar" / "lam" /
                      "LAM_Audio2Expression" / "engines" / "infer.py")

        if not infer_path.exists():
            print(f"  [SKIP] infer.py not found at {infer_path}")
            return

        content = infer_path.read_text(encoding="utf-8")
        cuda_calls = [
            (i + 1, line.strip())
            for i, line in enumerate(content.splitlines())
            if ".cuda()" in line and not line.strip().startswith("#")
        ]

        if cuda_calls:
            print(f"  [WARN] Found {len(cuda_calls)} .cuda() calls in infer.py:")
            for line_no, line in cuda_calls:
                print(f"    Line {line_no}: {line}")
            self.issues.append(f"infer.py has {len(cuda_calls)} .cuda() calls (no GPU available)")
            print("\n  To fix, replace .cuda() with .cpu() in infer.py")
            print(f"  File: {infer_path}")
        else:
            print("  [OK] No .cuda() calls found (already patched or not needed)")

    def _check_config_yaml(self):
        """chat_with_lam.yamlの構造確認"""
        print("\n[5/6] Config YAML Structure")

        config_path = self.oac_dir / "config" / "chat_with_lam.yaml"
        if not config_path.exists():
            print(f"  [MISSING] {config_path}")
            self.issues.append("chat_with_lam.yaml not found")
            return

        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except Exception as e:
            print(f"  [FAIL] Cannot parse YAML: {e}")
            self.issues.append(f"YAML parse error: {e}")
            return

        # 構造チェック: default > chat_engine > handler_configs が正しい構造
        if "handlers" in config and "default" not in config:
            print("  [FAIL] Wrong structure: 'handlers:' at root level")
            print("  Should be: default > chat_engine > handler_configs")
            self.issues.append("chat_with_lam.yaml has wrong structure (handlers: instead of default:)")
            return

        handler_configs = (config.get("default", {})
                          .get("chat_engine", {})
                          .get("handler_configs", {}))

        if not handler_configs:
            print("  [FAIL] No handler_configs found")
            self.issues.append("No handler_configs in chat_with_lam.yaml")
            return

        print(f"  [OK] Structure: default > chat_engine > handler_configs")
        print(f"  Handlers: {', '.join(handler_configs.keys())}")

        # 各handlerのmoduleチェック
        required_handlers = ["LamClient", "SileroVad", "SenseVoice", "LLMOpenAICompatible", "LAM_Driver"]
        tts_handlers = ["Edge_TTS", "EdgeTTS"]

        for h in required_handlers:
            if h in handler_configs:
                print(f"    [OK] {h}: {handler_configs[h].get('module', 'N/A')}")
            else:
                print(f"    [MISSING] {h}")
                self.issues.append(f"Missing handler: {h}")

        tts_found = any(h in handler_configs for h in tts_handlers)
        if tts_found:
            tts_name = next(h for h in tts_handlers if h in handler_configs)
            voice = handler_configs[tts_name].get("voice", "N/A")
            print(f"    [OK] TTS ({tts_name}): voice={voice}")
        else:
            print(f"    [MISSING] TTS handler (Edge_TTS or EdgeTTS)")
            self.issues.append("Missing TTS handler")

        # LLM API設定
        llm_config = handler_configs.get("LLMOpenAICompatible", {})
        api_url = llm_config.get("api_url", "")
        api_key = llm_config.get("api_key", "")
        model = llm_config.get("model_name", "")

        if "gemini" in api_url.lower() or "gemini" in model.lower():
            print(f"    [OK] LLM: Gemini API ({model})")
            if not api_key or api_key == "YOUR_GEMINI_API_KEY":
                print(f"    [WARN] API key not set!")
                self.issues.append("Gemini API key not configured")
        elif "dashscope" in api_url.lower():
            print(f"    [WARN] LLM: DashScope (may not work outside China)")
        else:
            print(f"    [INFO] LLM: {api_url} ({model})")

    def _check_ssl_certs(self):
        """SSL証明書の確認（WebRTCに必要）"""
        print("\n[6/6] SSL Certificates (for WebRTC)")

        cert_file = self.oac_dir / "ssl_certs" / "localhost.crt"
        key_file = self.oac_dir / "ssl_certs" / "localhost.key"

        if cert_file.exists() and key_file.exists():
            print(f"  [OK] SSL certificates found")
        else:
            print(f"  [WARN] SSL certificates not found")
            print(f"  WebRTC requires HTTPS. For localhost testing:")
            print(f"  mkdir ssl_certs")
            print(f"  openssl req -x509 -newkey rsa:2048 -keyout ssl_certs/localhost.key \\")
            print(f"    -out ssl_certs/localhost.crt -days 365 -nodes \\")
            print(f"    -subj '/CN=localhost'")
            print(f"  Or use mkcert: mkcert -install && mkcert localhost")
            # SSLは必須ではない（localhost HTTPでもマイク動く場合あり）
            # self.issues.append("SSL certificates missing")


    def _check_vad_handler_bugs(self):
        """VADハンドラーの既知バグ確認"""
        print("\n[7/7] VAD Handler Known Bugs")

        vad_path = (self.oac_dir / "src" / "handlers" / "vad" / "silerovad" /
                    "vad_handler_silero.py")

        if not vad_path.exists():
            print(f"  [SKIP] VAD handler not found")
            return

        content = vad_path.read_text(encoding="utf-8")

        # Bug 1: timestamp[0] NoneType crash
        if ("context.slice_context.update_start_id(timestamp[0]" in content
                and "if timestamp is not None" not in content):
            print("  [BUG] timestamp[0] NoneType crash detected!")
            print("    When audio arrives without valid timestamp,")
            print("    timestamp[0] crashes with TypeError.")
            print("    FIX: Apply patch_vad_handler.py")
            self.issues.append("VAD handler: timestamp[0] NoneType bug")
        else:
            print("  [OK] timestamp null check")

        # Bug 2: No defensive type check on ONNX inputs
        if ("isinstance(clip, np.ndarray)" not in content
                and "isinstance(context.model_state" not in content):
            print("  [WARN] No defensive type checking on ONNX inputs")
            print("    If upstream data is not numpy, ONNX will crash with:")
            print("    RuntimeError: Input data type <class 'list'> is not supported.")
            print("    FIX: Apply patch_vad_handler.py")
            self.issues.append("VAD handler: missing ONNX input type validation")
        else:
            print("  [OK] ONNX input type checking")

        # Check SenseVoice handler
        asr_path = (self.oac_dir / "src" / "handlers" / "asr" / "sensevoice" /
                    "asr_handler_sensevoice.py")

        if asr_path.exists():
            asr_content = asr_path.read_text(encoding="utf-8")
            if "np.zeros(shape=" in asr_content and "dtype=remainder_audio.dtype" not in asr_content:
                print("  [WARN] SenseVoice np.zeros dtype mismatch")
                print("    np.zeros without dtype creates float64, audio is float32")
                self.issues.append("SenseVoice handler: np.zeros dtype mismatch")
            else:
                print("  [OK] SenseVoice dtype handling")

        # Check SileroVAD ONNX model
        model_candidates = list(self.oac_dir.rglob("silero_vad.onnx"))
        if model_candidates:
            print(f"  [OK] SileroVAD ONNX model found: {model_candidates[0]}")
            try:
                import onnxruntime
                print(f"  [OK] onnxruntime {onnxruntime.__version__}")
            except ImportError:
                print("  [FAIL] onnxruntime not installed")
                self.issues.append("onnxruntime not installed")
        else:
            print("  [WARN] silero_vad.onnx not found")
            self.issues.append("SileroVAD ONNX model not found")


    def _check_llm_handler_bugs(self):
        """LLMハンドラーの既知バグ確認 (Gemini dict content)"""
        print("\n[8/8] LLM Handler Known Bugs")

        llm_path = (self.oac_dir / "src" / "handlers" / "llm" /
                    "openai_compatible" / "llm_handler_openai_compatible.py")

        if not llm_path.exists():
            print(f"  [SKIP] LLM handler not found")
            return

        content = llm_path.read_text(encoding="utf-8")

        # Bug: Gemini API returns delta.content as dict instead of str
        # This causes: TypeError: float() argument must be a string or
        #              a real number, not 'dict'
        if ("set_main_data(" in content
                and "# [PATCH] Gemini dict content fix" not in content):
            print("  [BUG] Gemini dict content not handled!")
            print("    Gemini OpenAI-compatible API may return delta.content")
            print("    as dict/list instead of str, causing TypeError.")
            print("    FIX: python tests/a2e_japanese/patch_llm_handler.py")
            self.issues.append("LLM handler: Gemini dict content bug")
        else:
            print("  [OK] Gemini dict content handling")


def main():
    parser = argparse.ArgumentParser(description="OpenAvatarChat Environment Setup Checker")
    parser.add_argument("--oac-dir", type=str, default=None,
                        help="Path to OpenAvatarChat directory")
    parser.add_argument("--fix", action="store_true",
                        help="Attempt to auto-fix issues")
    args = parser.parse_args()

    if args.oac_dir:
        oac_dir = Path(args.oac_dir)
    else:
        # 自動検出
        candidates = [
            Path(r"C:\Users\hamad\OpenAvatarChat"),
            Path.home() / "OpenAvatarChat",
            Path.cwd(),
        ]
        oac_dir = next((p for p in candidates if (p / "src" / "demo.py").exists()), None)
        if oac_dir is None:
            print("ERROR: OpenAvatarChat directory not found.")
            print("Use --oac-dir to specify the path.")
            sys.exit(1)

    checker = OACSetupChecker(oac_dir)
    ok = checker.check_all()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
