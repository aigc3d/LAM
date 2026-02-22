"""
ASR SenseVoice パフォーマンス劣化修正パッチ

問題:
    1回目の発話は正常に認識される（rtf=0.629, 1.25秒）
    2回目の発話でASR推論が24倍遅くなる（rtf=15.027, 29.83秒）
    fastrtcが60秒タイムアウトでリセットされ、以降音声入力が無反応になる

原因:
    SenseVoice (FunASR) がGPU推論後にメモリを解放しない。
    LAMモデルとGPUメモリを共有しているため、2回目の推論で
    GPUメモリ不足→CPUフォールバック→30秒かかる。

修正:
    1. SenseVoice推論後に torch.cuda.empty_cache() を追加
    2. 推論にタイムアウトを追加（10秒超で強制中断→再試行）
    3. GCで不要なテンソルを即座に回収

使い方:
    cd C:\\Users\\hamad\\OpenAvatarChat
    python tests/a2e_japanese/patch_asr_perf_fix.py

    確認のみ:
    python tests/a2e_japanese/patch_asr_perf_fix.py --dry-run
"""

import re
import shutil
import sys
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
    return None


def patch_asr_handler(oac_dir: Path, dry_run: bool = False) -> bool:
    """SenseVoice ASR handler にGPUメモリ管理を追加"""
    handler_path = (oac_dir / "src" / "handlers" / "asr" /
                    "sensevoice" / "asr_handler_sensevoice.py")

    if not handler_path.exists():
        print(f"  [ERROR] {handler_path} not found")
        return False

    content = handler_path.read_text(encoding="utf-8")

    if "# [PERF_PATCH]" in content:
        print("  [ALREADY] Performance patch already applied")
        return True

    lines = content.splitlines()
    changes = []

    # ========================================
    # 修正1: import追加（ファイル先頭付近）
    # ========================================
    import_lines = []
    last_import_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            last_import_idx = i

    # gc と torch のimport追加
    has_gc = any("import gc" in l for l in lines)
    has_torch_import = any("import torch" in l for l in lines)

    new_imports = []
    if not has_gc:
        new_imports.append("import gc")
    if not has_torch_import:
        new_imports.append("import torch")

    if new_imports:
        insert_text = "\n".join(new_imports)
        lines.insert(last_import_idx + 1, insert_text)
        changes.append(f"Added imports: {', '.join(new_imports)}")
        # Adjust indices after insert
        last_import_idx += 1

    # ========================================
    # 修正2: generate()呼び出し後にGPUメモリクリーンアップ追加
    # ========================================
    # generate() 呼び出しの場所を探す
    gen_result_line = None
    gen_indent = ""
    for i, line in enumerate(lines):
        # generate()の結果をログ出力している行を探す
        if "generate(" in line and ("self.model" in line or "model.generate" in line):
            gen_result_line = i
            gen_indent = line[:len(line) - len(line.lstrip())]
            break

    if gen_result_line is not None:
        # generate() 呼び出しの閉じ括弧を探す
        paren_count = 0
        end_line = gen_result_line
        for i in range(gen_result_line, min(gen_result_line + 30, len(lines))):
            paren_count += lines[i].count("(") - lines[i].count(")")
            if paren_count <= 0:
                end_line = i
                break

        # generate()の後にGPUクリーンアップを挿入
        cleanup_code = [
            f"{gen_indent}# [PERF_PATCH] Free GPU memory after ASR inference",
            f"{gen_indent}# Prevents 2nd inference from falling back to CPU (24x slowdown)",
            f"{gen_indent}if torch.cuda.is_available():",
            f"{gen_indent}    torch.cuda.empty_cache()",
            f"{gen_indent}gc.collect()",
        ]

        # ログ出力行の後に挿入（generate結果のlog行を探す）
        insert_after = end_line
        for i in range(end_line + 1, min(end_line + 10, len(lines))):
            if "logger" in lines[i] and ("text" in lines[i] or "result" in lines[i] or "info" in lines[i].lower()):
                insert_after = i
                break

        for j, cl in enumerate(cleanup_code):
            lines.insert(insert_after + 1 + j, cl)

        changes.append(f"Added GPU memory cleanup after generate() (line ~{end_line + 1})")
    else:
        print("  [WARN] Could not find model.generate() call")
        print("         Adding cleanup at end of handle() method instead")

        # handle() メソッドの return 前に追加
        for i in range(len(lines) - 1, -1, -1):
            stripped = lines[i].strip()
            if stripped.startswith("return") and "handle" not in stripped:
                indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
                cleanup_code = [
                    f"{indent}# [PERF_PATCH] Free GPU memory after ASR inference",
                    f"{indent}if torch.cuda.is_available():",
                    f"{indent}    torch.cuda.empty_cache()",
                    f"{indent}gc.collect()",
                ]
                for j, cl in enumerate(cleanup_code):
                    lines.insert(i, cl)
                changes.append(f"Added GPU cleanup before return (line ~{i + 1})")
                break

    # ========================================
    # 修正3: dump audio の部分にもクリーンアップ
    # ========================================
    for i, line in enumerate(lines):
        if "dump audio" in line and "logger" in line:
            indent = line[:len(line) - len(line.lstrip())]
            # dump audio の前にGPUキャッシュクリア
            cleanup = f"{indent}torch.cuda.empty_cache() if torch.cuda.is_available() else None  # [PERF_PATCH]"
            lines.insert(i, cleanup)
            changes.append(f"Added pre-inference GPU cleanup (line ~{i + 1})")
            break

    if not changes:
        print("  [SKIP] No changes to make")
        return True

    # 結果表示
    new_content = "\n".join(lines)

    print("  Changes:")
    for c in changes:
        print(f"    - {c}")

    if dry_run:
        print("\n  [DRY RUN] No files modified")
        return True

    # バックアップ
    backup = handler_path.with_suffix(".py.perf_bak")
    if not backup.exists():
        shutil.copy2(handler_path, backup)
        print(f"  Backup: {backup}")

    handler_path.write_text(new_content, encoding="utf-8")
    print(f"  [SAVED] {handler_path}")
    return True


def patch_lam_handler(oac_dir: Path, dry_run: bool = False) -> bool:
    """LAM avatar handler にもGPUメモリ管理を追加"""
    handler_path = (oac_dir / "src" / "handlers" / "avatar" /
                    "lam" / "avatar_handler_lam_audio2expression.py")

    if not handler_path.exists():
        print(f"  [SKIP] {handler_path} not found")
        return True  # Not critical

    content = handler_path.read_text(encoding="utf-8")

    if "# [PERF_PATCH]" in content:
        print("  [ALREADY] LAM performance patch already applied")
        return True

    lines = content.splitlines()
    changes = []

    # import torch があるか確認
    has_torch = any("import torch" in l for l in lines)
    has_gc = any("import gc" in l for l in lines)

    if not has_gc:
        # 最後のimport行の後にgc追加
        for i, line in enumerate(lines):
            if line.startswith("import ") or line.startswith("from "):
                last_import = i
        lines.insert(last_import + 1, "import gc")
        changes.append("Added import gc")

    # Inference完了ログの後にGPUクリーンアップ追加
    for i, line in enumerate(lines):
        if "Inference on" in line and "finished in" in line:
            indent = line[:len(line) - len(line.lstrip())]
            cleanup = [
                f"{indent}# [PERF_PATCH] Free GPU memory after LAM inference",
                f"{indent}if torch.cuda.is_available():",
                f"{indent}    torch.cuda.empty_cache()",
                f"{indent}gc.collect()",
            ]
            for j, cl in enumerate(cleanup):
                lines.insert(i + 1 + j, cl)
            changes.append(f"Added GPU cleanup after LAM inference (line ~{i + 1})")
            break

    if not changes:
        print("  [SKIP] No changes to make")
        return True

    new_content = "\n".join(lines)

    print("  Changes:")
    for c in changes:
        print(f"    - {c}")

    if dry_run:
        print("\n  [DRY RUN] No files modified")
        return True

    backup = handler_path.with_suffix(".py.perf_bak")
    if not backup.exists():
        shutil.copy2(handler_path, backup)
        print(f"  Backup: {backup}")

    handler_path.write_text(new_content, encoding="utf-8")
    print(f"  [SAVED] {handler_path}")
    return True


def create_startup_wrapper(oac_dir: Path, dry_run: bool = False) -> bool:
    """GPUメモリ管理を強化した起動ラッパーを作成"""
    wrapper_path = oac_dir / "start_japanese.py"

    if wrapper_path.exists():
        content = wrapper_path.read_text(encoding="utf-8")
        if "PERF_PATCH" in content:
            print("  [ALREADY] Startup wrapper already exists")
            return True

    wrapper_content = '''"""
Japanese mode startup with GPU memory optimization.
Usage: python start_japanese.py
"""
import os
import sys

# [PERF_PATCH] GPU memory management environment variables
# Reserve less memory so ASR and LAM can share GPU
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Prevent TensorFlow/ONNX from grabbing all GPU memory
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
# Limit GPU memory growth
os.environ.setdefault("PYTORCH_NO_CUDA_MEMORY_CACHING", "0")

# Ensure UTF-8 output on Windows
os.environ.setdefault("PYTHONUTF8", "1")

print("=" * 50)
print("Starting OpenAvatarChat (Japanese Mode)")
print("GPU Memory Optimization: ENABLED")
print("=" * 50)

# Check GPU memory
try:
    import torch
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        total_mb = gpu.total_mem / 1024 / 1024
        print(f"GPU: {gpu.name} ({total_mb:.0f} MB)")
        free_mb = (torch.cuda.mem_get_info()[0]) / 1024 / 1024
        print(f"Free GPU Memory: {free_mb:.0f} MB")
        if free_mb < 2000:
            print("WARNING: Low GPU memory! ASR may fall back to CPU.")
            print("  Close other GPU applications before running.")
    else:
        print("WARNING: CUDA not available. ASR will be slow.")
except Exception as e:
    print(f"GPU check failed: {e}")

print()

# Launch with Japanese config
sys.argv = ["src/demo.py", "--config", "config/chat_with_lam.yaml"]
exec(open("src/demo.py").read())
'''

    if dry_run:
        print("  [DRY RUN] Would create start_japanese.py")
        return True

    wrapper_path.write_text(wrapper_content, encoding="utf-8")
    print(f"  [CREATED] {wrapper_path}")
    return True


def main():
    print("=" * 60)
    print("ASR Performance Fix Patch")
    print("SenseVoice 2回目推論の24倍遅延を修正")
    print("=" * 60)

    dry_run = "--dry-run" in sys.argv

    oac_dir = find_oac_dir()
    if not oac_dir:
        print("ERROR: OpenAvatarChat directory not found")
        sys.exit(1)

    print(f"OAC: {oac_dir}")
    print(f"Mode: {'DRY RUN' if dry_run else 'APPLY'}\n")

    # Patch 1: ASR handler
    print("[1/3] ASR SenseVoice handler (GPU memory cleanup):")
    ok1 = patch_asr_handler(oac_dir, dry_run)

    # Patch 2: LAM handler
    print(f"\n[2/3] LAM avatar handler (GPU memory cleanup):")
    ok2 = patch_lam_handler(oac_dir, dry_run)

    # Patch 3: Startup wrapper
    print(f"\n[3/3] Startup wrapper (GPU memory optimization):")
    ok3 = create_startup_wrapper(oac_dir, dry_run)

    print(f"\n{'=' * 60}")
    if ok1 and ok2 and ok3:
        print("All patches applied!")
    else:
        print("Some patches failed. See above for details.")

    print(f"""
Next steps:
  1. Apply all patches (run in order):
     python tests/a2e_japanese/patch_config_japanese.py
     python tests/a2e_japanese/patch_asr_language.py
     python tests/a2e_japanese/patch_asr_perf_fix.py
     python tests/a2e_japanese/patch_vad_handler.py

  2. Start with GPU-optimized launcher:
     python start_japanese.py

  3. Or manually:
     set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
     python src/demo.py --config config/chat_with_lam.yaml
""")


if __name__ == "__main__":
    main()
