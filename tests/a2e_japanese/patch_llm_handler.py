"""
LLM Handler (OpenAI Compatible) 修正パッチ

問題:
    Gemini API の OpenAI互換エンドポイントが delta.content を
    文字列ではなく dict や list で返すことがある。
    これにより set_main_data() → np.array(data, dtype=np.float32) で
    TypeError: float() argument must be a string or a real number, not 'dict'
    が発生する。

エラー:
    File "llm_handler_openai_compatible.py", line 167, in handle
        output.set_main_data(output_text)
    ...
    TypeError: float() argument must be a string or a real number, not 'dict'

修正:
    output_text が dict/list の場合に文字列を正しく抽出する。

使い方:
    cd C:\\Users\\hamad\\OpenAvatarChat
    python tests/a2e_japanese/patch_llm_handler.py

    または --dry-run で変更内容だけ確認:
    python tests/a2e_japanese/patch_llm_handler.py --dry-run
"""

import re
import shutil
import sys
from pathlib import Path


def find_oac_dir() -> Path:
    """OpenAvatarChat ディレクトリを自動検出"""
    candidates = [
        Path(r"C:\Users\hamad\OpenAvatarChat"),
        Path.home() / "OpenAvatarChat",
        Path.cwd(),
    ]
    for p in candidates:
        if (p / "src" / "handlers").exists():
            return p
    return None


def patch_llm_handler(oac_dir: Path, dry_run: bool = False) -> bool:
    """LLMハンドラーにGemini dict対応パッチを適用"""
    handler_path = (oac_dir / "src" / "handlers" / "llm" /
                    "openai_compatible" / "llm_handler_openai_compatible.py")

    if not handler_path.exists():
        print(f"  [ERROR] File not found: {handler_path}")
        return False

    content = handler_path.read_text(encoding="utf-8")
    lines = content.splitlines()

    # --- 修正1: output_text の dict/list 安全変換 ---
    # パターン: output.set_main_data(output_text) の直前に型チェックを挿入
    #
    # Gemini API の OpenAI互換エンドポイントは delta.content を
    # 以下のいずれかの形式で返す可能性がある:
    #   (a) str: "こんにちは"  ← 正常
    #   (b) dict: {"type": "text", "text": "こんにちは"}
    #   (c) list: [{"type": "text", "text": "こんにちは"}]
    #   (d) None ← ストリームの最初/最後のチャンク

    # 既にパッチ済みか確認
    if "# [PATCH] Gemini dict content fix" in content:
        print("  [ALREADY] LLM handler already patched")
        return True

    # set_main_data(output_text) を含む行を探す
    target_line_idx = None
    for i, line in enumerate(lines):
        if "set_main_data(output_text)" in line:
            target_line_idx = i
            break

    if target_line_idx is None:
        # 別パターン: set_main_data(text) など
        for i, line in enumerate(lines):
            if re.search(r'set_main_data\(\s*\w*text\w*\s*\)', line):
                target_line_idx = i
                break

    if target_line_idx is None:
        print("  [WARN] Could not find set_main_data(output_text) line")
        print("         Manual patching required (see below)")
        print_manual_guide()
        return False

    # インデント検出
    target_line = lines[target_line_idx]
    indent = len(target_line) - len(target_line.lstrip())
    indent_str = target_line[:indent]

    # output_text 変数名を検出
    match = re.search(r'set_main_data\((\w+)\)', target_line)
    if not match:
        print("  [WARN] Cannot parse variable name from set_main_data call")
        print_manual_guide()
        return False
    var_name = match.group(1)

    # パッチ内容: set_main_data の前に安全変換を挿入
    patch_lines = [
        f"{indent_str}# [PATCH] Gemini dict content fix",
        f"{indent_str}if isinstance({var_name}, dict):",
        f"{indent_str}    {var_name} = {var_name}.get('text', '') or {var_name}.get('content', '') or str({var_name})",
        f"{indent_str}elif isinstance({var_name}, list):",
        f"{indent_str}    {var_name} = ''.join(",
        f"{indent_str}        part.get('text', '') if isinstance(part, dict) else str(part)",
        f"{indent_str}        for part in {var_name}",
        f"{indent_str}    )",
        f"{indent_str}elif {var_name} is None:",
        f"{indent_str}    {var_name} = ''",
        f"{indent_str}elif not isinstance({var_name}, str):",
        f"{indent_str}    {var_name} = str({var_name})",
    ]

    print(f"  Target: line {target_line_idx + 1}: {target_line.strip()}")
    print(f"  Variable: {var_name}")
    print(f"  Inserting {len(patch_lines)} lines of type-safety check before set_main_data")

    if dry_run:
        print("\n  --- Patch preview ---")
        for pl in patch_lines:
            print(f"  + {pl}")
        print(f"    {target_line}")
        print("  --- End preview ---")
        return True

    # バックアップ
    backup_path = handler_path.with_suffix(".py.bak")
    if not backup_path.exists():
        shutil.copy2(handler_path, backup_path)
        print(f"  Backup: {backup_path}")

    # パッチ適用
    new_lines = lines[:target_line_idx] + patch_lines + lines[target_line_idx:]
    new_content = "\n".join(new_lines)
    if content.endswith("\n"):
        new_content += "\n"

    handler_path.write_text(new_content, encoding="utf-8")
    print(f"  [APPLIED] Gemini dict content fix")
    return True


def patch_llm_skip_empty_text(oac_dir: Path, dry_run: bool = False) -> bool:
    """空文字列の set_main_data をスキップするパッチ"""
    handler_path = (oac_dir / "src" / "handlers" / "llm" /
                    "openai_compatible" / "llm_handler_openai_compatible.py")

    if not handler_path.exists():
        return False

    content = handler_path.read_text(encoding="utf-8")

    # 既にパッチ済みか確認
    if "# [PATCH] Skip empty text" in content:
        print("  [ALREADY] Skip-empty-text already patched")
        return True

    lines = content.splitlines()

    # set_main_data 行を探す
    for i, line in enumerate(lines):
        if "set_main_data(" in line and ("text" in line.lower() or "output" in line.lower()):
            indent = len(line) - len(line.lstrip())
            indent_str = line[:indent]

            match = re.search(r'set_main_data\((\w+)\)', line)
            if not match:
                continue
            var_name = match.group(1)

            # set_main_data の前にガードを挿入
            guard_lines = [
                f"{indent_str}# [PATCH] Skip empty text",
                f"{indent_str}if not {var_name}:",
                f"{indent_str}    continue",
            ]

            # 既に Gemini dict fix パッチがある場合、その後に挿入
            # （dict fix パッチは set_main_data の直前にある）
            insert_idx = i
            # Gemini dict fix パッチの後ろを探す
            for j in range(max(0, i - 15), i):
                if "# [PATCH] Gemini dict content fix" in lines[j]:
                    # dict fix パッチの最後の行の直後に挿入
                    for k in range(j + 1, i):
                        if not lines[k].strip().startswith(("if ", "elif ", var_name, "part.", "for ")):
                            if lines[k].strip() and not lines[k].strip().startswith(")"):
                                insert_idx = k
                                break
                    break

            if dry_run:
                print(f"\n  --- Skip-empty-text patch preview (before line {insert_idx + 1}) ---")
                for gl in guard_lines:
                    print(f"  + {gl}")
                print("  --- End preview ---")
                return True

            new_lines = lines[:insert_idx] + guard_lines + lines[insert_idx:]
            new_content = "\n".join(new_lines)
            if content.endswith("\n"):
                new_content += "\n"

            handler_path.write_text(new_content, encoding="utf-8")
            print(f"  [APPLIED] Skip empty text guard")
            return True

    print("  [SKIP] Could not find set_main_data for skip-empty patch")
    return True


def print_manual_guide():
    """手動修正ガイドを表示"""
    print("""
=== 手動修正ガイド ===

ファイル: src/handlers/llm/openai_compatible/llm_handler_openai_compatible.py

output.set_main_data(output_text) の直前に以下を追加:

                # [PATCH] Gemini dict content fix
                if isinstance(output_text, dict):
                    output_text = output_text.get('text', '') or output_text.get('content', '') or str(output_text)
                elif isinstance(output_text, list):
                    output_text = ''.join(
                        part.get('text', '') if isinstance(part, dict) else str(part)
                        for part in output_text
                    )
                elif output_text is None:
                    output_text = ''
                elif not isinstance(output_text, str):
                    output_text = str(output_text)
                # [PATCH] Skip empty text
                if not output_text:
                    continue
""")


def main():
    print("=" * 60)
    print("LLM Handler Patch Tool (Gemini dict content fix)")
    print("=" * 60)

    dry_run = "--dry-run" in sys.argv

    oac_dir = find_oac_dir()
    if oac_dir is None:
        print("ERROR: OpenAvatarChat directory not found")
        print("Run from the OpenAvatarChat directory")
        sys.exit(1)

    print(f"OAC: {oac_dir}")
    print(f"Mode: {'DRY RUN' if dry_run else 'APPLY PATCHES'}")
    print()

    print("[1/2] Gemini dict content fix:")
    ok1 = patch_llm_handler(oac_dir, dry_run=dry_run)

    print(f"\n[2/2] Skip empty text guard:")
    ok2 = patch_llm_skip_empty_text(oac_dir, dry_run=dry_run)

    print(f"\n{'=' * 60}")
    if ok1 and ok2:
        print("All patches applied successfully!")
    else:
        print("Some patches could not be applied. See manual guide:")
        print_manual_guide()

    if not dry_run:
        print(f"\nBackup files: *.py.bak")
        print(f"To revert: rename .bak files back to originals")

    print(f"\nNext: Restart OpenAvatarChat:")
    print(f"  python src/demo.py --config config/chat_with_lam_jp.yaml")


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print_manual_guide()
    else:
        main()
