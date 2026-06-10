"""
既存の chat_with_lam.yaml を日本語対応に自動パッチ

動いている config/chat_with_lam.yaml をそのまま使い、
日本語に必要な3箇所だけ変更する。新しい設定ファイルは作らない。

使い方:
    cd C:\\Users\\hamad\\OpenAvatarChat
    python tests/a2e_japanese/patch_config_japanese.py

    確認だけ:
    python tests/a2e_japanese/patch_config_japanese.py --dry-run
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


def patch_config(oac_dir: Path, dry_run: bool = False) -> bool:
    config_path = oac_dir / "config" / "chat_with_lam.yaml"

    if not config_path.exists():
        print(f"  [ERROR] {config_path} not found")
        return False

    content = config_path.read_text(encoding="utf-8")
    original = content

    changes = []

    # --- 1. TTS voice → 日本語 ---
    # voice: "xxx" → voice: "ja-JP-NanamiNeural"
    voice_pattern = r'(voice:\s*["\'])([^"\']+)(["\'])'
    voice_match = re.search(voice_pattern, content)
    if voice_match:
        old_voice = voice_match.group(2)
        if "ja-JP" not in old_voice:
            content = re.sub(
                voice_pattern,
                r'\g<1>ja-JP-NanamiNeural\g<3>',
                content
            )
            changes.append(f"TTS voice: {old_voice} → ja-JP-NanamiNeural")
        else:
            changes.append(f"TTS voice: already Japanese ({old_voice})")
    else:
        # voice行がない場合、Edge_TTS セクションに追加
        edge_pattern = r'(Edge_TTS:.*?module:\s*[^\n]+)'
        edge_match = re.search(edge_pattern, content, re.DOTALL)
        if edge_match:
            insert_after = edge_match.group(0)
            indent = "        "
            content = content.replace(
                insert_after,
                insert_after + f'\n{indent}voice: "ja-JP-NanamiNeural"'
            )
            changes.append("TTS voice: added ja-JP-NanamiNeural")

    # --- 2. LLM system_prompt → 日本語 ---
    jp_prompt = "あなたはAIコンシェルジュです。日本語で簡潔に2〜3文で回答してください。"
    prompt_pattern = r'(system_prompt:\s*["\'])([^"\']*?)(["\'])'
    prompt_match = re.search(prompt_pattern, content)
    if prompt_match:
        old_prompt = prompt_match.group(2)
        if "日本語" not in old_prompt:
            content = re.sub(
                prompt_pattern,
                f'\\g<1>{jp_prompt}\\g<3>',
                content
            )
            changes.append(f"system_prompt: → Japanese")
        else:
            changes.append(f"system_prompt: already Japanese")
    else:
        # system_prompt がない場合、LLM セクションに追加
        llm_pattern = r'(LLMOpenAICompatible:.*?model_name:\s*[^\n]+)'
        llm_match = re.search(llm_pattern, content, re.DOTALL)
        if llm_match:
            insert_after = llm_match.group(0)
            indent = "        "
            content = content.replace(
                insert_after,
                insert_after + f'\n{indent}system_prompt: "{jp_prompt}"'
            )
            changes.append("system_prompt: added Japanese prompt")

    # --- 3. SenseVoice language → ja ---
    # SenseVoice セクションに language: "ja" を追加
    if 'language:' in content and 'SenseVoice' in content:
        # 既に language がある場合、値を "ja" に変更
        lang_pattern = r'(language:\s*["\'])([^"\']*?)(["\'])'
        lang_match = re.search(lang_pattern, content)
        if lang_match and lang_match.group(2) != "ja":
            content = re.sub(lang_pattern, r'\g<1>ja\g<3>', content)
            changes.append(f"ASR language: {lang_match.group(2)} → ja")
        else:
            changes.append("ASR language: already ja")
    else:
        # SenseVoice セクションの model_name 行の後に追加
        sv_pattern = r'(SenseVoice:.*?model_name:\s*[^\n]+)'
        sv_match = re.search(sv_pattern, content, re.DOTALL)
        if sv_match:
            insert_after = sv_match.group(0)
            # model_name 行のインデントを取得
            model_line = re.search(r'(\s+)model_name:', insert_after)
            indent = model_line.group(1) if model_line else "        "
            content = content.replace(
                insert_after,
                insert_after + f'\n{indent}language: "ja"'
            )
            changes.append("ASR language: added ja")
        else:
            changes.append("[WARN] SenseVoice section not found")

    # --- 結果表示 ---
    if not changes:
        print("  No changes needed")
        return True

    print("  Changes:")
    for c in changes:
        print(f"    - {c}")

    if content == original:
        print("  [SKIP] Already configured for Japanese")
        return True

    if dry_run:
        print("\n  [DRY RUN] No files modified")
        return True

    # バックアップ
    backup = config_path.with_suffix(".yaml.bak")
    if not backup.exists():
        shutil.copy2(config_path, backup)
        print(f"  Backup: {backup}")

    config_path.write_text(content, encoding="utf-8")
    print(f"  [SAVED] {config_path}")
    return True


def main():
    print("=" * 60)
    print("Config Japanese Patch")
    print("config/chat_with_lam.yaml を日本語対応に変更")
    print("=" * 60)

    dry_run = "--dry-run" in sys.argv

    oac_dir = find_oac_dir()
    if not oac_dir:
        print("ERROR: OpenAvatarChat directory not found")
        sys.exit(1)

    print(f"OAC: {oac_dir}")
    print(f"Mode: {'DRY RUN' if dry_run else 'APPLY'}\n")

    ok = patch_config(oac_dir, dry_run)

    print(f"\n{'=' * 60}")
    if ok:
        print("Done!")
        print(f"\nNext:")
        print(f"  python tests/a2e_japanese/patch_asr_language.py")
        print(f"  python src/demo.py --config config/chat_with_lam.yaml")
    else:
        print("Failed. Please edit config/chat_with_lam.yaml manually.")


if __name__ == "__main__":
    main()
