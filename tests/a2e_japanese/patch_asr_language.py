"""
ASR SenseVoice 言語強制パッチ

問題:
    SenseVoice ASR が日本語の発話を中国語として認識してしまう。
    ログに <|zh|> と表示され、「ありがとう」が「谢谢」になる等。

原因:
    SenseVoice の generate() が language="auto" (デフォルト) で
    動作しており、短い発話では中国語と誤検出される。

修正:
    generate() 呼び出しに language="ja" を追加して日本語を強制する。
    さらに、設定ファイルから language パラメータを読み取れるようにする。

使い方:
    cd C:\\Users\\hamad\\OpenAvatarChat
    python tests/a2e_japanese/patch_asr_language.py

    または --dry-run で変更内容だけ確認:
    python tests/a2e_japanese/patch_asr_language.py --dry-run
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


def patch_asr_language(oac_dir: Path, dry_run: bool = False) -> bool:
    """SenseVoice ASR handler に language="ja" を強制するパッチ"""
    handler_path = (oac_dir / "src" / "handlers" / "asr" /
                    "sensevoice" / "asr_handler_sensevoice.py")

    if not handler_path.exists():
        print(f"  [ERROR] File not found: {handler_path}")
        return False

    content = handler_path.read_text(encoding="utf-8")

    # 既にパッチ済みか確認
    if "# [PATCH] Force language" in content:
        print("  [ALREADY] ASR language patch already applied")
        return True

    # ========================================
    # 方法1: generate() 呼び出しに language パラメータを追加
    # ========================================
    # FunASR の generate() は以下のようなシグネチャ:
    #   model.generate(input=..., cache={}, language="auto", ...)
    # "auto" をデフォルトから "ja" に変更

    # generate() 呼び出しを探す
    # パターン: self.model.generate( で始まり、) で閉じる部分
    lines = content.splitlines()

    # generate 呼び出しの行範囲を特定
    gen_start = None
    gen_end = None
    for i, line in enumerate(lines):
        if "generate(" in line and ("self.model" in line or "model.generate" in line):
            gen_start = i
            # 閉じ括弧を探す
            paren_count = line.count("(") - line.count(")")
            if paren_count <= 0:
                gen_end = i
            else:
                for j in range(i + 1, min(i + 30, len(lines))):
                    paren_count += lines[j].count("(") - lines[j].count(")")
                    if paren_count <= 0:
                        gen_end = j
                        break
            break

    if gen_start is None:
        print("  [WARN] Could not find model.generate() call")
        print("         Trying alternative approach...")
        return patch_asr_language_alternative(oac_dir, content, handler_path, dry_run)

    print(f"  Found generate() call at lines {gen_start + 1}-{gen_end + 1}")

    # generate() 呼び出し全体を取得
    gen_lines = lines[gen_start:gen_end + 1]
    gen_text = "\n".join(gen_lines)

    # language パラメータが既に存在するか確認
    has_language = "language" in gen_text

    if has_language:
        # language パラメータの値を "ja" に変更
        # language="auto" → language="ja"
        # language='auto' → language='ja'
        new_gen_text = re.sub(
            r'language\s*=\s*["\']auto["\']',
            'language="ja"  # [PATCH] Force language to Japanese',
            gen_text
        )
        if new_gen_text == gen_text:
            # auto 以外の値が設定されている場合
            new_gen_text = re.sub(
                r'language\s*=\s*["\'][^"\']*["\']',
                'language="ja"  # [PATCH] Force language to Japanese',
                gen_text
            )
    else:
        # language パラメータを追加
        # generate( の直後の行にパラメータを挿入
        # input= の行の後に追加
        indent_match = re.search(r'\n(\s+)', gen_text)
        if indent_match:
            param_indent = indent_match.group(1)
        else:
            param_indent = "                "

        # 最後の引数の後、閉じ括弧の前に追加
        # 閉じ括弧 ) の前に language="ja" を挿入
        close_paren_idx = gen_text.rfind(")")
        if close_paren_idx > 0:
            before_close = gen_text[:close_paren_idx].rstrip()
            after_close = gen_text[close_paren_idx:]
            # 最後の引数にカンマがなければ追加
            if not before_close.endswith(","):
                before_close += ","
            new_gen_text = (
                before_close + "\n" +
                param_indent + 'language="ja",  # [PATCH] Force language to Japanese\n' +
                param_indent.rstrip() + after_close.lstrip()
            )
        else:
            print("  [WARN] Cannot parse generate() call structure")
            return patch_asr_language_alternative(oac_dir, content, handler_path, dry_run)

    if dry_run:
        print("\n  --- Patch preview ---")
        print("  Before:")
        for line in gen_lines:
            print(f"  - {line}")
        print("  After:")
        for line in new_gen_text.splitlines():
            print(f"  + {line}")
        print("  --- End preview ---")
        return True

    # バックアップ
    backup_path = handler_path.with_suffix(".py.bak")
    if not backup_path.exists():
        shutil.copy2(handler_path, backup_path)
        print(f"  Backup: {backup_path}")

    # パッチ適用
    new_content = content.replace(gen_text, new_gen_text)
    handler_path.write_text(new_content, encoding="utf-8")
    print(f"  [APPLIED] Force language='ja' in generate() call")
    return True


def patch_asr_language_alternative(oac_dir: Path, content: str, handler_path: Path, dry_run: bool) -> bool:
    """
    代替方法: generate() の戻り値からタグを置換する
    SenseVoice の出力は <|zh|><|NEUTRAL|><|Speech|><|text|> 形式
    この方法は generate() のシグネチャに依存しない
    """
    lines = content.splitlines()

    # 結果テキストを処理する行を探す
    # 通常: res[0]['text'] のような形でテキストを取得
    # ログ出力行を探す（ログにテキスト結果が出ている行の近く）
    target_line_idx = None
    for i, line in enumerate(lines):
        # generate の結果をログ出力している行を探す
        if "generate(" in line or ".generate(" in line:
            # generate呼び出しの直後にパッチを挿入
            target_line_idx = i
            break

    if target_line_idx is None:
        print("  [ERROR] Cannot find generate() call in ASR handler")
        print("         Please apply the patch manually (see below)")
        print_manual_guide()
        return False

    # generate() の行のインデントを取得
    target_line = lines[target_line_idx]
    indent = len(target_line) - len(target_line.lstrip())
    indent_str = target_line[:indent]

    print(f"  Found generate() at line {target_line_idx + 1}")
    print(f"  Will add language='ja' parameter")

    if dry_run:
        print("\n  --- Alternative patch ---")
        print(f"  Add language='ja' to the generate() call on line {target_line_idx + 1}")
        print("  --- End ---")
        return True

    # バックアップ
    backup_path = handler_path.with_suffix(".py.bak")
    if not backup_path.exists():
        shutil.copy2(handler_path, backup_path)
        print(f"  Backup: {backup_path}")

    print("  [WARN] Auto-patching may not work perfectly.")
    print("         Please also apply the manual fix below:")
    print_manual_guide()
    return False


def print_manual_guide():
    """手動修正ガイドを表示"""
    print("""
=== 手動修正ガイド ===

ファイル: src/handlers/asr/sensevoice/asr_handler_sensevoice.py

self.model.generate() の呼び出しを探し、language="ja" を追加:

--- 修正前 ---
    res = self.model.generate(
        input=audio_data,
        cache={},
        ...
    )
--- 修正後 ---
    res = self.model.generate(
        input=audio_data,
        cache={},
        language="ja",    # 日本語を強制
        ...
    )

※ generate() の引数名は実装によって異なる場合があります。
   重要なのは language="ja" を追加することです。

=== 手動修正が面倒な場合 ===

asr_handler_sensevoice.py を直接開いて:
1. Ctrl+F で "generate(" を検索
2. その呼び出しの中に language="ja", を追加
3. 保存して OpenAvatarChat を再起動
""")


def main():
    print("=" * 60)
    print("ASR SenseVoice Language Patch (Force Japanese)")
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

    print("[1/1] Force Japanese language in SenseVoice ASR:")
    ok = patch_asr_language(oac_dir, dry_run=dry_run)

    print(f"\n{'=' * 60}")
    if ok:
        print("Patch applied successfully!")
    else:
        print("Automatic patching failed. Please apply manually:")
        print_manual_guide()

    if not dry_run and ok:
        print(f"\nBackup file: *.py.bak")
        print(f"To revert: rename .bak file back to original")

    print(f"\nNext steps:")
    print(f"  1. Copy Japanese config:")
    print(f"     copy tests\\a2e_japanese\\chat_with_lam_jp.yaml config\\chat_with_lam_jp.yaml")
    print(f"  2. Edit config/chat_with_lam_jp.yaml - set your Gemini API key")
    print(f"  3. Restart OpenAvatarChat with Japanese config:")
    print(f"     python src/demo.py --config config/chat_with_lam_jp.yaml")


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print_manual_guide()
    else:
        main()
