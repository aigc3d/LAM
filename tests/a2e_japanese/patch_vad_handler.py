"""
VAD ハンドラー修正パッチ

RuntimeError: Input data type <class 'list'> is not supported.
の原因を特定・修正するためのパッチ。

使い方（2通り）:

方法A: 直接適用（推奨）
    vad_handler_silero.py を直接編集する。
    このスクリプトの「修正内容」セクションを参照。

方法B: モンキーパッチ（デバッグ用）
    OpenAvatarChatの起動前に以下を実行:
    cd C:\\Users\\hamad\\OpenAvatarChat
    python tests/a2e_japanese/patch_vad_handler.py

修正内容:
    1. timestamp[0] の NoneType エラー修正
    2. ONNX入力の防御的 numpy 変換
    3. エラー発生時の詳細ログ追加
    4. SenseVoice の dtype 不一致修正
"""

import os
import re
import shutil
import sys
from pathlib import Path


# ============================================================
# 修正1: vad_handler_silero.py の handle() メソッド
# ============================================================

VAD_HANDLER_PATCHES = [
    {
        "description": "Fix timestamp[0] NoneType crash",
        "file": "src/handlers/vad/silerovad/vad_handler_silero.py",
        "find": "        context.slice_context.update_start_id(timestamp[0], force_update=False)",
        "replace": """        if timestamp is not None:
            context.slice_context.update_start_id(timestamp[0], force_update=False)
        else:
            context.slice_context.update_start_id(0, force_update=False)""",
    },
    {
        "description": "Add defensive numpy conversion in _inference",
        "file": "src/handlers/vad/silerovad/vad_handler_silero.py",
        "find": """    def _inference(self, context: HumanAudioVADContext, clip: np.ndarray, sr: int=16000):
        clip = clip.squeeze()
        if clip.ndim != 1:
            logger.warning("Input audio should be 1-dim array")
            return 0
        clip = np.expand_dims(clip, axis=0)
        inputs = {
            "input": clip,
            "sr": np.array([sr], dtype=np.int64),
            "state": context.model_state
        }
        prob, state = self.model.run(None, inputs)
        context.model_state = state
        return prob[0][0]""",
        "replace": """    def _inference(self, context: HumanAudioVADContext, clip: np.ndarray, sr: int=16000):
        # Ensure clip is a numpy array (defensive check)
        if not isinstance(clip, np.ndarray):
            logger.warning(f"VAD input clip is {type(clip).__name__}, converting to numpy")
            clip = np.array(clip, dtype=np.float32)
        clip = clip.squeeze()
        if clip.ndim != 1:
            logger.warning("Input audio should be 1-dim array")
            return 0
        clip = np.expand_dims(clip, axis=0).astype(np.float32)
        # Ensure model_state is a numpy array (defensive check)
        if context.model_state is None:
            context.model_state = np.zeros((2, 1, 128), dtype=np.float32)
        elif not isinstance(context.model_state, np.ndarray):
            logger.warning(f"VAD model_state is {type(context.model_state).__name__}, converting to numpy")
            context.model_state = np.array(context.model_state, dtype=np.float32)
        inputs = {
            "input": clip,
            "sr": np.array([sr], dtype=np.int64),
            "state": context.model_state
        }
        try:
            ort_outputs = self.model.run(None, inputs)
            if len(ort_outputs) == 2:
                prob, state = ort_outputs
            elif len(ort_outputs) == 3:
                # Silero VAD v5 may have 3 outputs: prob, hn, cn
                prob = ort_outputs[0]
                state = np.stack([ort_outputs[1], ort_outputs[2]])
            else:
                prob = ort_outputs[0]
                state = context.model_state  # keep current state
            # Ensure state remains a numpy array
            if not isinstance(state, np.ndarray):
                state = np.array(state, dtype=np.float32)
            context.model_state = state
            return prob.flatten()[0]
        except RuntimeError as e:
            logger.error(f"ONNX RuntimeError in VAD: {e}")
            logger.error(f"  input type={type(clip).__name__}, dtype={clip.dtype}, shape={clip.shape}")
            logger.error(f"  state type={type(context.model_state).__name__}")
            if isinstance(context.model_state, np.ndarray):
                logger.error(f"  state dtype={context.model_state.dtype}, shape={context.model_state.shape}")
            # Reset state and return 0 (no speech) to avoid crash loop
            context.model_state = np.zeros((2, 1, 128), dtype=np.float32)
            return 0""",
    },
]

# ============================================================
# 修正2: asr_handler_sensevoice.py の dtype 修正
# ============================================================

ASR_HANDLER_PATCHES = [
    {
        "description": "Fix np.zeros dtype mismatch in SenseVoice handler",
        "file": "src/handlers/asr/sensevoice/asr_handler_sensevoice.py",
        "find": "                remainder_audio = np.concatenate(\n                    [remainder_audio,\n                     np.zeros(shape=(context.audio_slice_context.slice_size - remainder_audio.shape[0]))])",
        "replace": "                remainder_audio = np.concatenate(\n                    [remainder_audio,\n                     np.zeros(shape=(context.audio_slice_context.slice_size - remainder_audio.shape[0]),\n                              dtype=remainder_audio.dtype)])",
    },
]


def apply_patches(oac_dir: Path, patches: list, dry_run: bool = False) -> int:
    """パッチを適用する"""
    applied = 0

    for patch in patches:
        filepath = oac_dir / patch["file"]
        if not filepath.exists():
            print(f"  [SKIP] {patch['file']} not found")
            continue

        content = filepath.read_text(encoding="utf-8")

        if patch["find"] not in content:
            if patch["replace"] in content:
                print(f"  [ALREADY] {patch['description']}")
                applied += 1
                continue
            else:
                print(f"  [WARN] Cannot find target text for: {patch['description']}")
                print(f"         File may have been modified. Manual patching required.")
                continue

        if dry_run:
            print(f"  [DRY-RUN] Would apply: {patch['description']}")
            applied += 1
            continue

        # バックアップ作成
        backup_path = filepath.with_suffix(filepath.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy2(filepath, backup_path)
            print(f"  Backup: {backup_path}")

        # パッチ適用
        new_content = content.replace(patch["find"], patch["replace"], 1)
        filepath.write_text(new_content, encoding="utf-8")
        print(f"  [APPLIED] {patch['description']}")
        applied += 1

    return applied


def main():
    print("=" * 60)
    print("VAD Handler Patch Tool")
    print("=" * 60)

    # OACディレクトリ解決
    if len(sys.argv) > 1 and sys.argv[1] == "--dry-run":
        dry_run = True
    else:
        dry_run = False

    oac_dir = None
    for candidate in [
        Path(r"C:\Users\hamad\OpenAvatarChat"),
        Path.home() / "OpenAvatarChat",
        Path.cwd(),
    ]:
        if (candidate / "src" / "handlers").exists():
            oac_dir = candidate
            break

    if oac_dir is None:
        print("ERROR: OpenAvatarChat directory not found")
        print("Run from the OpenAvatarChat directory or specify path")
        sys.exit(1)

    print(f"OAC: {oac_dir}")
    if dry_run:
        print("Mode: DRY RUN (no changes will be made)")
    else:
        print("Mode: APPLY PATCHES")
    print()

    # VAD handler patches
    print("[1/2] VAD Handler Patches:")
    vad_applied = apply_patches(oac_dir, VAD_HANDLER_PATCHES, dry_run=dry_run)

    # ASR handler patches
    print(f"\n[2/2] ASR Handler Patches:")
    asr_applied = apply_patches(oac_dir, ASR_HANDLER_PATCHES, dry_run=dry_run)

    total = vad_applied + asr_applied
    print(f"\n{'=' * 60}")
    print(f"Applied {total} patch(es)")

    if not dry_run and total > 0:
        print(f"\nBackup files created with .bak extension.")
        print(f"To revert: rename .bak files back to originals.")

    print(f"\nNext: Restart OpenAvatarChat and test voice input:")
    print(f"  python src/demo.py --config config/chat_with_lam_jp.yaml")


# ============================================================
# 手動修正ガイド（コピペ用）
# ============================================================

MANUAL_FIX_GUIDE = """
=== 手動修正ガイド ===

もしパッチスクリプトが動かない場合、以下を手動で修正:

【ファイル1】 src/handlers/vad/silerovad/vad_handler_silero.py

修正箇所A: handle() メソッド内の timestamp[0] 修正
--- 修正前 ---
        context.slice_context.update_start_id(timestamp[0], force_update=False)
--- 修正後 ---
        if timestamp is not None:
            context.slice_context.update_start_id(timestamp[0], force_update=False)
        else:
            context.slice_context.update_start_id(0, force_update=False)

修正箇所B: _inference() メソッドの防御的チェック追加
--- _inference の先頭に追加 ---
        if not isinstance(clip, np.ndarray):
            clip = np.array(clip, dtype=np.float32)
--- model_state チェック追加（inputs = { の前に追加） ---
        if context.model_state is None:
            context.model_state = np.zeros((2, 1, 128), dtype=np.float32)
        elif not isinstance(context.model_state, np.ndarray):
            context.model_state = np.array(context.model_state, dtype=np.float32)

【ファイル2】 src/handlers/asr/sensevoice/asr_handler_sensevoice.py

修正箇所: np.zeros に dtype 追加
--- 修正前 ---
                     np.zeros(shape=(context.audio_slice_context.slice_size - remainder_audio.shape[0]))])
--- 修正後 ---
                     np.zeros(shape=(context.audio_slice_context.slice_size - remainder_audio.shape[0]),
                              dtype=remainder_audio.dtype)])
"""


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(MANUAL_FIX_GUIDE)
    else:
        main()
