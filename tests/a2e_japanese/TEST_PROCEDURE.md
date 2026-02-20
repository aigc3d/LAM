# A2E + 日本語音声テスト手順

## 目的

A2E (Audio2Expression) が日本語音声で十分なリップシンクを生成するか検証する。
もし生成できるなら、公式HF SpacesのZIP（英語/中国語参照）をそのまま使え、
ZIPのmotion差し替えやVHAP、Modal問題を全てスキップできる。

## 前提条件

| 項目 | 状態 |
|------|------|
| OpenAvatarChat | `C:\Users\hamad\OpenAvatarChat` にインストール済み |
| conda環境 | `oac` (Python 3.11) |
| Gemini API | 設定済み |
| EdgeTTS | `ja-JP-NanamiNeural` |
| LAM_audio2exp モデル | ダウンロード済み |
| wav2vec2-base-960h | ダウンロード済み |
| SenseVoiceSmall | ダウンロード済み |
| GPU | なし（CPU mode） |
| 公式HF Spaces ZIP | `lam_samples/concierge.zip` |

## テスト手順

### Step 0: 環境チェック

```powershell
cd C:\Users\hamad\OpenAvatarChat
conda activate oac
python tests/a2e_japanese/setup_oac_env.py
```

問題がある場合は指示に従って修正。

### Step 1: テスト音声生成

```powershell
python tests/a2e_japanese/generate_test_audio.py
```

以下のWAVファイルが `tests/a2e_japanese/audio_samples/` に生成される:

| ファイル | 内容 | 目的 |
|----------|------|------|
| `vowels_aiueo.wav` | あ、い、う、え、お | 母音のリップシェイプ |
| `greeting_konnichiwa.wav` | こんにちは、お元気ですか？ | 自然な会話 |
| `long_sentence.wav` | AIコンシェルジュの定型文 | 長文テスト |
| `mixed_phonemes.wav` | さしすせそ、たちつてと... | 子音+母音 |
| `numbers_and_names.wav` | 東京タワー、富士山 | 固有名詞 |
| `english_compare.wav` | Hello, how are you? | 英語比較 |
| `chinese_compare.wav` | 你好，我是AI助手 | 中国語比較 |
| `silence_baseline.wav` | 無音 2秒 | ベースライン |
| `tone_440hz.wav` | 440Hz正弦波 1秒 | 非音声参照 |

### Step 2: A2Eテスト実行

```powershell
python tests/a2e_japanese/test_a2e_cpu.py
```

テスト内容:
1. **モデルロード確認** - 全モデルファイルの存在チェック
2. **Wav2Vec2特徴量抽出** - 日本語音声からの特徴量生成
3. **A2E推論** - 52次元ARKitブレンドシェイプ出力
4. **ブレンドシェイプ分析** - リップ関連の活性度
5. **ZIP構造検証** - 公式ZIPの整合性

### Step 3: ブレンドシェイプ出力保存

```powershell
python tests/a2e_japanese/save_a2e_output.py
```

### Step 4: 出力分析

```powershell
python tests/a2e_japanese/analyze_blendshapes.py --input-dir tests/a2e_japanese/blendshape_outputs/
```

### Step 4.5: パッチ適用（初回のみ）

OpenAvatarChatのハンドラーにバグ修正・日本語対応パッチを適用する。

```powershell
# ASR: 日本語言語強制（中国語誤検出の修正）
python tests/a2e_japanese/patch_asr_language.py

# VAD/ASR: numpy dtype修正
python tests/a2e_japanese/patch_vad_handler.py

# LLM: Gemini dict content修正
python tests/a2e_japanese/patch_llm_handler.py
```

パッチが自動適用できない場合は `--help` で手動修正ガイドを表示:
```powershell
python tests/a2e_japanese/patch_asr_language.py --help
```

### Step 5: OpenAvatarChatでの統合テスト

```powershell
# configをコピー
copy tests\a2e_japanese\chat_with_lam_jp.yaml config\chat_with_lam_jp.yaml

# Gemini APIキーを設定（既に設定済みの場合はスキップ）
# config/chat_with_lam_jp.yaml の api_key を編集

# 起動（※ chat_with_lam.yaml ではなく _jp.yaml を指定）
python src/demo.py --config config/chat_with_lam_jp.yaml
```

ブラウザで `https://localhost:8282` を開き、以下をテスト:

| テスト | 操作 | 観察ポイント |
|--------|------|-------------|
| テストA | 英語参照ZIP + 日本語で話す | 口の動きが日本語の母音に合うか |
| テストB | 中国語参照ZIP + 日本語で話す | テストAと差があるか |
| テストC | 同じZIPで英語で話す | 日本語との差があるか |

## 全テスト一括実行

```powershell
python tests/a2e_japanese/run_all_tests.py
```

## 判定基準

### A2Eが日本語で十分な場合（Step 2へ進む必要なし）
- jawOpen が発話時に適切に変動
- mouthFunnel/mouthPucker が「う」「お」で活性化
- mouthSmile系が「い」「え」で活性化
- 無音時にリップが閉じる
- 英語テストとの品質差が小さい

### A2Eが日本語で不十分な場合（Step 2: ZIP解析 + VHAPへ）
- リップが発話に追従しない
- 母音の区別ができない
- 英語と比べて明らかに品質が低い

## ファイル構成

```
tests/a2e_japanese/
├── __init__.py
├── TEST_PROCEDURE.md          # この文書
├── chat_with_lam_jp.yaml      # OpenAvatarChat設定ファイル
├── generate_test_audio.py     # テスト音声生成
├── test_a2e_cpu.py            # A2Eテストスイート
├── save_a2e_output.py         # A2E推論出力保存
├── analyze_blendshapes.py     # ブレンドシェイプ分析
├── setup_oac_env.py           # 環境チェック・修正
├── run_all_tests.py           # 全テスト一括実行
├── audio_samples/             # 生成されたテスト音声 (gitignore)
│   ├── vowels_aiueo.wav
│   ├── greeting_konnichiwa.wav
│   └── ...
└── blendshape_outputs/        # A2E出力 (gitignore)
    ├── vowels_aiueo.npy
    └── ...
```

## A2Eアーキテクチャ（参考）

```
音声入力 (WAV, 24kHz)
    ↓
[Wav2Vec2] (facebook/wav2vec2-base-960h)
    ↓ 音響特徴量 (T, 768)
    ↓ ※言語パラメータなし、音響レベルで動作
    ↓
[A2Eデコーダー] (LAM_audio2exp)
    ↓ 52次元 ARKit ブレンドシェイプ (T', 52)
    ↓
[OpenAvatarChat WebGL Renderer]
    ↓ skin.glb の頂点を変形
    ↓ vertex_order.json でマッピング
    ↓
アバター表示
```

重要: Wav2Vec2は音響レベルで動作し、言語パラメータはゼロ。
理論上、どの言語の音声でもブレンドシェイプを生成可能。
