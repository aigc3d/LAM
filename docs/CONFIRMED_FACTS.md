# 確定事実と否定済み仮説

> **作成日**: 2026-02-25
> **目的**: 後続セッションのClaudeが事実を無視して妄想しないための拘束ファイル
> **ルール**: このファイルに記載された事実を覆すには、新たなブラウザランタイムエビデンスが必要。推測で否定してはならない。

---

## 確定事実（実証済み）

### F1. 音声再生は正常に動作している

- **STT→LLM→TTS パイプライン**: 正常動作。ユーザーの音声入力がSTTでテキスト化され、LLM（Gemini 2.0 Flash）が応答を生成し、Google Cloud TTSが音声合成し、フロントエンドで再生される
- **TTSの読み上げ**: フロントエンドのチャットテキストに表示された内容がTTSで読み上げられている（ユーザー実証済み）
- **TTS synthesize エンドポイント**: HTTP 200 OK を返している（`claude_log_20260224.txt` 6374行）
- **実証回数**: 2026年2月24日だけでも20回以上のテスト実行、ブラウザコンソールログで裏付け済み
- **バグ修正履歴**: TTS再生に関する既知バグは修正済み
  - `2e16f78`: テキスト入力時にTTS再生されない問題を修正
  - `4332c8f`: autoplay deadlock → STT停止の問題を修正（play-and-waitパターン修正）

### F2. A2E 52次元Expressionデータはフロントエンドのバッファまで到達している

- **ブラウザコンソールログ**（`claude_log_20260224.txt` 6296-6303行）:
  ```
  LAMAvatar.astro:543 [LAM Avatar] Added 311 frames to buffer (total: 311) at 60fps
  concierge-controller.ts:443 [Concierge] Expression: 156→311 frames (30→60fps)
    | jaw: max=0.422 avg=0.071 | funnel: max=0.296 | smile: max=0.122

  LAMAvatar.astro:543 [LAM Avatar] Added 617 frames to buffer (total: 617) at 60fps
  concierge-controller.ts:443 [Concierge] Expression: 309→617 frames (30→60fps)
    | jaw: max=0.456 avg=0.073 | funnel: max=0.107 | smile: max=0.183
  ```
- **データの中身**: 空ではない。jawOpen最大0.456、mouthFunnel最大0.296など、有効な値が入っている
- **フレーム数**: 複数チャンクで311, 617フレーム。A2Eバックエンドから正常にデータが返っている
- **フレームレート変換**: 30fps→60fpsの補間処理がフロントエンドで実行されている
- **データフロー**: `audio2exp-service` → `gourmet-support (TTS応答に同梱)` → `concierge-controller.ts (applyExpressionFromTts)` → `lamAvatarController.queueExpressionFrames()` → LAMAvatar バッファ

### F3. アバターの口は動いている（タイミングもほぼ正しい）

- **ユーザー実証済み**: アバターの口はそれっぽく動いており、TTS音声とのタイミングもほぼ合っている
- **つまり**: バッファ→SDK `getExpressionData()` →頂点シェーダーのパイプライン全体が繋がって動作している
- **問題はクオリティ**: 動いてはいるが、リップシンクの質が低い（F9参照）

### F4. skin.glb に51個のARKit morph targetが正常に格納されている

- **検証方法**: pygltflib で `concierge_fne.zip` 内の `skin.glb` を直接解析
- **結果**: 51個のmorph target（sparse accessor形式）
- **各targetに実データあり**: 700〜7,287個の非ゼロ頂点デルタ
  ```
  jawOpen           : sparse count=2755
  eyeBlinkLeft      : sparse count=4491
  cheekPuff         : sparse count=7287
  mouthShrugLower   : sparse count=3539
  （全51 target確認済み — 全てデータあり）
  ```
- **欠損**: `tongueOut`（52個中の1個のみ）。リップシンクには影響なし
- **エビデンスファイル**: `docs/INVESTIGATION_SDK_EXPRESSION_52DIM.md` §1

### F5. SDK（gaussian-splat-renderer-for-lam@0.0.9-alpha.1）はsparse accessorに対応している

- **検証方法**: npmパッケージを展開し、内蔵Three.js GLTFLoader（r173ベース）のソースコードを直接確認
- **該当コード**: `accessorDef.sparse !== undefined` のブランチでsparseIndices/sparseValuesを展開する実装あり
- **エビデンスファイル**: `docs/INVESTIGATION_SDK_EXPRESSION_52DIM.md` §2.5

### F6. SDKの `expressionBSNum` はmorph target数から設定される

- **SDK内部コード**:
  ```javascript
  this.expressionBSNum = this.flameModel.geometry.morphAttributes.position.length;
  this.material.uniforms.bsCount.value = this.expressionBSNum;
  ```
- **理論値**: skin.glbに51個のmorph targetがあれば `expressionBSNum = 51`
- **注意**: ブラウザ実行時に実際に51になっているかは**未検証**（ランタイム確認なし）
- **エビデンスファイル**: `docs/INVESTIGATION_SDK_EXPRESSION_52DIM.md` §2.3

### F7. SDKのExpression処理フロー（コードレベルで確認済み）

- **毎フレームの処理**:
  1. `getExpressionData()` コールバック → `{ jawOpen: 0.45, mouthFunnel: 0.12, ... }`
  2. `setExpression()` → `splatMesh.bsWeight = expressionData`
  3. `updateBoneMatrixTexture()` → `morphTargetDictionary[name]` でindex取得 → GPUテクスチャにパック
  4. Vertex Shader → `for(int i = 0; i < bsCount; ++i)` ループでblendshape適用
- **名前ベースのマッピング**: SDKは配列indexではなく名前で辞書検索。順序非依存
- **エビデンスファイル**: `docs/INVESTIGATION_SDK_EXPRESSION_52DIM.md` §2.2, §2.4

### F8. audio2exp-service は Cloud Run にデプロイ済み、ヘルスチェック通過

- **URL**: `https://audio2exp-service-417509577941.us-central1.run.app`
- **ヘルスチェック**: `engine_ready: true`（`claude_log_20260224.txt` 内で確認）
- **メモリ**: 4Gi（2Giでは3回OOM、4Giで完走）
- **出力**: 52次元ARKit blendshape @ 30fps

### F9. リップシンクのクオリティが低い（日本語・英語とも）

- **ユーザー実証済み**: 日本語も英語も同様にクオリティが低い
- **言語差なし**: A2Eモデル（Wav2Vec2ベース）は音響ベースで動作するため、言語による品質差は小さい。両方とも低いのは言語の問題ではなくパイプライン全体の問題

### F10. ブラウザログの `_Vector3 12248829 0` の `0` は `expressionBSNum` ではない

- **正体**: SDK内部の `console.log(cameraPos, backgroundColor, alpha)` の出力
  - `_Vector3` = cameraPos（Vector3オブジェクト）
  - `12248829` = backgroundColor（parseInt結果）
  - `0` = alpha値（透明度パラメータ）
- **エビデンスファイル**: `docs/INVESTIGATION_SDK_EXPRESSION_52DIM.md` §3

---

## 否定済みの仮説（再提示禁止）

| # | 仮説 | 否定理由 | エビデンス |
|---|------|----------|-----------|
| H1 | 音声が再生されていない / audioフィールドが空 | **音声は正常再生されている。** STT→LLM→TTSパイプラインは動作し、チャットテキストがTTSで読み上げられている。20回以上のテストで実証済み | ユーザー実証、コミット `2e16f78` `4332c8f` |
| H2 | skin.glbにmorph targetがない | 51個のmorph targetが実データ付きで格納されている | pygltflib解析、`INVESTIGATION_SDK_EXPRESSION_52DIM.md` §1 |
| H3 | SDKがsparse accessorに非対応 | Three.js r173 GLTFLoaderに対応コードあり | `INVESTIGATION_SDK_EXPRESSION_52DIM.md` §2.5 |
| H4 | `expressionBSNum = 0` | `0`はalpha（透明度）パラメータ | `INVESTIGATION_SDK_EXPRESSION_52DIM.md` §3 |
| H5 | A2Eバックエンドがデータを返していない | 311, 617フレームがフロントエンドバッファに到達 | `claude_log_20260224.txt` 6296-6303行 |
| H6 | アバターの口が動いていない | **口は動いている。** タイミングもほぼ正しい。問題は動かないことではなくクオリティが低いこと | ユーザー実証済み |

---

## 未解決の問題（原因未特定）

### 核心的な問題

**リップシンクのクオリティが低い。** パイプライン全体は繋がって動いている（音声再生、Expressionデータ到達、口の動き、タイミング全てOK）が、口の動きの質が不十分。

### 品質が低い原因の候補（要調査）

以下は仮説ではなく、「まだ検証していない領域」の列挙。

1. **A2Eモデルの出力品質**: Wav2Vec2 → A2E Decoderの出力するblendshape係数自体の精度。jawOpen max=0.456 は十分か、他のblendshapeの値域は適切か
2. **blendshape増幅パラメータの調整**: `concierge-controller.ts` の `MOUTH_AMPLIFY` 係数が最適かどうか
3. **フレーム補間の品質**: 30fps→60fps線形補間が滑らかさに十分か
4. **SDKの `expressionBSNum` のランタイム値**: 理論上51だが、ブラウザで実測していない。仮に少ない数値だと一部blendshapeが無視される
5. **A2Eモデルが口以外のblendshapeを十分に活用しているか**: 眉、目、頬などの表情パラメータが生成されているか

---

## このファイルの使い方

1. 新しいセッションの最初に必ずこのファイルを読む
2. §否定済みの仮説 に記載された仮説を再提示しない
3. §未解決の問題 の検証から作業を開始する
4. 新たな事実が判明したら、このファイルを更新する
5. **推測で事実を覆さない。エビデンスがなければ「不明」と書く**
