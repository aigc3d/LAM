# リップシンク改善: 改訂プラン

> **作成日**: 2026-02-24
> **前提**: TEST_PROCEDURE.md の Step 0-4 は未実行だが、ライブ環境での実地テストで同等以上の知見を取得済み

---

## 現状の結論

### ✅ 確認済み（ライブテストで実証）

| 項目 | 結果 |
|------|------|
| A2E は日本語音声で52次元ブレンドシェイプを生成するか？ | **✅ する** — Wav2Vec2は音響レベル動作、言語依存なし |
| jawOpen は発話時に適切に変動するか？ | **✅ する** — A2E raw max=0.28, avg=0.055 |
| mouthFunnel/Pucker は「う」「お」で活性化するか？ | **✅ する** — funnel max=0.37, pucker max=0.49 |
| mouthSmile系は「い」「え」で活性化するか？ | **✅ する** — smile max=0.47, stretch max=0.68 |
| 無音時にリップが閉じるか？ | **✅ 閉じる** — 全チャンネル ~0.00 |

### ❌ 判明した根本問題

**`gaussian-splat-renderer-for-lam` SDK が jawOpen と mouthLowerDown の2チャンネルしかFLAME変形に使わない。**

- A2E の52次元出力は十分な品質 → **データは問題なし**
- SDK が50チャンネルを無視 → **レンダリングがボトルネック**
- 2チャンネルへの合成（remapForSdkLimitation）は構造的限界に到達
  - ハードクランプ → 33%のフレームでクリッピング、動的レンジ消失
  - 重み調整 → 口の開き量しか変わらず、口の「形」は変わらない

### 元の TEST_PROCEDURE.md への影響

| 元のステップ | 状態 | 備考 |
|-------------|------|------|
| Step 0-1: 環境・音声生成 | スキップ可 | ライブ環境で同等確認済み |
| Step 2: A2Eテスト | **合格** | 日本語52次元出力は十分 |
| Step 3-4: 出力保存・分析 | 参考用に実行可 | データ品質は問題なし確認済み |
| Step 4.5: OACパッチ | 完了済み | gourmet-sp統合で適用済み |
| Step 5: 統合テスト | **完了済み（結果:不十分）** | SDK 2ch制限が原因 |
| 判定: 「A2Eが不十分→VHAPへ」 | **該当しない** | A2Eは十分、SDKが不十分 |

---

## 改訂プラン: SDK 2チャンネル制約の突破

### Phase 0: SDK内部調査（最優先・1-2日）

**目的**: SDK が本当に2チャンネルしか使えないのか、設定や未公開APIで拡張可能かを確認

#### 0-1. LAM_WebRender ソース確認
```
GitHub: https://github.com/aigc3d/LAM_WebRender
```
- リポジトリの公開ファイル（README, examples）を精査
- 設定オプション（blendshape mask, expression channel list 等）の有無
- Issue/PR で 52チャンネル対応の議論がないか

#### 0-2. npm パッケージの逆解析
```bash
# gourmet-sp の node_modules から SDK バンドルを取得
# minified JS を beautify して内部構造を調査
```
- Transform Feedback 頂点シェーダーのソースを特定
- expression blendshape のインデックスマッピングを特定
- 「jawOpen」「mouthLowerDown」以外のチャンネルが本当にゼロ扱いか確認

#### 0-3. WebGL シェーダーインターセプト（PoC）
```typescript
// SDK がシェーダーをコンパイルする前にインターセプト
const origShaderSource = WebGL2RenderingContext.prototype.shaderSource;
WebGL2RenderingContext.prototype.shaderSource = function(shader, source) {
  if (source.includes('expression') || source.includes('blendshape')) {
    console.log('[Shader Intercept]', source.substring(0, 500));
    // → シェーダー内容を確認、必要なら改変
  }
  return origShaderSource.call(this, shader, source);
};
```
- SDK の Transform Feedback シェーダーを実行時にキャプチャ
- 52次元のうちどのインデックスが使われているか特定
- 改変可能な箇所を見極める

#### 判定ポイント
| 結果 | 次のアクション |
|------|---------------|
| SDK に設定オプションがある | → Phase 1A（設定変更のみ） |
| シェーダー改変で52ch対応可能 | → Phase 1B（シェーダーパッチ） |
| SDK 改変不可能 | → Phase 2（代替レンダラー） |

---

### Phase 1A: SDK 設定変更（もし設定が見つかった場合）

- SDK初期化パラメータで expression channel list を指定
- LAMAvatar.astro の `remapForSdkLimitation()` を無効化
- 52次元をそのまま渡してテスト

---

### Phase 1B: WebGL シェーダーパッチ（最も現実的な突破口）

**原理**: SDK の Transform Feedback 頂点シェーダーを実行時に書き換え

```
SDK のレンダリングパイプライン:
  [52次元 expression data] → [GPU Texture] → [Vertex Shader (Transform Feedback)]
                                                    ↑ ここを改変
  現状: shader が index[17](jawOpen) と index[37-38](lowerDown) しか読まない
  改変: 全52インデックスを読むように shader を書き換え
```

#### 実装ステップ
1. Phase 0-3 のインターセプトでシェーダーソースを取得
2. expression blendshape テクスチャの読み出しロジックを特定
3. 2チャンネル → 52チャンネル読み出しに改変
4. `remapForSdkLimitation()` を削除、A2E 52次元をそのまま渡す
5. MOUTH_AMPLIFY も最小限に（A2E出力を信頼）

#### リスク
- SDK アップデート時に互換性が壊れる
- シェーダーコンパイルエラーの可能性
- パフォーマンスへの影響（52次元テクスチャ読み出し vs 2次元）

---

### Phase 2: 代替レンダラー（SDK 改変不可能な場合）

#### Option A: Three.js + カスタム FLAME デフォーマー
```
FLAME 変形式（論文より）:
  T_G(θ,φ) = G_bar + B_P(θ;P) + B_E(φ;E)
  Animated_G = S(T_G, J_bar, θ, W)

  G_bar: canonical Gaussian positions (avatar.zip から)
  B_E(φ;E): expression blendshape bases × 52次元係数
  S(): Linear Blend Skinning
```

- avatar.zip から Gaussian データと FLAME パラメータを抽出
- Three.js の Transform Feedback で FLAME LBS を自前実装
- Gaussian Splatting レンダリングは既存 OSS ライブラリを使用
  - [mkkellogg/GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D) をベースに改造
- **メリット**: 完全な制御、52チャンネルフル活用
- **デメリット**: 開発工数大（2-3週間）、パフォーマンスチューニング必要

#### Option B: LAM 公式チームへのリクエスト
- GitHub Issue で 52チャンネル対応を要望
- 現状の2チャンネル制限のエビデンスを添付
- **メリット**: 正規対応なら最も安定
- **デメリット**: 対応時期が不明

#### Option C: Audio2Face-3D など他技術への移行
- NVIDIA Audio2Face-3D: ARKit 52次元フル対応
- Live2D: 2D だが母音表現は高品質
- **メリット**: 実績ある技術
- **デメリット**: LAM Gaussian Splatting の3Dクオリティを捨てる

---

### Phase 3: A2E 出力最適化（Phase 1/2 と並行可能）

52チャンネルレンダリングが実現した後の品質改善:

| 項目 | 内容 |
|------|------|
| EMA α 調整 | 0.85→0.7 でもっとスムーズに |
| 母音別ブースト | smile(い) raw~0.04 は弱すぎ → 個別チューニング |
| 頭の微動 | idle 時の自然な揺れ（現在は完全静止） |
| 瞬き | 定期的な自動瞬きアニメーション |

---

## 推奨アクション順序

```
1. Phase 0-3: WebGL シェーダーインターセプト ← 即日実行可能
   ↓ シェーダーソース取得
2. Phase 0-2: npm パッケージ逆解析
   ↓ 内部構造理解
3. Phase 1B: シェーダーパッチ実装（もし改変可能なら）
   ↓ 52チャンネルレンダリング実現
4. Phase 3: A2E 出力最適化
   ↓
5. 品質評価 → iPhone SE パフォーマンステスト
```

**最速で成果が出るのは Phase 0-3（シェーダーインターセプト）**。
SDK のシェーダーソースを見れば、2チャンネル制限が「意図的な設計」か「単なる実装不足」かが判明し、次の手が決まる。

---

## 元のテスト手順との対応

| 元のステップ | 改訂後 |
|-------------|--------|
| Step 0: 環境チェック | → Phase 0: SDK内部調査 |
| Step 1: テスト音声生成 | → 不要（ライブ環境で確認済み） |
| Step 2: A2Eテスト | → **合格済み** |
| Step 3: 出力保存 | → 参考用（任意） |
| Step 4: 出力分析 | → 参考用（任意） |
| Step 5: 統合テスト | → Phase 1B/2 完了後に再実施 |
| 判定: VHAP fallback | → **不要**（問題はA2Eではなくレンダラー） |
