# LAMAvatar.astro パッチ: 52次元ブレンドシェイプ対応

> **作成日**: 2026-02-24
> **問題**: LAMAvatar.astro が jawOpen と mouthLowerDown の2値しかレンダラーに渡しておらず、他の50個のARKitブレンドシェイプ（funnel, pucker, smile, stretch 等）が無視されている
> **影響**: 母音の口形状区別（あいうえお）が全くできない。口がパクパクするだけ

## 1. 問題の証拠

`__testLipSync()` 診断結果:

| 母音 | jawOpen | smile | funnel | pucker | stretch | lowerDown | **jaw結果** | **mouth結果** |
|---|---|---|---|---|---|---|---|---|
| あ | 0.7 | - | - | - | - | 0.5 | 0.700 | 0.500 |
| い | 0.2 | 0.6 | - | - | 0.4 | - | 0.200 | **0.000** |
| う | 0.15 | - | 0.6 | 0.5 | - | - | 0.150 | **0.000** |
| え | 0.4 | 0.3 | - | - | 0.5 | 0.3 | 0.400 | 0.300 |
| お | 0.5 | - | 0.5 | 0.3 | - | 0.2 | 0.500 | 0.200 |

**結論**: `jaw = jawOpen`, `mouth = mouthLowerDown` のみ。他は全て無視。

## 2. 修正箇所

### LAMAvatar.astro の `getExpressionData()` コールバック

**現状** (推定):
```typescript
// GaussianSplatRenderer の getExpressionData コールバック
getExpressionData: () => {
  const frame = getCurrentFrame(); // frameBuffer から現在フレーム取得
  return {
    jawOpen: frame['jawOpen'] || 0,
    // ← mouthLowerDown を何らかの形で mouth に変換
  };
}
```

**修正後**: フレームの全ブレンドシェイプをそのまま返す
```typescript
getExpressionData: () => {
  const frame = getCurrentFrame();
  if (!frame) return {};
  // 52次元の全ブレンドシェイプをそのまま返す
  // WebGL SDK の FLAME blendshape デコーダーが各値を処理
  return { ...frame };
}
```

### 診断ログも拡張

**現状のログ** (LAMAvatar.astro:454):
```typescript
console.log(`[LAM TTS-Sync] Frame ${idx}/${total}: jaw=${jawOpen}, mouth=${mouth}, time=${time}ms`);
```

**修正後**: 主要母音チャンネルも表示
```typescript
const f = frame;
console.log(
  `[LAM TTS-Sync] Frame ${idx}/${total}: ` +
  `jaw=${(f.jawOpen||0).toFixed(3)}, ` +
  `funnel=${(f.mouthFunnel||0).toFixed(3)}, ` +
  `smile=${(f.mouthSmileLeft||0).toFixed(3)}, ` +
  `stretch=${(f.mouthStretchLeft||0).toFixed(3)}, ` +
  `lowerDn=${(f.mouthLowerDownLeft||0).toFixed(3)}, ` +
  `time=${time}ms`
);
```

## 3. WebGL SDK (`gaussian-splat-renderer-for-lam`) の対応確認

SDK のドキュメントに記載されている `getExpressionData` コールバック:
```typescript
GaussianSplatRenderer.getInstance(container, assetPath, {
  getExpressionData: () => ({ jawOpen: 0.5, mouthFunnel: 0.2, ... }),
});
```

`...` は複数のブレンドシェイプ名をキーとして返せることを示唆。
SDK 内部で FLAME の expression blendshape (52次元) に対応する頂点変形を計算しているはず。

### 確認方法

LAMAvatar.astro を修正後、`__testLipSync()` を再実行:
- い(smile=0.6) と う(funnel=0.6) で**異なる口形状**が表示されれば → SDK は52次元対応
- 変わらなければ → SDK 自体が jawOpen のみ対応（npmパッケージの制約）

## 4. SDK が jawOpen のみ対応の場合の代替案

SDK が52次元に対応していない場合:
1. **ExpressionManager方式に切り替え**: `gvrm.updateLipSync(mouthOpenness)` の公式を改良
   - 現状: jawOpen×0.6 + lowerDown×0.2 + upperUp×0.1 + funnel×0.05 + pucker×0.05
   - 改良: 母音ごとに異なる重みで mouthOpenness を計算
2. **Three.js + GLBメッシュ**: Gaussian Splatting を捨てて、通常のメッシュ + 52 ARKit ブレンドシェイプ
3. **NVIDIA Audio2Face-3D**: より高品質な A2E モデルに切り替え
