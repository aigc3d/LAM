# LAMAvatar.astro リップシンク改善: SDK 2チャンネル制約の回避

> **作成日**: 2026-02-24
> **更新日**: 2026-02-24
> **問題**: `gaussian-splat-renderer-for-lam` SDK が jawOpen と mouthLowerDown の2値しか FLAME メッシュ変形に使用しない
> **影響**: 母音の口形状区別（あいうえお）ができない。口がパクパクするだけ
> **対策**: `remapForSdkLimitation()` で母音チャンネルを jaw/lowerDown に合成

## 1. 問題の証拠

`__testLipSync()` 診断結果（パッチ前）:

| 母音 | jawOpen | smile | funnel | pucker | stretch | lowerDown | **jaw結果** | **mouth結果** |
|---|---|---|---|---|---|---|---|---|
| あ | 0.7 | - | - | - | - | 0.5 | 0.700 | 0.500 |
| い | 0.2 | 0.6 | - | - | 0.4 | - | 0.200 | **0.000** |
| う | 0.15 | - | 0.6 | 0.5 | - | - | 0.150 | **0.000** |
| え | 0.4 | 0.3 | - | - | 0.5 | 0.3 | 0.400 | 0.300 |
| お | 0.5 | - | 0.5 | 0.3 | - | 0.2 | 0.500 | 0.200 |

**結論**: SDK は `jawOpen` と `mouthLowerDown` のみ FLAME 変形に使用。他は全て無視。

### 誤った初期仮説（訂正）

当初「LAMAvatar.astro が2値しか返していない」と推定していたが、実際のコード確認で
`getExpressionData()` は **全52次元をそのまま返している** ことが判明。
ボトルネックは LAMAvatar.astro ではなく、**クローズドソースの SDK 側** にある。

## 2. 修正内容: `remapForSdkLimitation()`

SDK が2チャンネルしか使わない制約を、LAMAvatar.astro 側で回避する。
母音チャンネル（smile, funnel, pucker, stretch）の情報を jawOpen と mouthLowerDown に合成。

### 合成ロジック

```typescript
// jawOpen 合成: 母音チャンネルの寄与を加算
compositeJaw = min(0.7,
  jawOpen
  + smile * 0.5      // い: 口角引きに伴う顎の動き
  + funnel * 0.35    // う/お: 唇突き出し
  + pucker * 0.2     // う: 唇すぼめ
  + stretch * 0.3    // え: 口横伸ばし
)

// mouthLowerDown 合成: 母音特性を下唇の動きに変換
compositeMouth = min(0.7, max(0,
  lowerDown
  + smile * 0.6      // い: 口角引き → 下唇の動き
  + stretch * 0.4    // え: 口横伸ばし → 下唇の動き
  - pucker * 0.15    // う: 唇すぼめ → 下唇を閉じる方向
))
```

### 合成後の期待値（A2Eブースト後）

| 母音 | jawOpen (合成前→後) | mouthLowerDown (合成前→後) | 変化 |
|---|---|---|---|
| あ | 0.70 → 0.70 | 0.42 → 0.42 | 変化なし（既に十分） |
| い | 0.05 → 0.11 | 0.00 → 0.07 | smile寄与で微動が見える |
| う | 0.03 → 0.18 | 0.00 → 0.00 | funnel/pucker寄与で顎が動く |
| え | 0.08 → 0.20 | ~0 → 0.15 | stretch/smile寄与で中程度 |
| お | 0.10 → 0.21 | ~0 → ~0 | funnel寄与でうより大きく開口 |

### 適用箇所

`getExpressionData()` の3つの return パス:
1. TTS-Sync フレーム読み出し（通常パス）
2. バッファ末尾超過（最終フレーム保持）
3. ※フェードアウト時は既に縮小中のため適用不要

## 3. その他の改善

### 診断ログの拡張

- Health check: jaw, mouth に加えて funnel, smile, pucker を表示
- TTS-Sync: 10フレームごとに funnel, smile, stretch も表示
- concierge-controller.ts: 既に全母音チャンネルの統計値をログ出力済み

### 確認方法

修正適用後、`__testLipSync()` を再実行:
- 5母音で jaw/mouth 値が **全て異なる** ことをログで確認
- 視覚的に い(jaw小+mouth微) と う(jaw中+mouth0) で異なる口の動きが見えれば成功

## 4. 今後の改善余地

1. **SDK アップデート待ち**: SDK が52次元対応すれば `remapForSdkLimitation()` を無効化
2. **A2Eモデルの改善**: 日本語母音の smile/funnel 出力が弱すぎる（smile raw ~0.04）→ より高品質なモデルへ
3. **NVIDIA Audio2Face-3D**: ARKit 52次元をフル活用できるレンダリングパイプラインへの移行
