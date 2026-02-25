# 調査報告: LAM WebGL SDK — 52次元 Expression Blendshape 処理

調査日: 2026-02-25

## 結論

**公式ModelScope SpaceのZIPに不備はない。** skin.glb に51個のARKit morph targetが正常に格納されており、SDKは正しく読み取れる設計になっている。

---

## 1. 公式ZIPの実データ検証

### ZIP構成 (`concierge_fne.zip` — ModelScope Space生成)

| ファイル | サイズ | 内容 |
|---------|-------|------|
| `skin.glb` | 3.6MB | 20,018頂点, 262ボーン, **51 morph targets (sparse)** |
| `offset.ply` | 1.3MB | 20,018 Gaussians × 17属性 (pos/color/opacity/scale/rotation) |
| `animation.glb` | 2.2MB | 12個のボーンアニメーション (idle, speak, think等) |
| `vertex_order.json` | 209KB | 頂点順序マッピング |

### skin.glb morph target 詳細

- **51個** のARKit blendshape（52個中 `tongueOut` のみ欠損）
- 全て **sparse accessor** 形式（glTF2.0仕様準拠、効率的な格納）
- 各targetに700〜7,287個の非ゼロ頂点デルタ（実データ確認済み）

```
mouthShrugLower   : sparse count=3539
jawOpen           : sparse count=2755
eyeBlinkLeft      : sparse count=4491
cheekPuff         : sparse count=7287
（全51 target確認済み — 全てデータあり）
```

### 欠損blendshape

| 名前 | 状態 | 影響 |
|------|------|------|
| `tongueOut` | 欠損 | 舌を出す表情のみ不可。リップシンクには影響なし |

---

## 2. SDK ソースコード解析 (`gaussian-splat-renderer-for-lam@0.0.9-alpha.1`)

npmパッケージを直接展開して確認。

### 2.1 アーキテクチャ: 2つのモード

```javascript
// SDK内部 (line ~152272)
var useFlame = "false";  // ← ハードコード
renderer.useFlame = (charactorConfig.useFlame == "false") ? false : true;

if (renderer.viewer.useFlame == true) {
    yield renderer.loadFlameModel(fileName, motionConfig);
    // → skin.glb + lbs_weight_20k.json + flame_params.json + vertex_order.json + bone_tree.json
} else {
    yield renderer.loadModel(fileName, animationConfig, motionConfig);
    // → skin.glb + animation.glb + vertex_order.json  ← こちらが使われる
}
```

**現在のSDKは `useFlame=false` がハードコード**されている。OAC ZIPはこのモード用。

### 2.2 Expression Blendshape 処理フロー (useFlame=false)

```
[毎フレーム]
1. getExpressionData() callback
   → { jawOpen: 0.45, mouthFunnel: 0.12, ... } (52次元)

2. setExpression()
   → splatMesh.bsWeight = expressionData  (名前→重み辞書)

3. updateBoneMatrixTexture()
   → morphTargetDictionary[name] でindexを取得
   → boneTexture[idx + bonesNum*16] = weight  (GPUテクスチャにパック)

4. Vertex Shader (GPU)
   for(int i = 0; i < bsCount; ++i) {
       float weight = boneTexture[i / 4 + 5 * 4][i % 4];
       splatCenter += weight * flameModelTexture[i];  // BS基底 × 重み
   }
```

### 2.3 expressionBSNum の設定タイミング

```javascript
// setupDataTextures() — offset.ply ロード後に実行
this.expressionBSNum = this.flameModel.geometry.morphAttributes.position.length;
this.material.uniforms.bsCount.value = this.expressionBSNum;
```

この時点で `flameModel` は既にskin.glbから読み込み済みなので、
morph targetが正常にロードされていれば `expressionBSNum = 51`。

### 2.4 buildModelTexture — morph target をGPUテクスチャにパック

```javascript
// 各morph targetの頂点データをflatに連結 → 4096x2048 テクスチャへ
morphTargetNames.forEach((name, newIndex) => {
    const originalIndex = flameModel.morphTargetDictionary[name];
    var bsMesh = flameModel.geometry.morphAttributes.position[originalIndex];
    shapedMeshArray = shapedMeshArray.concat(Array.from(bsMesh.array));
});
// ベースメッシュも追加
shapedMeshArray = shapedMeshArray.concat(Array.from(shapedMesh));
```

**→ SDKは morph target名前ベースで辞書検索。順序非依存。**

### 2.5 Three.js GLTFLoader — sparse accessor 対応済み

```javascript
// SDK内蔵のGLTFLoader (Three.js r173ベース)
if ( accessorDef.sparse !== undefined ) {
    const sparseIndices = new TypedArrayIndices(bufferViews[1], ...);
    const sparseValues = new TypedArray(bufferViews[2], ...);
    for (let i = 0; i < sparseIndices.length; i++) {
        bufferAttribute.setX(sparseIndices[i], sparseValues[i * itemSize]);
        // ... setY, setZ
    }
}
```

**→ sparse accessor は正しく展開される。**

---

## 3. ブラウザログの再解釈

```
gaussian-splat-renderer-for-lam.js:62550 download completed: ArrayBuffer(4094984)
gaussian-splat-renderer-for-lam.js:62588 _Vector3 12248829 0
```

この `_Vector3 12248829 0` は SDK内部の `console.log(cameraPos, backgroundColor, alpha)` の出力:
- `_Vector3` = cameraPos (Vector3オブジェクト)
- `12248829` = backgroundColor (parseInt結果)
- `0` = alpha値

**`0` は `expressionBSNum` ではなく、透明度(alpha)パラメータ。**

---

## 4. 真の問題: バックエンドがExpression dataを返していない

```
concierge-controller.ts:303 [Concierge] TTS response has NO expression data
LAMAvatar.astro:195 [LAM Health] state=Idle, jaw=0.000, mouth=0.000, buffer=0
```

**SDKもZIPも正常。問題はバックエンド側:**
- audio2exp-service のヘルスチェックがNG（CLAUDE.md記載通り）
- TTSレスポンスに expression data が含まれていない
- → フロントエンドの `getExpressionData()` が空データを返す
- → リップシンクが動かない

---

## 5. 副次的な発見

### 5.1 flame_arkit.py assertion バグ（本番影響なし）

```python
# flame_arkit.py:108
assert expr_params != 52, "The dimension of the ARKIT expression must be equal to 52."
# ↑ != は == であるべき。ただしこのモデルはOACパスでは使われないため本番影響なし。
```

### 5.2 h5_rendering パス（無効化済み）

`app_lam.py:42` で `h5_rendering = False`。このパスは：
- 100個のFLAME標準expression（52次元ARKitではない）
- `lbs_weight_20k.json` + `bone_tree.json` + `flame_params.json` を生成
- `useFlame=true` モード用
- 現在無効化

### 5.3 OACパスのZIP生成

`app_lam.py:304-342`:
- template FBX（ARKit blendshape内蔵）からskin.glbを生成
- animation.glb は固定ファイルをコピー
- **ZIPにExpression基底データは正しく含まれる**（template FBXに51個のblendshape内蔵済み）

---

## 6. 次のアクション

**最優先: audio2exp-service のヘルスチェックNG解決**

1. audio2exp-service が正常にレスポンスを返すようにする
2. バックエンドTTSエンドポイントで `AUDIO2EXP_SERVICE_URL` が正しく設定されているか確認
3. TTSレスポンスに `expression: { names, frames, frame_rate }` が含まれることを確認
4. フロントエンドの `getExpressionData()` が非空データを返すことを確認

**表情が動けば、51次元のリップシンクはSDK側で正常に機能する。**
