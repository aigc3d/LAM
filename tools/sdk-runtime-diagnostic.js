/**
 * LAM SDK ランタイム診断スクリプト
 *
 * 目的: expressionBSNum が実行時に正しい値(51)になっているか確認
 *
 * 使い方:
 *   1. gourmet-sp のコンシェルジュ画面をブラウザで開く
 *   2. DevTools Console を開く
 *   3. 以下のコードを全てコピー&ペーストして Enter
 *   4. 結果を確認
 */

(function LAM_SDK_DIAGNOSTIC() {
  const PASS = '\u2705';
  const FAIL = '\u274c';
  const WARN = '\u26a0\ufe0f';
  const INFO = '\u2139\ufe0f';

  console.log('='.repeat(60));
  console.log('LAM SDK Runtime Diagnostic');
  console.log('='.repeat(60));

  // ============================================================
  // Step 1: lamAvatarController を探す
  // ============================================================
  const lam = window.lamAvatarController;
  if (!lam) {
    console.error(FAIL + ' window.lamAvatarController が見つかりません');
    console.log('  LAMAvatar.astro が初期化されていない可能性があります');
    return;
  }
  console.log(PASS + ' lamAvatarController: found');
  console.log('  keys:', Object.keys(lam).join(', '));

  // ============================================================
  // Step 2: SDK renderer インスタンスを探索
  // ============================================================
  // LAMAvatar.astro が SDK をどのプロパティに保存しているか不明なため、
  // オブジェクトツリーを幅優先探索して expressionBSNum を持つオブジェクトを見つける
  console.log('\n--- SDK Renderer 探索 ---');

  let splatMesh = null;
  let splatMeshPath = '';
  let renderer = null;
  let rendererPath = '';

  // グローバルスコープも探索対象に含める
  const searchRoots = [
    { obj: lam, name: 'lamAvatarController' },
    { obj: window, name: 'window' },
  ];

  // 幅優先探索で expressionBSNum を持つオブジェクトを検出
  function findProperty(roots, targetProp, maxDepth) {
    for (const root of roots) {
      const queue = [{ obj: root.obj, path: root.name, depth: 0 }];
      const visited = new WeakSet();

      while (queue.length > 0) {
        const { obj, path, depth } = queue.shift();
        if (!obj || typeof obj !== 'object' || depth > maxDepth) continue;
        if (visited.has(obj)) continue;
        visited.add(obj);

        try {
          if (targetProp in obj) {
            return { obj, path };
          }
        } catch (e) { continue; }

        try {
          const keys = Object.getOwnPropertyNames(obj);
          for (const key of keys) {
            // DOM要素、長い配列、関数はスキップ
            if (key === 'window' || key === 'document' || key === 'parent'
                || key === 'top' || key === 'self' || key === 'frames'
                || key === 'globalThis' || key === 'constructor') continue;
            try {
              const val = obj[key];
              if (val && typeof val === 'object' && !(val instanceof HTMLElement)
                  && !(val instanceof ArrayBuffer) && !ArrayBuffer.isView(val)
                  && !(Array.isArray(val) && val.length > 100)) {
                queue.push({ obj: val, path: path + '.' + key, depth: depth + 1 });
              }
            } catch (e) { /* getter error */ }
          }
        } catch (e) { /* enumeration error */ }
      }
    }
    return null;
  }

  // expressionBSNum を探す
  const bsNumResult = findProperty(searchRoots, 'expressionBSNum', 6);
  if (bsNumResult) {
    splatMesh = bsNumResult.obj;
    splatMeshPath = bsNumResult.path;
    console.log(PASS + ' expressionBSNum 発見: ' + splatMeshPath);
  } else {
    console.warn(WARN + ' expressionBSNum が見つかりません (探索深度6)');
    console.log('  SDK がまだ初期化されていないか、プロパティ名が異なる可能性');
  }

  // flameModel を探す (morphTargetDictionary の親)
  const flameResult = findProperty(searchRoots, 'morphTargetDictionary', 6);
  if (flameResult) {
    console.log(PASS + ' morphTargetDictionary 発見: ' + flameResult.path);
  }

  // viewer を探す
  const viewerResult = findProperty(searchRoots, 'useFlame', 6);
  if (viewerResult) {
    renderer = viewerResult.obj;
    rendererPath = viewerResult.path;
    console.log(PASS + ' renderer (useFlame) 発見: ' + rendererPath);
  }

  // ============================================================
  // Step 3: expressionBSNum の値を確認
  // ============================================================
  console.log('\n--- expressionBSNum (核心の値) ---');

  if (splatMesh) {
    const bsNum = splatMesh.expressionBSNum;
    if (bsNum === 51) {
      console.log(PASS + ' expressionBSNum = ' + bsNum + ' (正常: 51個のmorph target)');
    } else if (bsNum === 0) {
      console.error(FAIL + ' expressionBSNum = 0 (morph targetが読み込まれていない!)');
    } else if (bsNum > 0) {
      console.warn(WARN + ' expressionBSNum = ' + bsNum + ' (期待値: 51)');
    } else {
      console.error(FAIL + ' expressionBSNum = ' + bsNum + ' (異常値)');
    }

    // bsCount uniform も確認
    try {
      const bsCount = splatMesh.material?.uniforms?.bsCount?.value;
      if (bsCount !== undefined) {
        console.log(INFO + ' shader uniform bsCount = ' + bsCount);
        if (bsCount !== bsNum) {
          console.error(FAIL + ' bsCount(' + bsCount + ') !== expressionBSNum(' + bsNum + ') 不一致!');
        }
      }
    } catch (e) {
      console.log(INFO + ' bsCount uniform: アクセス不可');
    }
  } else {
    console.error(FAIL + ' splatMesh にアクセスできないため expressionBSNum 確認不可');
  }

  // ============================================================
  // Step 4: morphTargetDictionary の内容
  // ============================================================
  console.log('\n--- morphTargetDictionary ---');

  if (flameResult) {
    const dict = flameResult.obj.morphTargetDictionary;
    if (dict) {
      const names = Object.keys(dict);
      console.log(PASS + ' morph target 数: ' + names.length);
      console.log('  名前一覧:');
      names.forEach(function(name, i) {
        console.log('    [' + dict[name] + '] ' + name);
      });

      // ARKit 必須blendshape の存在確認
      const required = ['jawOpen', 'mouthFunnel', 'mouthSmileLeft', 'eyeBlinkLeft', 'browInnerUp', 'cheekPuff'];
      const missing = required.filter(function(n) { return !(n in dict); });
      if (missing.length === 0) {
        console.log(PASS + ' ARKit 主要blendshape: 全て存在');
      } else {
        console.error(FAIL + ' 欠損: ' + missing.join(', '));
      }
    } else {
      console.error(FAIL + ' morphTargetDictionary が null/undefined');
    }
  }

  // ============================================================
  // Step 5: morphAttributes.position の数 (morph target 実データ)
  // ============================================================
  console.log('\n--- morph target 実データ ---');

  if (flameResult) {
    try {
      const morphPos = flameResult.obj.geometry?.morphAttributes?.position;
      if (morphPos) {
        console.log(PASS + ' morphAttributes.position.length = ' + morphPos.length);
        // 各targetのデータ量をサンプル表示
        for (var i = 0; i < Math.min(5, morphPos.length); i++) {
          var arr = morphPos[i];
          var nonZero = 0;
          if (arr && arr.array) {
            for (var j = 0; j < arr.array.length; j++) {
              if (arr.array[j] !== 0) nonZero++;
            }
          }
          console.log('    [' + i + '] count=' + (arr?.count || '?') + ', nonZero=' + nonZero);
        }
      } else {
        console.warn(WARN + ' morphAttributes.position が存在しない');
      }
    } catch (e) {
      console.warn(WARN + ' morphAttributes アクセスエラー: ' + e.message);
    }
  }

  // ============================================================
  // Step 6: 現在の bsWeight (expression data) を確認
  // ============================================================
  console.log('\n--- bsWeight (現在のExpression値) ---');

  if (splatMesh && splatMesh.bsWeight) {
    const bsWeight = splatMesh.bsWeight;
    const keys = Object.keys(bsWeight);
    console.log(INFO + ' bsWeight のキー数: ' + keys.length);
    const nonZero = keys.filter(function(k) { return bsWeight[k] !== 0 && bsWeight[k] !== undefined; });
    console.log(INFO + ' 非ゼロ値: ' + nonZero.length + '/' + keys.length);
    nonZero.forEach(function(k) {
      console.log('    ' + k + ' = ' + bsWeight[k].toFixed(4));
    });
    if (nonZero.length === 0) {
      console.log(INFO + ' 全てゼロ (Idle状態 or データ未到達)');
      console.log('  → 話しかけてTTS再生中にもう一度実行してください');
    }
  } else if (splatMesh) {
    console.warn(WARN + ' bsWeight プロパティが存在しない');
  }

  // ============================================================
  // Step 7: useFlame モード確認
  // ============================================================
  console.log('\n--- SDK モード ---');

  if (renderer) {
    console.log(INFO + ' useFlame = ' + renderer.useFlame);
    if (renderer.useFlame === false) {
      console.log(PASS + ' OAC (ARKit blendshape) モード — 正常');
    } else {
      console.warn(WARN + ' FLAME モード — OAC ZIP との不整合の可能性');
    }
  }

  // ============================================================
  // Step 8: GPU テクスチャ確認 (boneTexture)
  // ============================================================
  console.log('\n--- GPU テクスチャ (boneTexture) ---');

  if (splatMesh) {
    try {
      const boneTex = splatMesh.skeleton?.boneTexture
                    || splatMesh.boneTexture
                    || splatMesh.material?.uniforms?.boneTexture?.value;
      if (boneTex) {
        console.log(PASS + ' boneTexture: ' + boneTex.image.width + 'x' + boneTex.image.height);
        // blendshape weight 領域のデータを確認
        var texData = boneTex.image.data;
        if (texData) {
          var bonesNum = splatMesh.bonesNum || 0;
          console.log(INFO + ' bonesNum = ' + bonesNum);
          // BS weight はテクスチャの bonesNum*16 以降に格納
          var bsStart = bonesNum * 16;
          var bsSlice = [];
          for (var k = bsStart; k < Math.min(bsStart + 52, texData.length); k++) {
            bsSlice.push(texData[k]);
          }
          var nonZeroTex = bsSlice.filter(function(v) { return v !== 0; }).length;
          console.log(INFO + ' BS weight テクスチャ領域 (先頭52): nonZero=' + nonZeroTex);
          if (nonZeroTex > 0) {
            console.log(PASS + ' GPUテクスチャにblendshape weightが書き込まれている');
          } else {
            console.log(INFO + ' 全ゼロ (Idle状態なら正常。TTS再生中に再確認を)');
          }
        }
      } else {
        console.warn(WARN + ' boneTexture が見つかりません');
      }
    } catch (e) {
      console.warn(WARN + ' boneTexture アクセスエラー: ' + e.message);
    }
  }

  // ============================================================
  // Summary
  // ============================================================
  console.log('\n' + '='.repeat(60));
  console.log('SUMMARY');
  console.log('='.repeat(60));

  if (splatMesh) {
    var bsNum = splatMesh.expressionBSNum;
    if (bsNum === 51) {
      console.log(PASS + ' expressionBSNum = 51 — 52次元は正しく使用されている');
      console.log('  (tongueOut の1個が欠損しているのは skin.glb 由来で正常)');
    } else if (bsNum > 0 && bsNum < 51) {
      console.warn(WARN + ' expressionBSNum = ' + bsNum + ' — 一部blendshapeが欠損');
    } else if (bsNum === 0 || bsNum === undefined) {
      console.error(FAIL + ' expressionBSNum = ' + bsNum + ' — blendshapeが全く使われていない!');
      console.log('  → skin.glb の morph target ロードに失敗している可能性');
    }
  } else {
    console.error(FAIL + ' SDK の splatMesh にアクセスできませんでした');
    console.log('  → 結果を手動で確認する方法:');
    console.log('  1. DevTools Sources タブを開く');
    console.log('  2. gaussian-splat-renderer-for-lam.js を検索');
    console.log('  3. "expressionBSNum" で検索してブレークポイントを設定');
    console.log('  4. ページリロードして値を確認');
  }

  console.log('='.repeat(60));
})();
