# フロントエンド A2E 統合ガイド

## 概要

gourmet-support の `concierge-controller.ts` を修正して、
バックエンドから返却される A2E expression データを使った
高精度リップシンクを実現する。

## 変更対象ファイル

### 1. 新規ファイル追加
```
src/scripts/avatar/vrm-expression-manager.ts  ← このディレクトリにコピー
```

### 2. concierge-controller.ts の変更

#### 2a. インポート追加 (ファイル先頭)
```typescript
import { ExpressionManager, ExpressionData } from '../avatar/vrm-expression-manager';
```

#### 2b. プロパティ追加 (class ConciergeController内)
```typescript
private expressionManager: ExpressionManager | null = null;
```

#### 2c. init() メソッド内、GVRM初期化後に追加
```typescript
// ★追加: ExpressionManager初期化
if (this.guavaRenderer) {
    this.expressionManager = new ExpressionManager(this.guavaRenderer);
}
```

#### 2d. TTS API呼び出し時に session_id を追加

**すべての `/api/tts/synthesize` リクエストに `session_id` を追加する。**

変更前:
```typescript
body: JSON.stringify({
    text: cleanText,
    language_code: langConfig.tts,
    voice_name: langConfig.voice
})
```

変更後:
```typescript
body: JSON.stringify({
    text: cleanText,
    language_code: langConfig.tts,
    voice_name: langConfig.voice,
    session_id: this.sessionId  // ★追加
})
```

#### 2e. TTS再生時にexpressionデータを使う

音声再生ロジックを拡張して、expressionデータがある場合はExpressionManagerで再生する。

```typescript
// TTS APIレスポンス取得後
const result = await response.json();
if (result.success && result.audio) {
    const audioSrc = `data:audio/mp3;base64,${result.audio}`;

    // ★ A2E expression データがある場合、ExpressionManagerで再生
    if (result.expression && ExpressionManager.isValid(result.expression) && this.expressionManager) {
        // FFTベースのリップシンクではなく、A2Eベースを使用
        this.ttsPlayer.src = audioSrc;

        // ExpressionManagerで同期再生
        this.expressionManager.playExpressionFrames(result.expression, this.ttsPlayer);

        await new Promise<void>((resolve) => {
            this.ttsPlayer.onended = () => {
                this.expressionManager?.stop();
                resolve();
            };
            this.ttsPlayer.play();
        });
    } else {
        // フォールバック: 従来のFFTベースリップシンク
        this.ttsPlayer.src = audioSrc;
        this.setupAudioAnalysis();
        this.startLipSyncLoop();
        await new Promise<void>((resolve) => {
            this.ttsPlayer.onended = () => resolve();
            this.ttsPlayer.play();
        });
    }
}
```

#### 2f. stopAvatarAnimation() の修正

```typescript
private stopAvatarAnimation() {
    if (this.els.avatarContainer) {
        this.els.avatarContainer.classList.remove('speaking');
    }
    // ★ ExpressionManager停止
    this.expressionManager?.stop();
    // フォールバック用
    this.guavaRenderer?.updateLipSync(0);
    if (this.animationFrameId) {
        cancelAnimationFrame(this.animationFrameId);
        this.animationFrameId = null;
    }
}
```

## 動作フロー

```
1. ユーザーが音声/テキスト入力
2. バックエンドに /api/chat 送信
3. レスポンステキストを /api/tts/synthesize に送信（session_id付き）
4. バックエンド:
   a. Google Cloud TTS で MP3 生成
   b. MP3 を audio2exp-service に送信
   c. 52次元 ARKit blendshape フレーム取得
   d. JSON: { audio, expression: {names, frames, frame_rate} } 返却
5. フロントエンド:
   a. expression データがあれば ExpressionManager で再生
   b. なければ従来の FFT ベースリップシンク（フォールバック）
   c. ExpressionManager: 音声の currentTime に同期してフレーム選択
   d. フレームの jawOpen 等 → GVRM.updateLipSync() にマッピング
```

## テスト方法

### ローカルテスト
1. audio2exp-service を起動: `python app.py` (port 8081)
2. gourmet-support の環境変数: `AUDIO2EXP_SERVICE_URL=http://localhost:8081`
3. gourmet-support を起動: `python app_customer_support.py`
4. フロントエンドでコンシェルジュモードを開く
5. 日本語で話しかけ、リップシンクの品質を確認

### 品質確認ポイント
- [ ] 口の開閉タイミングが発話と合っているか
- [ ] 無音時に口が閉じるか
- [ ] 「あ」(jawOpen大) と「い」(mouthSmile) の区別があるか
- [ ] FFTベースよりも自然に見えるか
