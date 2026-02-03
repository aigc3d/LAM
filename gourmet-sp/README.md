# Gourmet Support AI - LAM 3D Avatar Integration

このディレクトリは、グルメサポートAIのコンシェルジュモードに LAM (Large Avatar Model) 3Dアバターを統合するためのテスト環境です。

## セットアップ手順

### 1. ローカル環境にコピー

このディレクトリの `src/` と `public/` を、ローカルの gourmet-sp プロジェクトにコピーしてください。

```bash
# ローカルのgourmet-spディレクトリで実行
cp -r /path/to/LAM_gpro/gourmet-sp/src ./
cp -r /path/to/LAM_gpro/gourmet-sp/public ./
```

### 2. NPMパッケージのインストール

LAM WebGL レンダラーをインストール：

```bash
npm install gaussian-splat-renderer-for-lam
```

### 3. アバターファイルの配置

LAMで生成した3Dアバター（.zipファイル）を配置：

```bash
mkdir -p public/avatar
cp /path/to/your-avatar.zip public/avatar/concierge.zip
```

### 4. 開発サーバーの起動

```bash
npm run dev
# http://localhost:4321/concierge でアクセス
```

## コンポーネント構成

```
src/
├── components/
│   ├── Concierge.astro      # メインコンシェルジュUI（LAM統合済み）
│   └── LAMAvatar.astro      # LAM 3Dアバターコンポーネント
└── pages/
    └── concierge.astro      # コンシェルジュページ
```

## LAMAvatar コンポーネントの使い方

```astro
---
import LAMAvatar from '../components/LAMAvatar.astro';
---

<LAMAvatar
  avatarPath="/avatar/concierge.zip"
  width="100%"
  height="300px"
/>
```

### Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `avatarPath` | string | `/avatar/concierge.zip` | アバター.zipファイルのパス |
| `width` | string | `100%` | コンテナの幅 |
| `height` | string | `100%` | コンテナの高さ |
| `wsUrl` | string | `''` | OpenAvatarChat WebSocket URL |
| `autoConnect` | boolean | `false` | 自動WebSocket接続 |

### JavaScript API

```javascript
// グローバルにアクセス可能
const controller = window.lamAvatarController;

// 状態を設定（Idle, Listening, Thinking, Responding）
controller.setChatState('Responding');

// 表情データを設定（Audio2Expressionの出力）
controller.setExpressionData({
  'jawOpen': 0.5,
  'mouthSmile_L': 0.3,
  'mouthSmile_R': 0.3,
  // ... 他のARKitブレンドシェイプ
});

// Audio2Expressionフレームから更新
controller.updateFromAudio2Expression({
  names: ['jawOpen', 'mouthSmile_L', ...],
  weights: [0.5, 0.3, ...]
});
```

## Concierge コンポーネントの設定

```astro
---
import ConciergeComponent from '../components/Concierge.astro';
---

<!-- LAM 3Dアバターを使用（デフォルト） -->
<ConciergeComponent
  apiBaseUrl={apiBaseUrl}
  useLAMAvatar={true}
  avatarPath="/avatar/concierge.zip"
/>

<!-- 2Dアバターにフォールバック -->
<ConciergeComponent
  apiBaseUrl={apiBaseUrl}
  useLAMAvatar={false}
/>
```

## 3Dアバターの生成方法

1. **コンシェルジュ画像を用意**
   - 正面向きの顔写真
   - 高解像度推奨（512x512以上）

2. **LAMで3Dアバターを生成**（GPU環境が必要）
   ```bash
   cd /path/to/LAM_gpro
   python app_lam.py
   # Gradio UIで画像をアップロード
   # ZIPファイルをエクスポート
   ```

3. **生成されたZIPを配置**
   ```bash
   cp generated_avatar.zip public/avatar/concierge.zip
   ```

## OpenAvatarChat WebSocket 連携（リップシンク）

OpenAvatarChatバックエンドとWebSocketで接続して、リアルタイムリップシンクを実現します。

### 接続方法

```astro
<!-- 自動接続（推奨） -->
<LAMAvatar
  avatarPath="/avatar/concierge.zip"
  wsUrl="wss://your-openavatarchat-server:8282/ws"
  autoConnect={true}
/>
```

```javascript
// または、JavaScriptから手動接続
const controller = window.lamAvatarController;

// WebSocket接続
await controller.connectWebSocket('wss://your-server:8282/ws');

// 接続状態の確認
console.log('Connected:', controller.isWebSocketConnected());

// 切断
controller.disconnectWebSocket();
```

### イベントリスナー

```javascript
// 接続状態の変更を監視
document.getElementById('lamAvatarContainer').addEventListener('lamConnectionChange', (e) => {
  console.log('WebSocket connected:', e.detail.connected);
});

// チャット状態の変更を監視
document.getElementById('lamAvatarContainer').addEventListener('lamStateChange', (e) => {
  console.log('Chat state:', e.detail.state);
});
```

### データフロー

1. **OpenAvatarChat バックエンド** がAudio2Expressionで音声を解析
2. **JBIN形式** でARKit表情データ（52チャンネル）をWebSocket送信
3. **LAMWebSocketManager** がバイナリをパースして表情データに変換
4. **GaussianSplatRenderer** がリアルタイムでアバターを更新

### ファイル構成

```
src/scripts/lam/
└── lam-websocket-manager.ts  # JBIN パーサー & WebSocket管理
```

## Audio2Expression との連携（手動モード）

WebSocketを使わずに、手動で表情データを設定する場合：

```javascript
// バックエンドからの表情データを受信
socket.on('expression_frame', (frame) => {
  window.lamAvatarController.updateFromAudio2Expression(frame);
});
```

## トラブルシューティング

### NPMパッケージがインストールできない

```bash
# Node.js 18以上が必要
node --version

# キャッシュクリア
npm cache clean --force
npm install gaussian-splat-renderer-for-lam
```

### 3Dアバターが表示されない

1. ブラウザがWebGL 2.0をサポートしているか確認
2. アバター.zipファイルのパスが正しいか確認
3. コンソールエラーを確認

### フォールバック画像が表示される

NPMパッケージがインストールされていないか、WebGLが利用できない場合、自動的に2D画像にフォールバックします。

## 関連リポジトリ

- [LAM (Large Avatar Model)](https://github.com/aigc3d/LAM) - 3Dアバター生成
- [LAM_WebRender](https://github.com/aigc3d/LAM_WebRender) - WebGLレンダラー
- [LAM_Audio2Expression](https://github.com/aigc3d/LAM_Audio2Expression) - 音声→表情変換
- [OpenAvatarChat](https://github.com/HumanAIGC-Engineering/OpenAvatarChat) - 統合SDK
