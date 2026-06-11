# gourmet-sp2 引継ぎドキュメント

> 前セッション (session_01E9rf3QsqK1jCcMpd5RR9f1) で実装済み・コミット済みだが、
> Git プロキシの認可制限により **push できていない**。
> このドキュメントは gourmet-sp2 リポジトリで新セッションを起動した際の引継ぎ用。

---

## 1. やるべきこと (最優先)

### ブランチをプッシュする

```bash
# gourmet-sp2 リポジトリ内で実行
git checkout claude/platform-design-docs-oEVkm
git push -u origin claude/platform-design-docs-oEVkm
```

ブランチ `claude/platform-design-docs-oEVkm` にコミット済み (1コミット先行)。
**main にマージ前にプッシュするだけでOK。**

### 未ステージの変更もコミット＆プッシュする

`package.json` と `package-lock.json` に未ステージの変更あり:
- `@astrojs/check` (devDependency) 追加
- `typescript` (devDependency) 追加

```bash
git add package.json package-lock.json
git commit -m "chore: add @astrojs/check and typescript devDependencies"
git push -u origin claude/platform-design-docs-oEVkm
```

---

## 2. 実装済みの内容

### コミット: `17a32a8` — feat: Live API platform integration + /api/v2/ migration

**変更ファイル一覧 (7ファイル, +1142行, -183行):**

| ファイル | 種別 | 説明 |
|---------|------|------|
| `src/scripts/platform/live-ws-client.ts` | **新規** | WebSocket client for support_base LiveRelay (relay.py) |
| `src/scripts/platform/live-audio-io.ts` | **新規** | PCM 16kHz mic capture + PCM 24kHz playback via AudioContext |
| `src/scripts/platform/dialogue-manager.ts` | **新規** | REST/Live API switching layer with unified session management |
| `src/scripts/chat/core-controller.ts` | **変更** | DialogueManager統合, Live APIイベントハンドラ, mic streaming, /api/v2/ 移行 |
| `src/scripts/chat/concierge-controller.ts` | **変更** | Live API expression→LAMAvatar, TTS via DialogueManager /api/v2/, session管理 |
| `vercel.json` | **新規** | COOP/COEP headers, API proxy rewrites, Astro config |
| `.env.example` | **新規** | PUBLIC_API_URL ドキュメント |

### アーキテクチャ

```
[ブラウザ (gourmet-sp2 / Astro)]
  ├── ConciergeController  ← UIイベント・アバター描画制御
  │     └── CoreController ← 対話ロジック・Live API制御
  │           └── DialogueManager ← REST / Live API 切替レイヤー
  │                 ├── (REST mode)  → /api/v2/* via fetch
  │                 └── (Live mode)  → LiveWSClient (WebSocket)
  │                                    LiveAudioIO  (mic/speaker)
  │
  ├── vercel.json rewrites → support_base Cloud Run
  └── .env.example → PUBLIC_API_URL
```

### エンドポイント移行

全 API コールを `/api/*` → `/api/v2/*` に移行済み:
- `/api/v2/session/start`
- `/api/v2/session/end`
- `/api/v2/tts`
- `/api/v2/chat`
- `/socket.io/*` (WebSocket relay)

---

## 3. vercel.json

```json
{
  "framework": "astro",
  "buildCommand": "npm run build",
  "outputDirectory": "dist",
  "headers": [
    {
      "source": "/(.*)",
      "headers": [
        { "key": "Cross-Origin-Opener-Policy", "value": "same-origin" },
        { "key": "Cross-Origin-Embedder-Policy", "value": "require-corp" }
      ]
    }
  ],
  "rewrites": [
    { "source": "/api/v2/:path*", "destination": "${PUBLIC_API_URL}/api/v2/:path*" },
    { "source": "/socket.io/:path*", "destination": "${PUBLIC_API_URL}/socket.io/:path*" }
  ]
}
```

COOP/COEP は SharedArrayBuffer (AudioWorklet) に必要。

---

## 4. .env.example

```
PUBLIC_API_URL=https://support-base-xxxxx-an.a.run.app
```

Vercel デプロイ時に Project Settings > Environment Variables で設定する。

---

## 5. リモートURLの修正が必要

前セッションで remote URL が誤ったプロキシに変更されている:

```
origin → http://local_proxy@127.0.0.1:18638/git/mirai-gpro/gourmet-sp2 (誤り)
```

新セッションでは自動的に正しいプロキシが設定されるはずだが、
もし GitHub URL に戻す必要がある場合:

```bash
git remote set-url origin https://github.com/mirai-gpro/gourmet-sp2.git
```

---

## 6. ミラーコピー (LAM_gpro 側)

全実装ファイルのコピーが LAM_gpro にも保存されている:

```
LAM_gpro/support_base/frontend/gourmet-sp2-impl/
  ├── live-ws-client.ts
  ├── live-audio-io.ts
  ├── dialogue-manager.ts
  ├── core-controller.ts
  ├── concierge-controller.ts
  ├── vercel.json
  └── .env.example
```

万が一 gourmet-sp2 のブランチが壊れた場合のバックアップ。

---

## 7. 次のステップ (プッシュ後)

1. **PR 作成**: `claude/platform-design-docs-oEVkm` → `main`
2. **Vercel 環境変数設定**: `PUBLIC_API_URL` を Cloud Run URL に設定
3. **結合テスト**: TTS → Audio2Exp → アバター描画のパイプライン全体テスト
4. **support_base 側**: LiveRelay (relay.py) と /api/v2/ エンドポイントの動作確認
