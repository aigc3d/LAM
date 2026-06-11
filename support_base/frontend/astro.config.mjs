import { defineConfig } from 'astro/config';

export default defineConfig({
  // バックエンド API へのプロキシ（開発時）
  vite: {
    server: {
      proxy: {
        '/api': {
          target: 'http://localhost:8080',
          changeOrigin: true,
          ws: true,  // WebSocket プロキシ（Live API 用）
        },
      },
    },
  },
});
