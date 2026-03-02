/**
 * LiveRelay WebSocket クライアント
 *
 * support_base/live/relay.py の LiveRelay に接続する。
 * ブラウザ ↔ サーバー 間のプロトコルを実装。
 *
 * プロトコル (relay.py):
 *   クライアント → サーバー:
 *     { "type": "audio", "data": "<base64 PCM 16kHz>" }
 *     { "type": "text",  "data": "テキスト入力" }
 *     { "type": "stop" }
 *
 *   サーバー → クライアント:
 *     { "type": "audio",         "data": "<base64 PCM 24kHz>" }
 *     { "type": "transcription", "role": "user"|"ai", "text": "...", "is_partial": bool }
 *     { "type": "expression",    "data": { names, frames, frame_rate } }
 *     { "type": "interrupted" }
 *     { "type": "reconnecting",  "reason": "..." }
 *     { "type": "reconnected",   "session_count": N }
 *     { "type": "error",         "message": "..." }
 */

export interface LiveWSClientOptions {
  wsUrl: string;
  connectTimeout?: number;
}

export type LiveWSMessageType =
  | 'audio'
  | 'transcription'
  | 'expression'
  | 'interrupted'
  | 'reconnecting'
  | 'reconnected'
  | 'error';

export interface LiveWSMessage {
  type: LiveWSMessageType;
  data?: any;
  role?: 'user' | 'ai';
  text?: string;
  is_partial?: boolean;
  reason?: string;
  session_count?: number;
  message?: string;
}

type LiveWSEventHandler = (msg: LiveWSMessage) => void;
type LiveWSConnectionHandler = (connected: boolean) => void;

export class LiveWSClient {
  private ws: WebSocket | null = null;
  private wsUrl: string;
  private connectTimeout: number;
  private handlers: Map<string, LiveWSEventHandler[]> = new Map();
  private connectionHandlers: LiveWSConnectionHandler[] = [];
  private _isConnected = false;

  constructor(options: LiveWSClientOptions) {
    this.wsUrl = options.wsUrl;
    this.connectTimeout = options.connectTimeout ?? 10000;
  }

  async connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        reject(new Error('WebSocket connection timeout'));
      }, this.connectTimeout);

      try {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host;
        const url = this.wsUrl.startsWith('ws')
          ? this.wsUrl
          : `${protocol}//${host}${this.wsUrl}`;

        this.ws = new WebSocket(url);

        this.ws.onopen = () => {
          clearTimeout(timer);
          this._isConnected = true;
          this.notifyConnection(true);
          console.log('[LiveWSClient] Connected:', url);
          resolve();
        };

        this.ws.onclose = (event) => {
          clearTimeout(timer);
          this._isConnected = false;
          this.notifyConnection(false);
          console.log(`[LiveWSClient] Closed: code=${event.code}, reason=${event.reason}`);
        };

        this.ws.onerror = (event) => {
          clearTimeout(timer);
          console.error('[LiveWSClient] Error:', event);
          reject(new Error('WebSocket connection error'));
        };

        this.ws.onmessage = (event) => {
          this.handleMessage(event.data);
        };
      } catch (e) {
        clearTimeout(timer);
        reject(e);
      }
    });
  }

  disconnect(): void {
    if (this.ws) {
      this.sendJson({ type: 'stop' });
      this.ws.close();
      this.ws = null;
    }
    this._isConnected = false;
  }

  sendAudio(base64Pcm: string): void {
    this.sendJson({ type: 'audio', data: base64Pcm });
  }

  sendText(text: string): void {
    this.sendJson({ type: 'text', data: text });
  }

  on(event: LiveWSMessageType | 'connection', handler: any): void {
    if (event === 'connection') {
      this.connectionHandlers.push(handler as LiveWSConnectionHandler);
      return;
    }
    const existing = this.handlers.get(event) || [];
    existing.push(handler as LiveWSEventHandler);
    this.handlers.set(event, existing);
  }

  off(event: LiveWSMessageType | 'connection', handler: any): void {
    if (event === 'connection') {
      this.connectionHandlers = this.connectionHandlers.filter(h => h !== handler);
      return;
    }
    const existing = this.handlers.get(event) || [];
    this.handlers.set(event, existing.filter(h => h !== handler));
  }

  get isConnected(): boolean {
    return this._isConnected;
  }

  private sendJson(obj: any): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(obj));
    }
  }

  private handleMessage(raw: string): void {
    try {
      const msg: LiveWSMessage = JSON.parse(raw);
      const handlers = this.handlers.get(msg.type);
      if (handlers) {
        handlers.forEach(h => h(msg));
      }
    } catch (e) {
      console.warn('[LiveWSClient] Failed to parse message:', raw);
    }
  }

  private notifyConnection(connected: boolean): void {
    this.connectionHandlers.forEach(h => h(connected));
  }
}
