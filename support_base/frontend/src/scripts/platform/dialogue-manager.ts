/**
 * DialogueManager — REST/Live API 対話の共通インターフェース
 *
 * PLATFORM_ARCHITECTURE.md §8.3 の設計に準拠。
 * モード（gourmet, support, interview）に依存しない対話管理レイヤー。
 *
 * REST 経路:
 *   POST /api/v2/rest/session/start → session_id
 *   POST /api/v2/rest/chat → { response, audio, expression, shops }
 *   POST /api/v2/rest/tts/synthesize → { audio, expression }
 *
 * Live API 経路:
 *   POST /api/v2/session/start → { session_id, ws_url }
 *   WebSocket /api/v2/live/{session_id} → 音声ストリーミング
 */

import { LiveWSClient, type LiveWSMessage } from './live-ws-client';
import { LiveAudioIO } from './audio-io';

export type DialogueType = 'rest' | 'live';

export interface SessionInfo {
  sessionId: string;
  mode: string;
  language: string;
  dialogueType: DialogueType;
  greeting: string;
  wsUrl?: string;
}

export interface ChatResponse {
  response: string;
  summary?: string;
  shops?: any[];
  shouldConfirm?: boolean;
  isFollowup?: boolean;
}

export interface TTSResponse {
  success: boolean;
  audio?: string;
  expression?: {
    names: string[];
    frames: any[];
    frame_rate: number;
  };
}

export interface ExpressionData {
  names: string[];
  frames: any[];
  frame_rate: number;
}

type EventHandler<T = any> = (data: T) => void;

export class DialogueManager {
  private apiBase: string;
  private sessionId: string | null = null;
  private mode: string = 'gourmet';
  private language: string = 'ja';
  private dialogueType: DialogueType = 'live';

  // Live API
  private wsClient: LiveWSClient | null = null;
  private audioIO: LiveAudioIO | null = null;

  // イベントハンドラ
  private eventHandlers: Map<string, EventHandler[]> = new Map();

  constructor(apiBase: string) {
    this.apiBase = apiBase;
  }

  // ========================================
  // セッション管理
  // ========================================

  /**
   * セッション開始
   * server.py L134-172: POST /api/v2/session/start
   */
  async startSession(
    mode: string = 'gourmet',
    language: string = 'ja',
    dialogueType: DialogueType = 'live',
    userId?: string
  ): Promise<SessionInfo> {
    this.mode = mode;
    this.language = language;
    this.dialogueType = dialogueType;

    const res = await fetch(`${this.apiBase}/api/v2/session/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        mode,
        language,
        dialogue_type: dialogueType,
        user_id: userId || null,
      }),
    });

    if (!res.ok) {
      throw new Error(`Session start failed: ${res.status}`);
    }

    const data = await res.json();
    this.sessionId = data.session_id;

    const info: SessionInfo = {
      sessionId: data.session_id,
      mode: data.mode,
      language: data.language,
      dialogueType: data.dialogue_type,
      greeting: data.greeting,
      wsUrl: data.ws_url,
    };

    // Live API モードの場合、WebSocket 接続を準備
    if (dialogueType === 'live' && info.wsUrl) {
      await this.connectLive(info.wsUrl);
    }

    return info;
  }

  /**
   * セッション終了
   */
  async endSession(): Promise<void> {
    // Live API 接続を切断
    this.disconnectLive();

    if (this.sessionId) {
      try {
        await fetch(`${this.apiBase}/api/v2/session/end`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: this.sessionId }),
        });
      } catch (e) {
        console.warn('[DialogueManager] Session end failed:', e);
      }
      this.sessionId = null;
    }
  }

  // ========================================
  // REST 対話（既存 gourmet-support 互換）
  // ========================================

  /**
   * REST チャット送信
   * rest/router.py L196-319: POST /api/v2/rest/chat
   */
  async sendChat(message: string, stage: string = 'conversation'): Promise<ChatResponse> {
    if (!this.sessionId) throw new Error('No active session');

    const res = await fetch(`${this.apiBase}/api/v2/rest/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: this.sessionId,
        message,
        stage,
        language: this.language,
        mode: this.mode === 'gourmet' ? 'concierge' : this.mode,
      }),
    });

    if (!res.ok) throw new Error(`Chat failed: ${res.status}`);
    return await res.json();
  }

  /**
   * REST TTS 合成（Expression 同梱返却）
   * rest/router.py L371-433: POST /api/v2/rest/tts/synthesize
   */
  async synthesizeTTS(
    text: string,
    langCode: string = 'ja-JP',
    voiceName: string = 'ja-JP-Chirp3-HD-Leda'
  ): Promise<TTSResponse> {
    if (!this.sessionId) throw new Error('No active session');

    const res = await fetch(`${this.apiBase}/api/v2/rest/tts/synthesize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text,
        language_code: langCode,
        voice_name: voiceName,
        session_id: this.sessionId,
      }),
    });

    if (!res.ok) throw new Error(`TTS failed: ${res.status}`);
    return await res.json();
  }

  // ========================================
  // Live API 対話
  // ========================================

  /**
   * Live API WebSocket に接続
   */
  private async connectLive(wsUrl: string): Promise<void> {
    this.wsClient = new LiveWSClient({ wsUrl });

    // サーバーからのイベントをハンドリング
    this.wsClient.on('audio', (msg: LiveWSMessage) => {
      // PCM 24kHz 音声を再生キューに追加
      if (msg.data && this.audioIO) {
        this.audioIO.queuePlayback(msg.data);
      }
      this.emit('ai_audio', msg.data);
    });

    this.wsClient.on('transcription', (msg: LiveWSMessage) => {
      if (msg.role === 'user') {
        this.emit('user_text', { text: msg.text, isPartial: msg.is_partial });
      } else if (msg.role === 'ai') {
        this.emit('ai_text', { text: msg.text, isPartial: msg.is_partial });
      }
    });

    this.wsClient.on('expression', (msg: LiveWSMessage) => {
      this.emit('expression', msg.data);
    });

    this.wsClient.on('interrupted', () => {
      // barge-in: 再生中の音声を停止
      if (this.audioIO) {
        this.audioIO.stopPlayback();
      }
      this.emit('interrupted', null);
    });

    this.wsClient.on('reconnecting', (msg: LiveWSMessage) => {
      this.emit('reconnecting', msg.reason);
    });

    this.wsClient.on('reconnected', (msg: LiveWSMessage) => {
      this.emit('reconnected', msg.session_count);
    });

    this.wsClient.on('error', (msg: LiveWSMessage) => {
      console.error('[DialogueManager] Live error:', msg.message);
      this.emit('error', msg.message);
    });

    this.wsClient.on('connection', (connected: boolean) => {
      this.emit('connection', connected);
    });

    await this.wsClient.connect();
  }

  /**
   * Live API 接続を切断
   */
  private disconnectLive(): void {
    if (this.audioIO) {
      this.audioIO.destroy();
      this.audioIO = null;
    }
    if (this.wsClient) {
      this.wsClient.disconnect();
      this.wsClient = null;
    }
  }

  /**
   * Live API: マイク音声ストリーミング開始
   *
   * ★ 必ずユーザーインタラクション（tap/click）のイベントハンドラ内から呼ぶこと
   *   iPhone の autoplay policy 回避のため
   */
  async startLiveStream(): Promise<void> {
    if (!this.wsClient || !this.wsClient.isConnected) {
      throw new Error('Live API not connected');
    }

    this.audioIO = new LiveAudioIO({
      wsClient: this.wsClient,
      sendSampleRate: 16000,
      receiveSampleRate: 24000,
    });

    await this.audioIO.startMic();
    console.log('[DialogueManager] Live stream started');
  }

  /**
   * Live API: マイク音声ストリーミング停止
   */
  stopLiveStream(): void {
    if (this.audioIO) {
      this.audioIO.stopMic();
    }
    console.log('[DialogueManager] Live stream stopped');
  }

  /**
   * Live API: テキスト送信
   */
  sendLiveText(text: string): void {
    if (this.wsClient && this.wsClient.isConnected) {
      this.wsClient.sendText(text);
    }
  }

  // ========================================
  // イベント管理
  // ========================================

  on(event: string, handler: EventHandler): void {
    const handlers = this.eventHandlers.get(event) || [];
    handlers.push(handler);
    this.eventHandlers.set(event, handlers);
  }

  off(event: string, handler: EventHandler): void {
    const handlers = this.eventHandlers.get(event) || [];
    this.eventHandlers.set(event, handlers.filter((h) => h !== handler));
  }

  private emit(event: string, data: any): void {
    const handlers = this.eventHandlers.get(event) || [];
    handlers.forEach((h) => h(data));
  }

  // ========================================
  // アクセサ
  // ========================================

  get currentSessionId(): string | null {
    return this.sessionId;
  }

  get currentMode(): string {
    return this.mode;
  }

  get currentLanguage(): string {
    return this.language;
  }

  get currentDialogueType(): DialogueType {
    return this.dialogueType;
  }

  get isLiveConnected(): boolean {
    return this.wsClient?.isConnected ?? false;
  }

  get isMicActive(): boolean {
    return this.audioIO?.micActive ?? false;
  }
}
