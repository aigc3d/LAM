/**
 * DialogueManager — REST/Live API 対話の共通インターフェース
 *
 * PLATFORM_SPEC_v2.md §4 の設計に準拠。
 * モード（gourmet, concierge）に依存しない対話管理レイヤー。
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
import { LiveAudioIO } from './live-audio-io';

export type DialogueType = 'rest' | 'live';

export interface SessionStartParams {
  mode?: string;
  language?: string;
  dialogueType?: DialogueType;
  userId?: string;
  userInfo?: any;
}

export interface SessionInfo {
  sessionId: string;
  mode: string;
  language: string;
  dialogueType: DialogueType;
  greeting: string;
  initialMessage?: string;
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
   * support_base server.py: POST /api/v2/session/start
   */
  async startSession(params: SessionStartParams = {}): Promise<SessionInfo> {
    this.mode = params.mode ?? 'gourmet';
    this.language = params.language ?? 'ja';
    this.dialogueType = params.dialogueType ?? 'live';

    const res = await fetch(`${this.apiBase}/api/v2/session/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        mode: this.mode,
        language: this.language,
        dialogue_type: this.dialogueType,
        user_id: params.userId || null,
        user_info: params.userInfo || {},
      }),
    });

    if (!res.ok) {
      throw new Error(`Session start failed: ${res.status}`);
    }

    const data = await res.json();
    this.sessionId = data.session_id;

    const info: SessionInfo = {
      sessionId: data.session_id,
      mode: data.mode || this.mode,
      language: data.language || this.language,
      dialogueType: data.dialogue_type || this.dialogueType,
      greeting: data.greeting || data.initial_message || '',
      initialMessage: data.initial_message,
      wsUrl: data.ws_url,
    };

    // Live API モードの場合、WebSocket 接続を準備
    if (this.dialogueType === 'live' && info.wsUrl) {
      await this.connectLive(info.wsUrl);
    }

    return info;
  }

  /**
   * セッション終了
   */
  async endSession(): Promise<void> {
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
   * rest/router.py: POST /api/v2/rest/chat
   */
  async sendChat(
    message: string,
    stage: string = 'conversation',
    mode?: string
  ): Promise<ChatResponse> {
    if (!this.sessionId) throw new Error('No active session');

    const res = await fetch(`${this.apiBase}/api/v2/rest/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: this.sessionId,
        message,
        stage,
        language: this.language,
        mode: mode || this.mode,
      }),
    });

    if (!res.ok) throw new Error(`Chat failed: ${res.status}`);
    return await res.json();
  }

  /**
   * REST TTS 合成（Expression 同梱返却）
   * rest/router.py: POST /api/v2/rest/tts/synthesize
   */
  async synthesizeTTS(
    text: string,
    langCode: string = 'ja-JP',
    voiceName: string = 'ja-JP-Chirp3-HD-Leda',
    sessionId?: string
  ): Promise<TTSResponse> {
    const sid = sessionId || this.sessionId;

    const res = await fetch(`${this.apiBase}/api/v2/rest/tts/synthesize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text,
        language_code: langCode,
        voice_name: voiceName,
        session_id: sid,
      }),
    });

    if (!res.ok) throw new Error(`TTS failed: ${res.status}`);
    return await res.json();
  }

  /**
   * REST キャンセル
   */
  async cancel(): Promise<void> {
    if (!this.sessionId) return;
    try {
      await fetch(`${this.apiBase}/api/v2/rest/cancel`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: this.sessionId }),
      });
    } catch (e) {
      console.warn('[DialogueManager] Cancel failed:', e);
    }
  }

  // ========================================
  // Live API 対話
  // ========================================

  private async connectLive(wsUrl: string): Promise<void> {
    this.wsClient = new LiveWSClient({ wsUrl });

    this.wsClient.on('audio', (msg: LiveWSMessage) => {
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
   * ★ 必ずユーザーインタラクション（tap/click）のイベントハンドラ内から呼ぶこと
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

  stopLiveStream(): void {
    if (this.audioIO) {
      this.audioIO.stopMic();
    }
    console.log('[DialogueManager] Live stream stopped');
  }

  sendLiveText(text: string): void {
    if (this.wsClient && this.wsClient.isConnected) {
      this.wsClient.sendText(text);
    }
  }

  /** Live API 再生停止（barge-in） */
  stopLivePlayback(): void {
    if (this.audioIO) {
      this.audioIO.stopPlayback();
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

  set currentSessionId(id: string | null) {
    this.sessionId = id;
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

  get liveAudioIO(): LiveAudioIO | null {
    return this.audioIO;
  }
}
