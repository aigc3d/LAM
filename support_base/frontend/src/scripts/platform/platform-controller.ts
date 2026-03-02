/**
 * PlatformController — プラットフォーム共通コントローラー
 *
 * concierge-controller.ts (CoreController → ConciergeController) の設計パターンを踏襲し、
 * REST / Live API 両対応のプラットフォーム共通基盤を提供する。
 *
 * 設計原則 (PLATFORM_ARCHITECTURE.md §8):
 * - モード非依存の共通層 → modes/gourmet-mode.ts 等で拡張
 * - REST 経路と Live API 経路を DialogueManager で切替
 * - LAMAvatar との統合は ConciergeController のパターンをそのまま踏襲
 * - オーディオ制御コード (AudioManager, ttsPlayer) は一切改変しない
 *
 * ★★★ 注意 ★★★
 * ttsPlayer, AudioManager 関連のコードは既存パッチから変更せずコピーすること。
 * iPhone 16/17 の iOS セキュリティ制限対策が含まれており、
 * 書き換えると動かなくなる（実証済み）。
 */

import {
  DialogueManager,
  type DialogueType,
  type SessionInfo,
  type ExpressionData,
} from './dialogue-manager';

// ========================================
// 多言語設定 (PLATFORM_ARCHITECTURE.md §7.4)
// concierge-controller.ts L199, L285 の LANGUAGE_CODE_MAP を再現
// ========================================

interface LanguageConfig {
  code: string;
  tts: string;
  voice: string;
  sentenceSplitter: 'cjk' | 'latin';
}

const LANGUAGE_CONFIG_MAP: Record<string, LanguageConfig> = {
  ja: { code: 'ja', tts: 'ja-JP', voice: 'ja-JP-Chirp3-HD-Leda', sentenceSplitter: 'cjk' },
  en: { code: 'en', tts: 'en-US', voice: 'en-US-Wavenet-D', sentenceSplitter: 'latin' },
  ko: { code: 'ko', tts: 'ko-KR', voice: 'ko-KR-Wavenet-D', sentenceSplitter: 'latin' },
  zh: { code: 'zh', tts: 'cmn-CN', voice: 'cmn-CN-Wavenet-D', sentenceSplitter: 'cjk' },
};

// ========================================
// DOM 要素インターフェース
// ========================================

interface PlatformElements {
  chatArea: HTMLElement | null;
  userInput: HTMLInputElement | null;
  sendBtn: HTMLButtonElement | null;
  micBtn: HTMLButtonElement | null;
  voiceStatus: HTMLElement | null;
  avatarContainer: HTMLElement | null;
  modeIndicator: HTMLElement | null;
}

// ========================================
// FLAME LBS 安全上限 (concierge-controller.ts L381)
// ========================================
const BLENDSHAPE_SAFE_MAX = 0.7;

// 口周り blendshape スケール係数 (concierge-controller.ts L359-378)
// 全て 1.0（増幅なし）— チューニング時にここを調整
const MOUTH_AMPLIFY: Record<string, number> = {
  jawOpen: 1.0, mouthClose: 1.0, mouthFunnel: 1.0, mouthPucker: 1.0,
  mouthSmileLeft: 1.0, mouthSmileRight: 1.0, mouthStretchLeft: 1.0, mouthStretchRight: 1.0,
  mouthLowerDownLeft: 1.0, mouthLowerDownRight: 1.0, mouthUpperUpLeft: 1.0, mouthUpperUpRight: 1.0,
  mouthDimpleLeft: 1.0, mouthDimpleRight: 1.0, mouthRollLower: 1.0, mouthRollUpper: 1.0,
  mouthShrugLower: 1.0, mouthShrugUpper: 1.0,
};

export class PlatformController {
  private container: HTMLElement;
  private apiBase: string;
  private dialogueManager: DialogueManager;
  private els: PlatformElements;

  // 状態
  private currentLanguage: string = 'ja';
  private currentMode: string = 'gourmet';
  private dialogueType: DialogueType = 'live';
  private sessionInfo: SessionInfo | null = null;

  // TTS プレーヤー（既存 ConciergeController と同一パターン）
  // ★ このプレーヤーの制御パターンは変更禁止（iPhone対策済み）
  private ttsPlayer: HTMLAudioElement;

  // Live API 状態
  private isLiveStreaming = false;
  private isAISpeaking = false;

  // LAMAvatar 連携用フラグ
  private lamLinked = false;

  constructor(container: HTMLElement, apiBase: string) {
    this.container = container;
    this.apiBase = apiBase;
    this.dialogueManager = new DialogueManager(apiBase);
    this.ttsPlayer = new Audio();

    this.els = {
      chatArea: container.querySelector('.chat-area'),
      userInput: container.querySelector('#userInput') as HTMLInputElement,
      sendBtn: container.querySelector('#sendBtn') as HTMLButtonElement,
      micBtn: container.querySelector('#micBtn') as HTMLButtonElement,
      voiceStatus: container.querySelector('.voice-status'),
      avatarContainer: container.querySelector('.avatar-container'),
      modeIndicator: container.querySelector('#modeIndicator'),
    };

    this.init();
  }

  // ========================================
  // 初期化
  // ========================================

  private async init(): Promise<void> {
    this.setupEventListeners();
    this.linkLAMAvatar();
    this.setupDialogueEvents();
    await this.initializeSession();
  }

  /**
   * UI イベントリスナー設定
   */
  private setupEventListeners(): void {
    // テキスト送信
    this.els.sendBtn?.addEventListener('click', () => this.handleSendMessage());
    this.els.userInput?.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.handleSendMessage();
      }
    });

    // マイクボタン
    // ★ ユーザーインタラクション起点で呼ぶこと（iPhone autoplay policy）
    this.els.micBtn?.addEventListener('click', () => this.toggleMic());
  }

  /**
   * LAMAvatar との統合
   * concierge-controller.ts L43-65 のパターンを踏襲
   */
  private linkLAMAvatar(): void {
    const tryLink = (attempt: number): boolean => {
      const lam = (window as any).lamAvatarController;
      if (lam && typeof lam.setExternalTtsPlayer === 'function') {
        lam.setExternalTtsPlayer(this.ttsPlayer);
        this.lamLinked = true;
        console.log(`[Platform] TTS player linked with LAMAvatar (attempt #${attempt})`);
        return true;
      }
      return false;
    };

    if (!tryLink(1)) {
      [500, 1000, 2000, 4000].forEach((delay, i) => {
        setTimeout(() => tryLink(i + 2), delay);
      });
    }
  }

  /**
   * DialogueManager のイベントハンドラ設定（Live API 経路）
   */
  private setupDialogueEvents(): void {
    // AI テキスト受信
    this.dialogueManager.on('ai_text', (data: { text: string; isPartial: boolean }) => {
      if (!data.isPartial) {
        this.addMessage('assistant', data.text);
      }
    });

    // ユーザーテキスト受信（Live API transcription）
    this.dialogueManager.on('user_text', (data: { text: string; isPartial: boolean }) => {
      if (this.els.userInput) {
        this.els.userInput.value = data.text;
      }
      if (!data.isPartial) {
        this.addMessage('user', data.text);
        if (this.els.userInput) this.els.userInput.value = '';
      }
    });

    // Expression 受信 (Live API 経路)
    this.dialogueManager.on('expression', (data: ExpressionData) => {
      this.applyExpressionFromLive(data);
    });

    // 割り込み
    this.dialogueManager.on('interrupted', () => {
      console.log('[Platform] Barge-in detected');
      this.isAISpeaking = false;
      this.stopAvatarAnimation();
    });

    // 再接続
    this.dialogueManager.on('reconnecting', (reason: string) => {
      console.log(`[Platform] Reconnecting: ${reason}`);
      this.setVoiceStatus('reconnecting');
    });

    this.dialogueManager.on('reconnected', (count: number) => {
      console.log(`[Platform] Reconnected: session #${count}`);
      this.setVoiceStatus('connected');
    });
  }

  // ========================================
  // セッション管理
  // ========================================

  /**
   * セッション初期化
   * concierge-controller.ts L162-233 の initializeSession() に相当
   */
  private async initializeSession(): Promise<void> {
    try {
      this.sessionInfo = await this.dialogueManager.startSession(
        this.currentMode,
        this.currentLanguage,
        this.dialogueType
      );

      // 初回挨拶表示
      this.addMessage('assistant', this.sessionInfo.greeting);

      // Live API モードの場合、マイクボタンを有効化
      if (this.dialogueType === 'live') {
        this.updateModeIndicator('Live API');
      } else {
        this.updateModeIndicator('REST');
      }

      // UI 有効化
      this.enableInput();

      console.log(
        `[Platform] Session started: ${this.sessionInfo.sessionId} ` +
        `mode=${this.currentMode} type=${this.dialogueType}`
      );
    } catch (e) {
      console.error('[Platform] Session init failed:', e);
      this.addMessage('system', 'セッションの初期化に失敗しました。ページをリロードしてください。');
    }
  }

  // ========================================
  // メッセージ送信
  // ========================================

  /**
   * テキストメッセージ送信
   */
  private async handleSendMessage(): Promise<void> {
    const text = this.els.userInput?.value.trim();
    if (!text) return;

    this.addMessage('user', text);
    if (this.els.userInput) this.els.userInput.value = '';
    this.disableInput();

    if (this.dialogueType === 'live') {
      // Live API: テキスト入力を WebSocket 経由で送信
      this.dialogueManager.sendLiveText(text);
      this.enableInput();
    } else {
      // REST: 従来の chat API を使用
      await this.handleRestChat(text);
    }
  }

  /**
   * REST 経路のチャット処理
   * concierge-controller.ts L767-1040 の sendMessage() のパターンを踏襲
   */
  private async handleRestChat(message: string): Promise<void> {
    try {
      const result = await this.dialogueManager.sendChat(message);
      this.addMessage('assistant', result.response);

      // TTS + Expression（REST 経路）
      if (result.response) {
        await this.speakTextRest(result.response);
      }
    } catch (e) {
      console.error('[Platform] Chat error:', e);
      this.addMessage('system', 'メッセージの送信に失敗しました。');
    } finally {
      this.enableInput();
    }
  }

  // ========================================
  // TTS 再生 (REST 経路)
  // ★ ttsPlayer の制御パターンは concierge-controller.ts から変更禁止
  // ========================================

  /**
   * REST 経路の TTS 再生
   * concierge-controller.ts L263-347 の speakTextGCP() パターンを踏襲
   *
   * ★ ttsPlayer.src, .play(), .onended のパターンは変更禁止
   *   iPhone autoplay policy 対策済みのコードパス
   */
  private async speakTextRest(text: string): Promise<void> {
    if (!text.trim()) return;

    const langConfig = LANGUAGE_CONFIG_MAP[this.currentLanguage] || LANGUAGE_CONFIG_MAP['ja'];

    try {
      this.isAISpeaking = true;
      this.startAvatarAnimation();

      const result = await this.dialogueManager.synthesizeTTS(
        text,
        langConfig.tts,
        langConfig.voice
      );

      if (result.success && result.audio) {
        // Expression 同梱データをバッファ投入（concierge-controller.ts L300-304）
        if (result.expression) {
          this.applyExpressionFromTts(result.expression);
        }

        // ★ ttsPlayer 再生パターン — 変更禁止
        this.ttsPlayer.src = `data:audio/mp3;base64,${result.audio}`;
        await new Promise<void>((resolve) => {
          this.ttsPlayer.onended = () => {
            this.isAISpeaking = false;
            this.stopAvatarAnimation();
            resolve();
          };
          this.ttsPlayer.onerror = () => {
            this.isAISpeaking = false;
            this.stopAvatarAnimation();
            resolve();
          };
          this.ttsPlayer.play().catch(() => {
            this.isAISpeaking = false;
            this.stopAvatarAnimation();
            resolve();
          });
        });
      } else {
        this.isAISpeaking = false;
        this.stopAvatarAnimation();
      }
    } catch (e) {
      console.error('[Platform] TTS error:', e);
      this.isAISpeaking = false;
      this.stopAvatarAnimation();
    }
  }

  // ========================================
  // Expression 適用
  // ========================================

  /**
   * REST 経路: TTS 応答に同梱された Expression を LAMAvatar に投入
   * concierge-controller.ts L392-465 の applyExpressionFromTts() をそのまま踏襲
   */
  private applyExpressionFromTts(expression: any): void {
    const lamController = (window as any).lamAvatarController;
    if (!lamController) return;

    if (typeof lamController.clearFrameBuffer === 'function') {
      lamController.clearFrameBuffer();
    }

    if (expression?.names && expression?.frames?.length > 0) {
      const srcFrameRate = expression.frame_rate || 30;

      // Step 1: フォーマット変換 + blendshape 増幅
      // concierge-controller.ts L411-427
      const rawFrames = expression.frames.map((f: any) => {
        const frame: Record<string, number> = {};
        const values: number[] = Array.isArray(f) ? f : (f.weights || []);
        expression.names.forEach((name: string, i: number) => {
          let val = values[i] || 0;
          const amp = MOUTH_AMPLIFY[name];
          if (amp && amp !== 1.0) {
            val = val * amp;
          }
          val = Math.min(BLENDSHAPE_SAFE_MAX, val);
          frame[name] = val;
        });
        return frame;
      });

      // Step 2: フレーム補間 (30fps → 60fps)
      // concierge-controller.ts L430-443
      const interpolatedFrames: Record<string, number>[] = [];
      for (let i = 0; i < rawFrames.length; i++) {
        interpolatedFrames.push(rawFrames[i]);
        if (i < rawFrames.length - 1) {
          const curr = rawFrames[i];
          const next = rawFrames[i + 1];
          const mid: Record<string, number> = {};
          for (const key of Object.keys(curr)) {
            mid[key] = (curr[key] + next[key]) * 0.5;
          }
          interpolatedFrames.push(mid);
        }
      }
      const outputFrameRate = srcFrameRate * 2;

      // Step 3: LAMAvatar にキュー投入
      lamController.queueExpressionFrames(interpolatedFrames, outputFrameRate);

      console.log(
        `[Platform] Expression: ${rawFrames.length}→${interpolatedFrames.length} frames ` +
        `(${srcFrameRate}→${outputFrameRate}fps)`
      );
    }
  }

  /**
   * Live API 経路: Expression データを LAMAvatar に投入
   * relay.py L397-404 から受信した expression を LAMAvatar のフレームバッファに追加
   */
  private applyExpressionFromLive(data: ExpressionData): void {
    const lamController = (window as any).lamAvatarController;
    if (!lamController || !data?.names || !data?.frames?.length) return;

    const frameRate = data.frame_rate || 30;

    // Live API の expression は REST と同じフォーマット
    // relay.py L399-403: { names, frames, frame_rate }
    const frames = data.frames.map((f: any) => {
      const frame: Record<string, number> = {};
      const values: number[] = Array.isArray(f) ? f : (f.weights || []);
      data.names.forEach((name: string, i: number) => {
        let val = values[i] || 0;
        val = Math.min(BLENDSHAPE_SAFE_MAX, val);
        frame[name] = val;
      });
      return frame;
    });

    // Live API はストリーミングなので append（clearFrameBuffer しない）
    lamController.queueExpressionFrames(frames, frameRate);
  }

  // ========================================
  // マイク制御 (Live API)
  // ========================================

  /**
   * マイクの ON/OFF 切替
   * ★ ユーザーインタラクション（click）起点で呼ばれる
   */
  private async toggleMic(): Promise<void> {
    if (this.dialogueType !== 'live') return;

    if (this.isLiveStreaming) {
      this.dialogueManager.stopLiveStream();
      this.isLiveStreaming = false;
      this.setVoiceStatus('stopped');
      this.els.micBtn?.classList.remove('recording');
    } else {
      try {
        await this.dialogueManager.startLiveStream();
        this.isLiveStreaming = true;
        this.setVoiceStatus('listening');
        this.els.micBtn?.classList.add('recording');
      } catch (e) {
        console.error('[Platform] Mic start failed:', e);
        this.setVoiceStatus('error');
      }
    }
  }

  // ========================================
  // UI ヘルパー
  // ========================================

  private addMessage(role: 'user' | 'assistant' | 'system', text: string): void {
    if (!this.els.chatArea) return;

    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;

    const textSpan = document.createElement('span');
    textSpan.className = 'message-text';
    textSpan.textContent = text;
    msgDiv.appendChild(textSpan);

    this.els.chatArea.appendChild(msgDiv);
    this.els.chatArea.scrollTop = this.els.chatArea.scrollHeight;
  }

  private enableInput(): void {
    if (this.els.userInput) this.els.userInput.disabled = false;
    if (this.els.sendBtn) this.els.sendBtn.disabled = false;
    if (this.els.micBtn) this.els.micBtn.disabled = false;
  }

  private disableInput(): void {
    if (this.els.sendBtn) this.els.sendBtn.disabled = true;
    if (this.els.micBtn) this.els.micBtn.disabled = true;
  }

  private setVoiceStatus(status: string): void {
    if (!this.els.voiceStatus) return;
    this.els.voiceStatus.className = `voice-status ${status}`;

    const labels: Record<string, string> = {
      listening: '聞いています...',
      stopped: '',
      speaking: '話しています...',
      reconnecting: '再接続中...',
      connected: '接続済み',
      error: 'エラー',
    };
    this.els.voiceStatus.textContent = labels[status] || '';
  }

  private startAvatarAnimation(): void {
    this.els.avatarContainer?.classList.add('speaking');
  }

  private stopAvatarAnimation(): void {
    this.els.avatarContainer?.classList.remove('speaking');
  }

  private updateModeIndicator(label: string): void {
    if (this.els.modeIndicator) {
      this.els.modeIndicator.textContent = label;
    }
  }

  // ========================================
  // ライフサイクル
  // ========================================

  async destroy(): Promise<void> {
    await this.dialogueManager.endSession();
    this.ttsPlayer.pause();
    this.ttsPlayer.src = '';
  }
}
