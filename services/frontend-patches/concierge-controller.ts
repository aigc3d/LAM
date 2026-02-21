/**
 * ConciergeController - コンシェルジュモード メインコントローラー
 *
 * ★ A2E (Audio2Expression) 統合済み完全差し替え版
 *
 * 変更点:
 *   - ExpressionManager import & 初期化
 *   - /api/tts/synthesize に session_id を追加
 *   - TTS再生時に A2E expression データを優先使用
 *   - stopAvatarAnimation() に ExpressionManager.stop() を追加
 *   - フォールバック: expression データがなければ従来の FFT リップシンク
 *
 * ★マークのコメントが A2E 統合による追加/変更箇所
 */

import { ExpressionManager, ExpressionData } from '../avatar/vrm-expression-manager'; // ★追加

// --- 型定義 ---

interface DOMElements {
  avatarContainer: HTMLElement | null;
  chatContainer: HTMLElement | null;
  inputField: HTMLInputElement | null;
  sendButton: HTMLElement | null;
  micButton: HTMLElement | null;
  statusIndicator: HTMLElement | null;
}

interface LanguageConfig {
  tts: string;
  voice: string;
  stt: string;
}

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

interface TTSResponse {
  success: boolean;
  audio: string;                    // base64 MP3
  expression?: ExpressionData;      // ★ A2E expression データ (optional)
}

// --- 定数 ---

const LANGUAGE_CONFIGS: Record<string, LanguageConfig> = {
  'ja-JP': {
    tts: 'ja-JP',
    voice: 'ja-JP-Neural2-B',
    stt: 'ja-JP',
  },
  'en-US': {
    tts: 'en-US',
    voice: 'en-US-Neural2-J',
    stt: 'en-US',
  },
};

const API_ENDPOINTS = {
  chat: '/api/chat',
  tts: '/api/tts/synthesize',
} as const;

// --- コントローラー本体 ---

export class ConciergeController {
  // DOM要素
  private els: DOMElements = {
    avatarContainer: null,
    chatContainer: null,
    inputField: null,
    sendButton: null,
    micButton: null,
    statusIndicator: null,
  };

  // アバター・描画
  private guavaRenderer: any = null;                           // GVRM レンダラー

  // 音声再生
  private ttsPlayer: HTMLAudioElement = new Audio();
  private audioContext: AudioContext | null = null;
  private analyserNode: AnalyserNode | null = null;
  private animationFrameId: number | null = null;

  // ★追加: A2E Expression Manager
  private expressionManager: ExpressionManager | null = null;

  // セッション管理
  private sessionId: string = '';
  private chatHistory: ChatMessage[] = [];
  private language: string = 'ja-JP';
  private isSpeaking: boolean = false;
  private isListening: boolean = false;

  // 音声認識
  private recognition: any = null;  // SpeechRecognition

  constructor() {
    this.sessionId = crypto.randomUUID();
  }

  // ====================================================
  // 初期化
  // ====================================================

  public async init(guavaRenderer: any) {
    // DOM取得
    this.els.avatarContainer = document.getElementById('avatar-container');
    this.els.chatContainer = document.getElementById('chat-container');
    this.els.inputField = document.querySelector<HTMLInputElement>('#chat-input');
    this.els.sendButton = document.getElementById('send-button');
    this.els.micButton = document.getElementById('mic-button');
    this.els.statusIndicator = document.getElementById('status-indicator');

    // GVRM レンダラー
    this.guavaRenderer = guavaRenderer;

    // ★追加: ExpressionManager 初期化
    if (this.guavaRenderer) {
      this.expressionManager = new ExpressionManager(this.guavaRenderer);
    }

    // イベントリスナー
    this.els.sendButton?.addEventListener('click', () => this.handleSend());
    this.els.inputField?.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.handleSend();
      }
    });
    this.els.micButton?.addEventListener('click', () => this.toggleListening());

    // 音声認識セットアップ
    this.initSpeechRecognition();

    console.log('[ConciergeController] initialized, sessionId:', this.sessionId);
  }

  // ====================================================
  // チャット送信
  // ====================================================

  private async handleSend() {
    const input = this.els.inputField;
    if (!input || !input.value.trim()) return;

    const text = input.value.trim();
    input.value = '';

    await this.sendMessage(text);
  }

  public async sendMessage(text: string) {
    // ユーザーメッセージ表示
    this.appendMessage('user', text);
    this.chatHistory.push({ role: 'user', content: text });
    this.setStatus('thinking');

    try {
      // /api/chat 呼び出し
      const response = await fetch(API_ENDPOINTS.chat, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          session_id: this.sessionId,
          language: this.language,
          history: this.chatHistory.slice(-10),
        }),
      });

      if (!response.ok) throw new Error(`Chat API error: ${response.status}`);

      const data = await response.json();
      const replyText = data.response || data.message || '';

      // アシスタントメッセージ表示
      this.appendMessage('assistant', replyText);
      this.chatHistory.push({ role: 'assistant', content: replyText });

      // TTS再生
      await this.speakText(replyText);

    } catch (error) {
      console.error('[ConciergeController] sendMessage error:', error);
      this.appendMessage('assistant', 'エラーが発生しました。もう一度お試しください。');
    } finally {
      this.setStatus('idle');
    }
  }

  // ====================================================
  // TTS 再生 (★ A2E 統合)
  // ====================================================

  private async speakText(text: string) {
    if (!text.trim()) return;

    // HTMLタグ・マークダウン除去
    const cleanText = text
      .replace(/<[^>]*>/g, '')
      .replace(/[*_~`#]/g, '')
      .trim();

    if (!cleanText) return;

    const langConfig = LANGUAGE_CONFIGS[this.language] || LANGUAGE_CONFIGS['ja-JP'];

    this.isSpeaking = true;
    this.setStatus('speaking');
    this.startAvatarAnimation();

    try {
      const response = await fetch(API_ENDPOINTS.tts, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: cleanText,
          language_code: langConfig.tts,
          voice_name: langConfig.voice,
          session_id: this.sessionId,       // ★追加: A2Eに必要
        }),
      });

      if (!response.ok) throw new Error(`TTS API error: ${response.status}`);

      const result: TTSResponse = await response.json();

      if (result.success && result.audio) {
        const audioSrc = `data:audio/mp3;base64,${result.audio}`;

        // ★ A2E expression データがある場合、ExpressionManager で再生
        if (
          result.expression &&
          ExpressionManager.isValid(result.expression) &&
          this.expressionManager
        ) {
          console.log('[ConciergeController] A2E expression available, frames:',
            result.expression.frames.length);

          this.ttsPlayer.src = audioSrc;

          // FFTリップシンクを無効化（A2Eが担当）
          this.stopFftLipSync();

          // ExpressionManager で音声同期再生
          this.expressionManager.playExpressionFrames(
            result.expression,
            this.ttsPlayer
          );

          await new Promise<void>((resolve) => {
            this.ttsPlayer.onended = () => {
              this.expressionManager?.stop();
              resolve();
            };
            this.ttsPlayer.onerror = () => {
              this.expressionManager?.stop();
              resolve();
            };
            this.ttsPlayer.play().catch((err) => {
              console.warn('[ConciergeController] audio play failed:', err);
              this.expressionManager?.stop();
              resolve();
            });
          });

        } else {
          // ★ フォールバック: 従来の FFT ベースリップシンク
          console.log('[ConciergeController] fallback to FFT lip sync');
          this.ttsPlayer.src = audioSrc;
          this.setupAudioAnalysis();
          this.startLipSyncLoop();

          await new Promise<void>((resolve) => {
            this.ttsPlayer.onended = () => resolve();
            this.ttsPlayer.onerror = () => resolve();
            this.ttsPlayer.play().catch((err) => {
              console.warn('[ConciergeController] audio play failed:', err);
              resolve();
            });
          });
        }
      }

    } catch (error) {
      console.error('[ConciergeController] speakText error:', error);
    } finally {
      this.isSpeaking = false;
      this.stopAvatarAnimation();
      this.setStatus('idle');
    }
  }

  // ====================================================
  // FFT ベース リップシンク (フォールバック)
  // ====================================================

  /**
   * AudioContext + AnalyserNode を使った FFT 分析セットアップ
   * A2E expression が無い場合のフォールバック用
   */
  private setupAudioAnalysis() {
    try {
      if (!this.audioContext) {
        this.audioContext = new AudioContext();
      }

      const source = this.audioContext.createMediaElementSource(this.ttsPlayer);
      this.analyserNode = this.audioContext.createAnalyser();
      this.analyserNode.fftSize = 256;
      this.analyserNode.smoothingTimeConstant = 0.7;

      source.connect(this.analyserNode);
      this.analyserNode.connect(this.audioContext.destination);
    } catch (error) {
      // 既にconnect済みの場合は無視
      console.warn('[ConciergeController] setupAudioAnalysis:', error);
    }
  }

  /**
   * FFT の音量データからリップシンク値を毎フレーム更新
   */
  private startLipSyncLoop() {
    this.stopFftLipSync();

    const updateLipSync = () => {
      if (!this.analyserNode || !this.isSpeaking) {
        this.guavaRenderer?.updateLipSync(0);
        return;
      }

      const dataArray = new Uint8Array(this.analyserNode.frequencyBinCount);
      this.analyserNode.getByteFrequencyData(dataArray);

      // 低周波〜中周波の平均音量を取得（人声の帯域）
      const voiceBins = dataArray.slice(2, 30);
      const avg = voiceBins.reduce((sum, v) => sum + v, 0) / voiceBins.length;

      // 0.0〜1.0 に正規化
      const level = Math.min(1.0, avg / 128);

      this.guavaRenderer?.updateLipSync(level);
      this.animationFrameId = requestAnimationFrame(updateLipSync);
    };

    this.animationFrameId = requestAnimationFrame(updateLipSync);
  }

  /**
   * FFTリップシンクループ停止
   */
  private stopFftLipSync() {
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
  }

  // ====================================================
  // アバター アニメーション制御
  // ====================================================

  private startAvatarAnimation() {
    if (this.els.avatarContainer) {
      this.els.avatarContainer.classList.add('speaking');
    }
  }

  /**
   * ★修正: ExpressionManager も停止する
   */
  private stopAvatarAnimation() {
    if (this.els.avatarContainer) {
      this.els.avatarContainer.classList.remove('speaking');
    }

    // ★ ExpressionManager 停止
    this.expressionManager?.stop();

    // FFT フォールバック用クリーンアップ
    this.guavaRenderer?.updateLipSync(0);
    this.stopFftLipSync();
  }

  // ====================================================
  // 音声認識 (STT)
  // ====================================================

  private initSpeechRecognition() {
    const SpeechRecognition =
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

    if (!SpeechRecognition) {
      console.warn('[ConciergeController] SpeechRecognition not supported');
      if (this.els.micButton) {
        this.els.micButton.style.display = 'none';
      }
      return;
    }

    this.recognition = new SpeechRecognition();
    this.recognition.lang = LANGUAGE_CONFIGS[this.language]?.stt || 'ja-JP';
    this.recognition.interimResults = false;
    this.recognition.continuous = false;

    this.recognition.onresult = (event: any) => {
      const transcript = event.results[0][0].transcript;
      if (transcript.trim()) {
        this.sendMessage(transcript.trim());
      }
    };

    this.recognition.onend = () => {
      this.isListening = false;
      this.els.micButton?.classList.remove('listening');
      this.setStatus('idle');
    };

    this.recognition.onerror = (event: any) => {
      console.warn('[ConciergeController] STT error:', event.error);
      this.isListening = false;
      this.els.micButton?.classList.remove('listening');
    };
  }

  private toggleListening() {
    if (!this.recognition) return;
    if (this.isSpeaking) return; // 発話中はマイク無効

    if (this.isListening) {
      this.recognition.stop();
    } else {
      this.recognition.lang = LANGUAGE_CONFIGS[this.language]?.stt || 'ja-JP';
      this.recognition.start();
      this.isListening = true;
      this.els.micButton?.classList.add('listening');
      this.setStatus('listening');
    }
  }

  // ====================================================
  // UI ヘルパー
  // ====================================================

  private appendMessage(role: 'user' | 'assistant', content: string) {
    if (!this.els.chatContainer) return;

    const msgDiv = document.createElement('div');
    msgDiv.className = `chat-message ${role}`;
    msgDiv.textContent = content;
    this.els.chatContainer.appendChild(msgDiv);
    this.els.chatContainer.scrollTop = this.els.chatContainer.scrollHeight;
  }

  private setStatus(status: 'idle' | 'thinking' | 'speaking' | 'listening') {
    if (!this.els.statusIndicator) return;
    this.els.statusIndicator.dataset.status = status;
  }

  // ====================================================
  // 公開 API
  // ====================================================

  public setLanguage(lang: string) {
    if (LANGUAGE_CONFIGS[lang]) {
      this.language = lang;
      if (this.recognition) {
        this.recognition.lang = LANGUAGE_CONFIGS[lang].stt;
      }
    }
  }

  public getSessionId(): string {
    return this.sessionId;
  }

  /**
   * TTS再生を中断
   */
  public stopSpeaking() {
    this.ttsPlayer.pause();
    this.ttsPlayer.currentTime = 0;
    this.isSpeaking = false;
    this.stopAvatarAnimation();
  }

  /**
   * リソース解放
   */
  public dispose() {
    this.stopSpeaking();
    this.recognition?.stop();
    this.audioContext?.close();

    this.expressionManager?.stop();   // ★追加
    this.expressionManager = null;    // ★追加

    this.guavaRenderer = null;
    this.audioContext = null;
    this.analyserNode = null;
  }
}
