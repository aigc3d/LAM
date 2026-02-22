// src/scripts/chat/concierge-controller.ts
import { CoreController } from './core-controller';
import { AudioManager } from './audio-manager';
// ★追加: 3Dアバターレンダラーのインポート
import { GVRM } from '../../gvrm-format/gvrm';

declare const io: any;

export class ConciergeController extends CoreController {

  // ★追加: GUAVA関連のプロパティ
  private guavaRenderer: GVRM | null = null;
  private analysisContext: AudioContext | null = null;
  private audioAnalyser: AnalyserNode | null = null;
  private analysisSource: MediaElementAudioSourceNode | null = null;
  private animationFrameId: number | null = null;

  // ★A2E: 表情フレーム格納用プロパティ
  private a2eFrames: number[][] | null = null;
  private a2eFrameRate: number = 30;
  private a2eNames: string[] = [];

  // ★並行処理用
  private pendingAckPromise: Promise<void> | null = null;

  constructor(container: HTMLElement, apiBase: string) {
    super(container, apiBase);

    // ★コンシェルジュモード用のAudioManagerを再初期化 (沈黙検知時間を長めに設定)
    this.audioManager = new AudioManager(8000);

    // コンシェルジュモードに設定
    this.currentMode = 'concierge';
    this.init();
  }

  // 初期化プロセスをオーバーライド
  protected async init() {
    // 親クラスの初期化を実行
    await super.init();

    // コンシェルジュ固有の要素とイベントを追加
    const query = (sel: string) => this.container.querySelector(sel) as HTMLElement;

    // ★修正: アバターコンテナの取得 (Concierge.astroの変更に対応)
    this.els.avatarContainer = query('#avatar3DContainer');
    this.els.modeSwitch = query('#modeSwitch') as HTMLInputElement;

    // ★追加: GUAVAレンダラーの初期化
    if (this.els.avatarContainer) {
      this.guavaRenderer = new GVRM(this.els.avatarContainer);

      try {
        // ★修正: 画像パスも正しく指定
        const success = await this.guavaRenderer.loadAssets('/assets/avatar_24p.ply', '/assets/source.png');

        if (success) {
          // 読み込み成功時: フォールバック画像を非表示に
          this.els.avatarContainer.classList.add('loaded');
          const fallback = document.getElementById('avatarFallback');
          if (fallback) fallback.style.display = 'none';
        } else {
          // 読み込み失敗時: フォールバック画像を表示
          console.warn('[GVRM] Asset loading failed, using fallback image');
          this.els.avatarContainer.classList.add('fallback');
        }
      } catch (error) {
        console.error('[GVRM] Initialization error:', error);
        this.els.avatarContainer.classList.add('fallback');
      }
    }

    // モードスイッチのイベントリスナー追加
    if (this.els.modeSwitch) {
      this.els.modeSwitch.addEventListener('change', () => {
        this.toggleMode();
      });
    }
  }

  // ========================================
  // 🎯 セッション初期化をオーバーライド
  // ========================================
  protected async initializeSession() {
    try {
      if (this.sessionId) {
        try {
          await fetch(`${this.apiBase}/api/session/end`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: this.sessionId })
          });
        } catch (e) {}
      }

      // 親クラスのgetUserIdを使用
      const userId = this.getUserId();

      const res = await fetch(`${this.apiBase}/api/session/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_info: { user_id: userId },
          language: this.currentLanguage,
          mode: 'concierge'
        })
      });
      const data = await res.json();
      this.sessionId = data.session_id;

      const greetingText = data.initial_message || this.t('initialGreetingConcierge');
      this.addMessage('assistant', greetingText, null, true);

      const ackTexts = [
        this.t('ackConfirm'), this.t('ackSearch'), this.t('ackUnderstood'),
        this.t('ackYes'), this.t('ttsIntro')
      ];
      const langConfig = this.LANGUAGE_CODE_MAP[this.currentLanguage];

      const ackPromises = ackTexts.map(async (text) => {
        try {
          const ackResponse = await fetch(`${this.apiBase}/api/tts/synthesize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              text: text, language_code: langConfig.tts, voice_name: langConfig.voice,
              session_id: this.sessionId  // ★A2E: session_id追加
            })
          });
          const ackData = await ackResponse.json();
          if (ackData.success && ackData.audio) {
            this.preGeneratedAcks.set(text, ackData.audio);
          }
        } catch (_e) { }
      });

      await Promise.all([
        this.speakTextGCP(greetingText),
        ...ackPromises
      ]);

      this.els.userInput.disabled = false;
      this.els.sendBtn.disabled = false;
      this.els.micBtn.disabled = false;
      this.els.speakerBtn.disabled = false;
      this.els.speakerBtn.classList.remove('disabled');
      this.els.reservationBtn.classList.remove('visible');

    } catch (e) {
      console.error('[Session] Initialization error:', e);
    }
  }

  // ========================================
  // 🔧 Socket.IOの初期化をオーバーライド
  // ========================================
  protected initSocket() {
    // @ts-ignore
    this.socket = io(this.apiBase || window.location.origin);

    this.socket.on('connect', () => { });

    this.socket.on('transcript', (data: any) => {
      const { text, is_final } = data;
      if (this.isAISpeaking) return;
      if (is_final) {
        this.handleStreamingSTTComplete(text);
        this.currentAISpeech = "";
      } else {
        this.els.userInput.value = text;
      }
    });

    this.socket.on('error', (data: any) => {
      this.addMessage('system', `${this.t('sttError')} ${data.message}`);
      if (this.isRecording) this.stopStreamingSTT();
    });
  }

  // ========================================
  // 👄 GUAVA連携: 音声再生とリップシンク + A2E統合
  // ========================================

  // ★オーバーライド: 音声再生時にA2EリップシンクまたはFFTフォールバック
  // ※ session_id を送るため super.speakTextGCP() は使わず、インラインでTTSフェッチ
  protected async speakTextGCP(text: string, stopPrevious: boolean = true, autoRestartMic: boolean = false, skipAudio: boolean = false) {
    if (skipAudio || !this.isTTSEnabled || !text) return Promise.resolve();

    if (stopPrevious) {
      this.stopCurrentAudio();
    }

    // ★GUAVA: リップシンク用のオーディオ解析をセットアップ
    this.setupAudioAnalysis();

    // ★GUAVA: 待機アニメーションなどを制御
    if (this.els.avatarContainer) {
      this.els.avatarContainer.classList.add('speaking');
    }

    const cleanText = this.stripMarkdown(text);
    try {
      this.isAISpeaking = true;
      if (this.isRecording && (this.isIOS || this.isAndroid)) {
        this.stopStreamingSTT();
      }

      this.els.voiceStatus.innerHTML = this.t('voiceStatusSynthesizing');
      this.els.voiceStatus.className = 'voice-status speaking';
      const langConfig = this.LANGUAGE_CODE_MAP[this.currentLanguage];

      // ★A2E: session_id付きでTTS取得（expressionデータ同梱）
      const response = await fetch(`${this.apiBase}/api/tts/synthesize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: cleanText, language_code: langConfig.tts, voice_name: langConfig.voice,
          session_id: this.sessionId  // ★A2E: session_id追加
        })
      });
      const data = await response.json();

      if (data.success && data.audio) {
        // ★A2E: expressionデータがあればA2Eフレームを設定
        this.setA2EFrames(data.expression);

        this.ttsPlayer.src = `data:audio/mp3;base64,${data.audio}`;
        const playPromise = new Promise<void>((resolve) => {
          this.ttsPlayer.onended = async () => {
            this.els.voiceStatus.innerHTML = this.t('voiceStatusStopped');
            this.els.voiceStatus.className = 'voice-status stopped';
            this.isAISpeaking = false;
            this.stopAvatarAnimation();
            if (autoRestartMic) {
              if (!this.isRecording) {
                try { await this.toggleRecording(); } catch (_error) { this.showMicPrompt(); }
              }
            }
            resolve();
          };
          this.ttsPlayer.onerror = () => {
            this.isAISpeaking = false;
            this.stopAvatarAnimation();
            resolve();
          };
        });

        if (this.isUserInteracted) {
          this.lastAISpeech = this.normalizeText(cleanText);
          // ★GUAVA: リップシンクループ開始
          this.startLipSyncLoop();
          await this.ttsPlayer.play();
          await playPromise;
        } else {
          this.showClickPrompt();
          this.els.voiceStatus.innerHTML = this.t('voiceStatusStopped');
          this.els.voiceStatus.className = 'voice-status stopped';
          this.isAISpeaking = false;
          this.stopAvatarAnimation();
        }
      } else {
        this.isAISpeaking = false;
        this.stopAvatarAnimation();
      }
    } catch (_error) {
      this.els.voiceStatus.innerHTML = this.t('voiceStatusStopped');
      this.els.voiceStatus.className = 'voice-status stopped';
      this.isAISpeaking = false;
      this.stopAvatarAnimation();
    }
  }

  // ★追加: 音声解析のセットアップ
  private setupAudioAnalysis() {
    if (!this.guavaRenderer) return;

    // AudioContextの作成（初回のみ）
    if (!this.analysisContext) {
      const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
      this.analysisContext = new AudioContextClass();
    }

    // ユーザー操作後なのでresumeを試みる
    if (this.analysisContext.state === 'suspended') {
      this.analysisContext.resume().catch(e => console.log('AudioContext resume failed:', e));
    }

    // AnalyserNodeの作成
    if (!this.audioAnalyser) {
      this.audioAnalyser = this.analysisContext.createAnalyser();
      this.audioAnalyser.fftSize = 256;
    }

    // MediaElementSourceの接続（初回のみ）
    if (!this.analysisSource && this.ttsPlayer) {
      try {
        this.analysisSource = this.analysisContext.createMediaElementSource(this.ttsPlayer);
        this.analysisSource.connect(this.audioAnalyser);
        this.audioAnalyser.connect(this.analysisContext.destination);
      } catch (e) {
        console.warn('MediaElementSource connection error:', e);
      }
    }
  }

  // ★A2E: 表情フレームの設定（A2Eデータがあればセット、なければクリア）
  private setA2EFrames(expression: any) {
    if (expression?.names && expression?.frames?.length > 0) {
      this.a2eNames = expression.names;
      this.a2eFrames = expression.frames;
      this.a2eFrameRate = expression.frame_rate || 30;
    } else {
      this.a2eFrames = null;
    }
  }

  // ★A2E: 52次元ARKitブレンドシェイプから口の開き度合い（0.0-1.0）を計算
  private computeMouthOpenness(frame: number[]): number {
    // ARKitブレンドシェイプのインデックス（a2e_engine.pyのARKIT_BLENDSHAPE_NAMESに対応）
    const jawOpenIdx = this.a2eNames.indexOf('jawOpen');
    const mouthFunnelIdx = this.a2eNames.indexOf('mouthFunnel');
    const mouthPuckerIdx = this.a2eNames.indexOf('mouthPucker');
    const mouthLowerDownLIdx = this.a2eNames.indexOf('mouthLowerDownLeft');
    const mouthLowerDownRIdx = this.a2eNames.indexOf('mouthLowerDownRight');
    const mouthUpperUpLIdx = this.a2eNames.indexOf('mouthUpperUpLeft');
    const mouthUpperUpRIdx = this.a2eNames.indexOf('mouthUpperUpRight');

    const jawOpen = jawOpenIdx >= 0 ? (frame[jawOpenIdx] || 0) : 0;
    const mouthFunnel = mouthFunnelIdx >= 0 ? (frame[mouthFunnelIdx] || 0) : 0;
    const mouthPucker = mouthPuckerIdx >= 0 ? (frame[mouthPuckerIdx] || 0) : 0;
    const mouthLowerDownL = mouthLowerDownLIdx >= 0 ? (frame[mouthLowerDownLIdx] || 0) : 0;
    const mouthLowerDownR = mouthLowerDownRIdx >= 0 ? (frame[mouthLowerDownRIdx] || 0) : 0;
    const mouthUpperUpL = mouthUpperUpLIdx >= 0 ? (frame[mouthUpperUpLIdx] || 0) : 0;
    const mouthUpperUpR = mouthUpperUpRIdx >= 0 ? (frame[mouthUpperUpRIdx] || 0) : 0;

    // 重み付き合成（vrm-expression-manager.tsのapplyBlendshapesと同じロジック）
    return Math.min(1.0,
      jawOpen * 0.6 +
      ((mouthLowerDownL + mouthLowerDownR) / 2) * 0.2 +
      ((mouthUpperUpL + mouthUpperUpR) / 2) * 0.1 +
      mouthFunnel * 0.05 +
      mouthPucker * 0.05
    );
  }

  // ★修正: リップシンクループ - A2Eフレーム優先、FFTフォールバック
  private startLipSyncLoop() {
    if (this.animationFrameId) cancelAnimationFrame(this.animationFrameId);

    const update = () => {
      // 再生停止中または終了時は口を閉じる
      if (this.ttsPlayer.paused || this.ttsPlayer.ended) {
        this.guavaRenderer?.updateLipSync(0);

        if (this.ttsPlayer.ended) {
           this.animationFrameId = null;
           return;
        }
      }

      if (this.guavaRenderer && !this.ttsPlayer.paused) {
        // ★A2E: フレームがあればA2Eデータを使用（フォーマット対応可能）
        if (this.a2eFrames && this.a2eFrames.length > 0) {
          const currentTime = this.ttsPlayer.currentTime;
          const frameIdx = Math.min(
            Math.floor(currentTime * this.a2eFrameRate),
            this.a2eFrames.length - 1
          );
          if (frameIdx >= 0) {
            const mouthOpenness = this.computeMouthOpenness(this.a2eFrames[frameIdx]);
            this.guavaRenderer.updateLipSync(mouthOpenness);
          }
        }
        // ★FFTフォールバック: A2Eデータがなければ従来の音量分析
        else if (this.audioAnalyser) {
          const dataArray = new Uint8Array(this.audioAnalyser.frequencyBinCount);
          this.audioAnalyser.getByteFrequencyData(dataArray);

          let sum = 0;
          const range = dataArray.length;
          for (let i = 0; i < range; i++) {
            sum += dataArray[i];
          }
          const average = sum / range;
          const normalizedLevel = Math.min(1.0, (average / 255.0) * 2.5);
          this.guavaRenderer.updateLipSync(normalizedLevel);
        }
      }

      this.animationFrameId = requestAnimationFrame(update);
    };

    this.animationFrameId = requestAnimationFrame(update);
  }

  // アバターアニメーション停止
  private stopAvatarAnimation() {
    if (this.els.avatarContainer) {
      this.els.avatarContainer.classList.remove('speaking');
    }
    // 口を閉じる
    this.guavaRenderer?.updateLipSync(0);
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
    // ★A2E: フレームクリア
    this.a2eFrames = null;
  }

  // ========================================
  // 🎯 UI言語更新をオーバーライド
  // ========================================
  protected updateUILanguage() {
    const initialMessage = this.els.chatArea.querySelector('.message.assistant[data-initial="true"] .message-text');
    const savedGreeting = initialMessage?.textContent;

    super.updateUILanguage();

    if (initialMessage && savedGreeting) {
      initialMessage.textContent = savedGreeting;
    }

    const pageTitle = document.getElementById('pageTitle');
    if (pageTitle) {
      pageTitle.innerHTML = `<img src="/pwa-152x152.png" alt="Logo" class="app-logo" /> ${this.t('pageTitleConcierge')}`;
    }
  }

  // モード切り替え処理 - ページ遷移
  private toggleMode() {
    const isChecked = this.els.modeSwitch?.checked;
    if (!isChecked) {
      console.log('[ConciergeController] Switching to Chat mode...');
      window.location.href = '/';
    }
  }

  // すべての活動を停止
  protected stopAllActivities() {
    super.stopAllActivities();
    this.stopAvatarAnimation();
  }

  // ========================================
  // 🎯 並行処理フロー: 応答を分割してTTS処理
  // ========================================
  private splitIntoSentences(text: string, language: string): string[] {
    let separator: RegExp;

    if (language === 'ja' || language === 'zh') {
      separator = /。/;
    } else {
      separator = /\.\s+/;
    }

    const sentences = text.split(separator).filter(s => s.trim().length > 0);

    return sentences.map((s, idx) => {
      if (idx < sentences.length - 1 || text.endsWith('。') || text.endsWith('. ')) {
        return language === 'ja' || language === 'zh' ? s + '。' : s + '. ';
      }
      return s;
    });
  }

  private async speakResponseInChunks(response: string, isTextInput: boolean = false) {
    if (isTextInput || !this.isTTSEnabled) {
      return this.speakTextGCP(response, true, false, isTextInput);
    }

    try {
      // ★ ack再生中ならttsPlayer解放を待つ
      if (this.pendingAckPromise) {
        await this.pendingAckPromise;
        this.pendingAckPromise = null;
      }
      this.stopCurrentAudio();

      this.isAISpeaking = true;
      if (this.isRecording) {
        this.stopStreamingSTT();
      }

      // ★GUAVA: リップシンク準備
      this.setupAudioAnalysis();

      const sentences = this.splitIntoSentences(response, this.currentLanguage);

      if (sentences.length <= 1) {
        await this.speakTextGCP(response, true, false, isTextInput);
        this.isAISpeaking = false;
        return;
      }

      const firstSentence = sentences[0];
      const remainingSentences = sentences.slice(1).join('');
      const langConfig = this.LANGUAGE_CODE_MAP[this.currentLanguage];

      if (this.isUserInteracted) {
        const cleanFirst = this.stripMarkdown(firstSentence);
        const cleanRemaining = remainingSentences.trim().length > 0
          ? this.stripMarkdown(remainingSentences) : null;

        // ★A2E: session_id付きでTTS取得
        const firstTtsPromise = fetch(`${this.apiBase}/api/tts/synthesize`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            text: cleanFirst, language_code: langConfig.tts,
            voice_name: langConfig.voice, session_id: this.sessionId
          })
        }).then(r => r.json());

        const remainingTtsPromise = cleanRemaining
          ? fetch(`${this.apiBase}/api/tts/synthesize`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                text: cleanRemaining, language_code: langConfig.tts,
                voice_name: langConfig.voice, session_id: this.sessionId
              })
            }).then(r => r.json())
          : null;

        const firstTtsResult = await firstTtsPromise;
        if (firstTtsResult.success && firstTtsResult.audio) {
          // ★A2E: expressionデータをセット
          this.setA2EFrames(firstTtsResult.expression);

          this.lastAISpeech = this.normalizeText(cleanFirst);
          this.stopCurrentAudio();
          this.ttsPlayer.src = `data:audio/mp3;base64,${firstTtsResult.audio}`;

          let remainingTtsResult: any = null;
          if (remainingTtsPromise) {
            remainingTtsResult = await remainingTtsPromise;
          }

          // ★GUAVA: リップシンクループ開始
          this.startLipSyncLoop();

          await new Promise<void>((resolve) => {
            this.ttsPlayer.onended = () => {
              this.els.voiceStatus.innerHTML = this.t('voiceStatusStopped');
              this.els.voiceStatus.className = 'voice-status stopped';
              resolve();
            };
            this.els.voiceStatus.innerHTML = this.t('voiceStatusSpeaking');
            this.els.voiceStatus.className = 'voice-status speaking';
            this.ttsPlayer.play();
          });

          if (remainingTtsResult?.success && remainingTtsResult?.audio) {
            this.lastAISpeech = this.normalizeText(cleanRemaining || '');

            // ★A2E: 次セグメントのexpressionをセット
            this.setA2EFrames(remainingTtsResult.expression);

            this.stopCurrentAudio();
            this.ttsPlayer.src = `data:audio/mp3;base64,${remainingTtsResult.audio}`;

            // ★GUAVA: リップシンク継続
            this.startLipSyncLoop();

            await new Promise<void>((resolve) => {
              this.ttsPlayer.onended = () => {
                this.els.voiceStatus.innerHTML = this.t('voiceStatusStopped');
                this.els.voiceStatus.className = 'voice-status stopped';
                resolve();
              };
              this.els.voiceStatus.innerHTML = this.t('voiceStatusSpeaking');
              this.els.voiceStatus.className = 'voice-status speaking';
              this.ttsPlayer.play();
            });
          }
        }
      }

      this.stopAvatarAnimation();
      this.isAISpeaking = false;
    } catch (error) {
      console.error('[TTS並行処理エラー]', error);
      this.isAISpeaking = false;
      await this.speakTextGCP(response, true, false, isTextInput);
    }
  }

  // ========================================
  // 🎯 コンシェルジュモード専用: 音声入力完了時の即答処理
  // ========================================
  protected async handleStreamingSTTComplete(transcript: string) {
    this.stopStreamingSTT();

    if ('mediaSession' in navigator) {
      try { navigator.mediaSession.playbackState = 'playing'; } catch (e) {}
    }

    this.els.voiceStatus.innerHTML = this.t('voiceStatusComplete');
    this.els.voiceStatus.className = 'voice-status';

    const normTranscript = this.normalizeText(transcript);
    if (this.isSemanticEcho(normTranscript, this.lastAISpeech)) {
        this.els.voiceStatus.innerHTML = this.t('voiceStatusStopped');
        this.els.voiceStatus.className = 'voice-status stopped';
        this.lastAISpeech = '';
        return;
    }

    this.els.userInput.value = transcript;
    this.addMessage('user', transcript);

    const textLength = transcript.trim().replace(/\s+/g, '').length;
    if (textLength < 2) {
        const msg = this.t('shortMsgWarning');
        this.addMessage('assistant', msg);
        if (this.isTTSEnabled && this.isUserInteracted) {
          await this.speakTextGCP(msg, true);
        } else {
          await new Promise(r => setTimeout(r, 2000));
        }
        this.els.userInput.value = '';
        this.els.voiceStatus.innerHTML = this.t('voiceStatusStopped');
        this.els.voiceStatus.className = 'voice-status stopped';
        return;
    }

    const ackText = this.t('ackYes');
    const preGeneratedAudio = this.preGeneratedAcks.get(ackText);

    if (preGeneratedAudio && this.isTTSEnabled && this.isUserInteracted) {
      this.pendingAckPromise = new Promise<void>((resolve) => {
        // ★GUAVA: リップシンク準備
        this.setupAudioAnalysis();

        this.lastAISpeech = this.normalizeText(ackText);
        this.ttsPlayer.src = `data:audio/mp3;base64,${preGeneratedAudio}`;
        this.ttsPlayer.onended = () => resolve();
        this.ttsPlayer.play().catch(_e => resolve());
      });
    } else if (this.isTTSEnabled) {
      this.pendingAckPromise = this.speakTextGCP(ackText, false);
    }

    this.addMessage('assistant', ackText);

    // ★ 並行処理: ack再生完了を待たず、即LLMリクエスト開始
    if (this.els.userInput.value.trim()) {
      this.isFromVoiceInput = true;
      this.sendMessage();
    }

    this.els.voiceStatus.innerHTML = this.t('voiceStatusStopped');
    this.els.voiceStatus.className = 'voice-status stopped';
  }

  // ========================================
  // 🎯 コンシェルジュモード専用: メッセージ送信処理
  // ========================================
  protected async sendMessage() {
    let firstAckPromise: Promise<void> | null = null;
    // ★ voice入力時はunlockAudioParamsスキップ（ack再生中のttsPlayerを中断させない）
    if (!this.pendingAckPromise) {
      this.unlockAudioParams();
    }
    const message = this.els.userInput.value.trim();
    if (!message || this.isProcessing) return;

    const currentSessionId = this.sessionId;
    const isTextInput = !this.isFromVoiceInput;

    this.isProcessing = true;
    this.els.sendBtn.disabled = true;
    this.els.micBtn.disabled = true;
    this.els.userInput.disabled = true;

    if (!this.isFromVoiceInput) {
      this.addMessage('user', message);
      const textLength = message.trim().replace(/\s+/g, '').length;
      if (textLength < 2) {
           const msg = this.t('shortMsgWarning');
           this.addMessage('assistant', msg);
           if (this.isTTSEnabled && this.isUserInteracted) await this.speakTextGCP(msg, true);
           this.resetInputState();
           return;
      }

      this.els.userInput.value = '';

      const ackText = this.t('ackYes');
      this.currentAISpeech = ackText;
      this.addMessage('assistant', ackText);

      if (this.isTTSEnabled && !isTextInput) {
        try {
          const preGeneratedAudio = this.preGeneratedAcks.get(ackText);
          if (preGeneratedAudio && this.isUserInteracted) {
            firstAckPromise = new Promise<void>((resolve) => {
              // ★GUAVA: リップシンク準備
              this.setupAudioAnalysis();

              this.lastAISpeech = this.normalizeText(ackText);
              this.ttsPlayer.src = `data:audio/mp3;base64,${preGeneratedAudio}`;
              this.ttsPlayer.onended = () => resolve();
              this.ttsPlayer.play().catch(_e => resolve());
            });
          } else {
            firstAckPromise = this.speakTextGCP(ackText, false);
          }
        } catch (_e) {}
      }
      if (firstAckPromise) await firstAckPromise;
    }

    this.isFromVoiceInput = false;

    if (this.waitOverlayTimer) clearTimeout(this.waitOverlayTimer);
    let responseReceived = false;

    this.waitOverlayTimer = window.setTimeout(() => {
      if (!responseReceived) {
        this.showWaitOverlay();
      }
    }, 6500);

    try {
      const response = await fetch(`${this.apiBase}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: currentSessionId,
          message: message,
          stage: this.currentStage,
          language: this.currentLanguage,
          mode: this.currentMode
        })
      });
      const data = await response.json();
      responseReceived = true;

      if (this.sessionId !== currentSessionId) return;

      if (this.waitOverlayTimer) {
        clearTimeout(this.waitOverlayTimer);
        this.waitOverlayTimer = null;
      }
      this.hideWaitOverlay();
      this.currentAISpeech = data.response;
      this.addMessage('assistant', data.response, data.summary);

      if (!isTextInput && this.isTTSEnabled) {
        this.stopCurrentAudio();
      }

      if (data.shops && data.shops.length > 0) {
        this.currentShops = data.shops;
        this.els.reservationBtn.classList.add('visible');
        this.els.userInput.value = '';
        document.dispatchEvent(new CustomEvent('displayShops', {
          detail: { shops: data.shops, language: this.currentLanguage }
        }));

        const section = document.getElementById('shopListSection');
        if (section) section.classList.add('has-shops');
        if (window.innerWidth < 1024) {
          setTimeout(() => {
            const shopSection = document.getElementById('shopListSection');
            if (shopSection) shopSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
           }, 300);
        }

        (async () => {
          try {
            // ★ ack再生中ならttsPlayer解放を待つ
            if (this.pendingAckPromise) {
              await this.pendingAckPromise;
              this.pendingAckPromise = null;
            }
            this.stopCurrentAudio();

            this.isAISpeaking = true;
            if (this.isRecording) { this.stopStreamingSTT(); }

            await this.speakTextGCP(this.t('ttsIntro'), true, false, isTextInput);

            const lines = data.response.split('\n\n');
            let introText = "";
            let shopLines = lines;
            if (lines[0].includes('ご希望に合うお店') && lines[0].includes('ご紹介します')) {
              introText = lines[0];
              shopLines = lines.slice(1);
            }

            let introPart2Promise: Promise<void> | null = null;
            if (introText && this.isTTSEnabled && this.isUserInteracted && !isTextInput) {
                const preGeneratedIntro = this.preGeneratedAcks.get(introText);
              if (preGeneratedIntro) {
                introPart2Promise = new Promise<void>((resolve) => {
                  this.setupAudioAnalysis();
                  this.lastAISpeech = this.normalizeText(introText);
                  this.ttsPlayer.src = `data:audio/mp3;base64,${preGeneratedIntro}`;
                  this.ttsPlayer.onended = () => resolve();
                  this.ttsPlayer.play();
                });
              } else {
                introPart2Promise = this.speakTextGCP(introText, false, false, isTextInput);
              }
            }

            // ★A2E: session_id付きでショップTTS取得
            let firstShopTtsPromise: Promise<any> | null = null;
            let remainingShopTtsPromise: Promise<any> | null = null;
            const shopLangConfig = this.LANGUAGE_CODE_MAP[this.currentLanguage];

            if (shopLines.length > 0 && this.isTTSEnabled && this.isUserInteracted && !isTextInput) {
              const firstShop = shopLines[0];
              const restShops = shopLines.slice(1).join('\n\n');
              firstShopTtsPromise = fetch(`${this.apiBase}/api/tts/synthesize`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  text: this.stripMarkdown(firstShop), language_code: shopLangConfig.tts,
                  voice_name: shopLangConfig.voice, session_id: this.sessionId
                })
              }).then(r => r.json());

              if (restShops) {
                remainingShopTtsPromise = fetch(`${this.apiBase}/api/tts/synthesize`, {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({
                    text: this.stripMarkdown(restShops), language_code: shopLangConfig.tts,
                    voice_name: shopLangConfig.voice, session_id: this.sessionId
                  })
                }).then(r => r.json());
              }
            }

            if (introPart2Promise) await introPart2Promise;

            if (firstShopTtsPromise) {
              const firstResult = await firstShopTtsPromise;
              if (firstResult?.success && firstResult?.audio) {
                const firstShopText = this.stripMarkdown(shopLines[0]);
                this.lastAISpeech = this.normalizeText(firstShopText);

                // ★A2E: expressionデータをセット
                this.setA2EFrames(firstResult.expression);

                if (!isTextInput && this.isTTSEnabled) {
                  this.stopCurrentAudio();
                }

                this.ttsPlayer.src = `data:audio/mp3;base64,${firstResult.audio}`;
                this.startLipSyncLoop();

                let remainingResult: any = null;
                if (remainingShopTtsPromise) {
                  remainingResult = await remainingShopTtsPromise;
                }

                await new Promise<void>((resolve) => {
                  this.ttsPlayer.onended = () => {
                    this.els.voiceStatus.innerHTML = this.t('voiceStatusStopped');
                    this.els.voiceStatus.className = 'voice-status stopped';
                    resolve();
                  };
                  this.els.voiceStatus.innerHTML = this.t('voiceStatusSpeaking');
                  this.els.voiceStatus.className = 'voice-status speaking';
                  this.ttsPlayer.play();
                });

                if (remainingResult?.success && remainingResult?.audio) {
                  const restShopsText = this.stripMarkdown(shopLines.slice(1).join('\n\n'));
                  this.lastAISpeech = this.normalizeText(restShopsText);

                  // ★A2E: expressionデータをセット
                  this.setA2EFrames(remainingResult.expression);

                  if (!isTextInput && this.isTTSEnabled) {
                    this.stopCurrentAudio();
                  }

                  this.ttsPlayer.src = `data:audio/mp3;base64,${remainingResult.audio}`;
                  this.startLipSyncLoop();

                  await new Promise<void>((resolve) => {
                    this.ttsPlayer.onended = () => {
                      this.els.voiceStatus.innerHTML = this.t('voiceStatusStopped');
                      this.els.voiceStatus.className = 'voice-status stopped';
                      resolve();
                    };
                    this.els.voiceStatus.innerHTML = this.t('voiceStatusSpeaking');
                    this.els.voiceStatus.className = 'voice-status speaking';
                    this.ttsPlayer.play();
                  });
                }
              }
            }
            this.isAISpeaking = false;
            this.stopAvatarAnimation();
          } catch (_e) {
            this.isAISpeaking = false;
            this.stopAvatarAnimation();
          }
        })();
      } else {
        if (data.response) {
          const extractedShops = this.extractShopsFromResponse(data.response);
          if (extractedShops.length > 0) {
            this.currentShops = extractedShops;
            this.els.reservationBtn.classList.add('visible');
            document.dispatchEvent(new CustomEvent('displayShops', {
              detail: { shops: extractedShops, language: this.currentLanguage }
            }));
            const section = document.getElementById('shopListSection');
            if (section) section.classList.add('has-shops');
            this.speakResponseInChunks(data.response, isTextInput);
          } else {
            this.speakResponseInChunks(data.response, isTextInput);
          }
        }
      }
    } catch (error) {
      console.error('送信エラー:', error);
      this.hideWaitOverlay();
      this.showError('メッセージの送信に失敗しました。');
    } finally {
      this.resetInputState();
      this.els.userInput.blur();
    }
  }

}
