

// src/scripts/chat/concierge-controller.ts
import { CoreController } from './core-controller';
import { AudioManager } from './audio-manager';

declare const io: any;

export class ConciergeController extends CoreController {
  // Audio2Expression WebSocket URL (Cloud Run)
  private audio2expWsUrl = 'wss://audio2exp-service-6s2ds5mdba-uc.a.run.app/ws';

  constructor(container: HTMLElement, apiBase: string) {
    super(container, apiBase);

    // ★コンシェルジュモード用のAudioManagerを6.5秒設定で再初期化２
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
    this.els.avatarContainer = query('.avatar-container');
    this.els.avatarImage = query('#avatarImage') as HTMLImageElement;
    this.els.modeSwitch = query('#modeSwitch') as HTMLInputElement;

    // モードスイッチのイベントリスナー追加
    if (this.els.modeSwitch) {
      this.els.modeSwitch.addEventListener('change', () => {
        this.toggleMode();
      });
    }

    // ★ LAMAvatar との統合: 外部TTSプレーヤーをリンク
    // LAMAvatar が後から初期化される可能性があるため、即時 + 遅延でリンク
    const linkTtsPlayer = () => {
      const lam = (window as any).lamAvatarController;
      if (lam && typeof lam.setExternalTtsPlayer === 'function') {
        lam.setExternalTtsPlayer(this.ttsPlayer);
        console.log('[Concierge] Linked external TTS player with LAMAvatar');
        return true;
      }
      return false;
    };
    if (!linkTtsPlayer()) {
      setTimeout(() => linkTtsPlayer(), 2000);
    }
  }

  // ========================================
  // 🎯 セッション初期化をオーバーライド(挨拶文を変更)
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

      // ★ user_id を取得（親クラスのメソッドを使用）
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

      // ★ LAMAvatar を audio2exp-service WebSocket に接続（リップシンク用）
      this.connectLAMAvatarWebSocket();

      // ✅ バックエンドからの初回メッセージを使用（長期記憶対応）
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
              session_id: this.sessionId
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
    
    // ✅ コンシェルジュ版のhandleStreamingSTTCompleteを呼ぶように再登録
    this.socket.on('transcript', (data: any) => {
      const { text, is_final } = data;
      if (this.isAISpeaking) return;
      if (is_final) {
        this.handleStreamingSTTComplete(text); // ← オーバーライド版が呼ばれる
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

  // Audio2Expression REST API URL (Cloud Run)
  private audio2expApiUrl = 'https://audio2exp-service-6s2ds5mdba-uc.a.run.app';

  // 表情フレーム一時保存（並行取得時にバッファ競合を防ぐ）
  private pendingExpressionFrames: { frames: any[], frameRate: number } | null = null;

  // コンシェルジュモード固有: アバターアニメーション制御 + 公式リップシンク
  protected async speakTextGCP(text: string, stopPrevious: boolean = true, autoRestartMic: boolean = false, skipAudio: boolean = false) {
    if (skipAudio || !this.isTTSEnabled || !text) return Promise.resolve();

    if (stopPrevious) {
      this.ttsPlayer.pause();
    }

    // アバターアニメーションを開始
    if (this.els.avatarContainer) {
      this.els.avatarContainer.classList.add('speaking');
    }

    // ★ 公式同期: TTS音声をaudio2exp-serviceに送信して表情を生成
    const cleanText = this.stripMarkdown(text);
    try {
      this.isAISpeaking = true;
      if (this.isRecording && (this.isIOS || this.isAndroid)) {
        this.stopStreamingSTT();
      }

      this.els.voiceStatus.innerHTML = this.t('voiceStatusSynthesizing');
      this.els.voiceStatus.className = 'voice-status speaking';
      const langConfig = this.LANGUAGE_CODE_MAP[this.currentLanguage];

      // TTS音声を取得
      const response = await fetch(`${this.apiBase}/api/tts/synthesize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: cleanText, language_code: langConfig.tts, voice_name: langConfig.voice,
          session_id: this.sessionId
        })
      });
      const data = await response.json();

      if (data.success && data.audio) {
        // ★ 表情生成を開始（タイムアウト超過時はリップシンクなしでTTS再生続行）
        const expPromise = this.sendAudioToExpression(data.audio, true, true);
        this.ttsPlayer.src = `data:audio/mp3;base64,${data.audio}`;
        await Promise.race([
          expPromise,
          new Promise<void>(resolve => setTimeout(resolve, this.EXPRESSION_API_TIMEOUT_MS))
        ]);
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

  /**
   * 公式リップシンク: TTS音声をaudio2exp-serviceに送信
   * 表情データを受け取り、LAMAvatarのキューに追加
   *
   * ★ isStart=true の場合のみバッファをクリア（新しいスピーチの開始）
   * 残りのセグメントは呼び出し元で明示的にクリアする
   */
  // 表情APIタイムアウト（コールドスタート考慮、これを超えたらリップシンクなしで続行）
  private readonly EXPRESSION_API_TIMEOUT_MS = 8000;

  private async sendAudioToExpression(audioBase64: string, isStart: boolean = false, isFinal: boolean = false): Promise<void> {
    if (!this.sessionId) return;

    const lamController = (window as any).lamAvatarController;

    // ★ 新しいスピーチの開始時のみバッファをクリア
    if (isStart && lamController && typeof lamController.clearFrameBuffer === 'function') {
      lamController.clearFrameBuffer();
      console.log('[Concierge] Cleared frame buffer for new speech');
    }

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.EXPRESSION_API_TIMEOUT_MS);

      const response = await fetch(`${this.audio2expApiUrl}/api/audio2expression`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          audio_base64: audioBase64,
          session_id: this.sessionId,
          is_start: isStart,
          is_final: isFinal,
          audio_format: 'mp3'
        }),
        signal: controller.signal
      });
      clearTimeout(timeoutId);

      if (!response.ok) {
        console.warn('[Concierge] audio2exp API failed:', response.status);
      } else {
        const result = await response.json();
        const frameCount = result.frames?.length || 0;
        console.log(`[Concierge] Expression generated: ${frameCount} frames, batch=${result.batch_id}`);

        // ★ 公式形式: names + frames[{weights}] を変換
        if (frameCount > 0 && result.names && result.frames) {
          if (lamController && typeof lamController.queueExpressionFrames === 'function') {
            // 公式 gaussianAvatar.ts と同じ変換ロジック
            const frames = result.frames.map((frameData: { weights: number[] }) => {
              const frame: { [key: string]: number } = {};
              result.names.forEach((name: string, index: number) => {
                frame[name] = frameData.weights[index];
              });
              return frame;
            });

            const frameRate = result.frame_rate || 30;
            lamController.queueExpressionFrames(frames, frameRate);
            console.log(`[Concierge] Queued ${frames.length} frames to LAMAvatar at ${frameRate}fps`);
          } else {
            console.warn('[Concierge] LAMAvatar queueExpressionFrames not available');
          }
        }
      }
    } catch (error) {
      console.warn('[Concierge] audio2exp API error:', error);
    }
  }

  /**
   * 表情フレームを取得して一時保存（バッファに直接入れない）
   * 並行処理時に使用：最初のセンテンス再生中に次の表情を先行取得
   */
  private async fetchExpressionFrames(audioBase64: string): Promise<void> {
    this.pendingExpressionFrames = null;
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.EXPRESSION_API_TIMEOUT_MS);

      const response = await fetch(`${this.audio2expApiUrl}/api/audio2expression`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          audio_base64: audioBase64,
          session_id: this.sessionId,
          is_start: false,
          is_final: true,
          audio_format: 'mp3'
        }),
        signal: controller.signal
      });
      clearTimeout(timeoutId);

      if (response.ok) {
        const result = await response.json();
        const frameCount = result.frames?.length || 0;
        console.log(`[Concierge] Pre-fetched expression: ${frameCount} frames`);

        if (frameCount > 0 && result.names && result.frames) {
          const frames = result.frames.map((frameData: { weights: number[] }) => {
            const frame: { [key: string]: number } = {};
            result.names.forEach((name: string, index: number) => {
              frame[name] = frameData.weights[index];
            });
            return frame;
          });
          this.pendingExpressionFrames = { frames, frameRate: result.frame_rate || 30 };
        }
      }
    } catch (error) {
      console.warn('[Concierge] fetchExpressionFrames error:', error);
    }
  }

  // アバターアニメーション停止
  private stopAvatarAnimation() {
    if (this.els.avatarContainer) {
      this.els.avatarContainer.classList.remove('speaking');
    }
    // ※ LAMAvatar の状態は ttsPlayer イベント（ended/pause）で管理
  }

  // ★ LAMAvatar を audio2exp-service WebSocket に接続
  private connectLAMAvatarWebSocket() {
    if (!this.sessionId) return;

    const lamController = (window as any).lamAvatarController;
    if (lamController && typeof lamController.connectWebSocket === 'function') {
      const wsUrl = `${this.audio2expWsUrl}/${this.sessionId}`;
      lamController.connectWebSocket(wsUrl)
        .then(() => console.log('[Concierge] LAMAvatar WebSocket connected:', wsUrl))
        .catch((e: any) => console.warn('[Concierge] LAMAvatar WebSocket connection failed:', e));
    }
  }

  // ========================================
  // 🎯 UI言語更新をオーバーライド(挨拶文をコンシェルジュ用に)
  // ========================================
  protected updateUILanguage() {
    // ✅ バックエンドからの長期記憶対応済み挨拶を保持
    const initialMessage = this.els.chatArea.querySelector('.message.assistant[data-initial="true"] .message-text');
    const savedGreeting = initialMessage?.textContent;

    // 親クラスのupdateUILanguageを実行（UIラベル等を更新）
    super.updateUILanguage();

    // ✅ 長期記憶対応済み挨拶を復元（親が上書きしたものを戻す）
    if (initialMessage && savedGreeting) {
      initialMessage.textContent = savedGreeting;
    }

    // ✅ ページタイトルをコンシェルジュ用に設定
    const pageTitle = document.getElementById('pageTitle');
    if (pageTitle) {
      pageTitle.innerHTML = `<img src="/pwa-152x152.png" alt="Logo" class="app-logo" /> ${this.t('pageTitleConcierge')}`;
    }
  }

  // モード切り替え処理 - ページ遷移
  private toggleMode() {
    const isChecked = this.els.modeSwitch?.checked;
    if (!isChecked) {
      // チャットモードへページ遷移
      console.log('[ConciergeController] Switching to Chat mode...');
      window.location.href = '/';
    }
    // コンシェルジュモードは既に現在のページなので何もしない
  }

  // すべての活動を停止(アバターアニメーションも含む)
  protected stopAllActivities() {
    super.stopAllActivities();
    this.stopAvatarAnimation();
  }

  // ========================================
  // 🎯 並行処理フロー: 応答を分割してTTS処理
  // ========================================

  /**
   * センテンス単位でテキストを分割
   * 日本語: 。で分割
   * 英語・韓国語: . で分割
   * 中国語: 。で分割
   */
  private splitIntoSentences(text: string, language: string): string[] {
    let separator: RegExp;

    if (language === 'ja' || language === 'zh') {
      // 日本語・中国語: 。で分割
      separator = /。/;
    } else {
      // 英語・韓国語: . で分割
      separator = /\.\s+/;
    }

    const sentences = text.split(separator).filter(s => s.trim().length > 0);

    // 分割したセンテンスに句点を戻す
    return sentences.map((s, idx) => {
      if (idx < sentences.length - 1 || text.endsWith('。') || text.endsWith('. ')) {
        return language === 'ja' || language === 'zh' ? s + '。' : s + '. ';
      }
      return s;
    });
  }

  /**
   * 応答を分割して並行処理でTTS生成・再生
   * チャットモードのお店紹介フローを参考に実装
   */
  private async speakResponseInChunks(response: string, isTextInput: boolean = false) {
    // テキスト入力またはTTS無効の場合は従来通り
    if (isTextInput || !this.isTTSEnabled) {
      return this.speakTextGCP(response, true, false, isTextInput);
    }

    try {
      this.isAISpeaking = true;
      if (this.isRecording) {
        this.stopStreamingSTT();
      }

      // センテンス分割
      const sentences = this.splitIntoSentences(response, this.currentLanguage);

      // 1センテンスしかない場合は従来通り
      if (sentences.length <= 1) {
        await this.speakTextGCP(response, true, false, isTextInput);
        this.isAISpeaking = false;
        return;
      }

      // 最初のセンテンスと残りのセンテンスに分割
      const firstSentence = sentences[0];
      const remainingSentences = sentences.slice(1).join('');

      const langConfig = this.LANGUAGE_CODE_MAP[this.currentLanguage];

      // ★並行処理: TTS生成と表情生成を同時に実行して遅延を最小化
      if (this.isUserInteracted) {
        const cleanFirst = this.stripMarkdown(firstSentence);
        const cleanRemaining = remainingSentences.trim().length > 0
          ? this.stripMarkdown(remainingSentences) : null;

        // ★ 4つのAPIコールを可能な限り並行で開始
        // 1. 最初のセンテンスTTS
        const firstTtsPromise = fetch(`${this.apiBase}/api/tts/synthesize`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            text: cleanFirst, language_code: langConfig.tts,
            voice_name: langConfig.voice, session_id: this.sessionId
          })
        }).then(r => r.json());

        // 2. 残りのセンテンスTTS（あれば）
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

        // ★ 最初のTTSが返ったら、即座にその音声で表情生成を開始
        const firstTtsResult = await firstTtsPromise;
        if (firstTtsResult.success && firstTtsResult.audio) {
          // TTS音声を使って表情生成を開始（タイムアウト時はリップシンクなしで続行）
          const firstExpPromise = this.sendAudioToExpression(firstTtsResult.audio, true, !cleanRemaining);
          await Promise.race([
            firstExpPromise,
            new Promise<void>(resolve => setTimeout(resolve, this.EXPRESSION_API_TIMEOUT_MS))
          ]);

          this.lastAISpeech = this.normalizeText(cleanFirst);
          this.stopCurrentAudio();
          this.ttsPlayer.src = `data:audio/mp3;base64,${firstTtsResult.audio}`;

          // ★ 最初のセンテンス再生中に、残りのセンテンスの表情生成を並行実行
          let remainingExpPromise: Promise<void> | null = null;
          let remainingTtsResult: any = null;

          if (remainingTtsPromise) {
            remainingTtsResult = await remainingTtsPromise;
            if (remainingTtsResult?.success && remainingTtsResult?.audio) {
              // 最初のセンテンス再生中にバックグラウンドで表情生成
              // ★ ただしフレームバッファは最初のセンテンス再生後にクリアするため、
              //    結果を一時保存して再生直前にキューに入れる
              remainingExpPromise = this.fetchExpressionFrames(remainingTtsResult.audio);
            }
          }

          // 最初のセンテンス再生
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

          // ★ 残りのセンテンスを続けて再生
          if (remainingTtsResult?.success && remainingTtsResult?.audio) {
            this.lastAISpeech = this.normalizeText(cleanRemaining || '');

            // 表情フレームの取得完了を待つ（既に並行実行中なので高速）
            if (remainingExpPromise) await remainingExpPromise;

            // フレームバッファを入れ替え
            const lamController = (window as any).lamAvatarController;
            if (lamController && typeof lamController.clearFrameBuffer === 'function') {
              lamController.clearFrameBuffer();
            }
            if (this.pendingExpressionFrames) {
              lamController?.queueExpressionFrames?.(this.pendingExpressionFrames.frames, this.pendingExpressionFrames.frameRate);
              this.pendingExpressionFrames = null;
            }

            this.stopCurrentAudio();
            this.ttsPlayer.src = `data:audio/mp3;base64,${remainingTtsResult.audio}`;

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
    } catch (error) {
      console.error('[TTS並行処理エラー]', error);
      this.isAISpeaking = false;
      // エラー時はフォールバック
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

    // オウム返し判定(エコーバック防止)
    const normTranscript = this.normalizeText(transcript);
    if (this.isSemanticEcho(normTranscript, this.lastAISpeech)) {
        this.els.voiceStatus.innerHTML = this.t('voiceStatusStopped');
        this.els.voiceStatus.className = 'voice-status stopped';
        this.lastAISpeech = '';
        return;
    }

    this.els.userInput.value = transcript;
    this.addMessage('user', transcript);
    
    // 短すぎる入力チェック
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

    // ✅ 修正: 即答を「はい」だけに簡略化
    const ackText = this.t('ackYes'); // 「はい」のみ
    const preGeneratedAudio = this.preGeneratedAcks.get(ackText);
    
    // 即答を再生
    let firstAckPromise: Promise<void> | null = null;
    if (preGeneratedAudio && this.isTTSEnabled && this.isUserInteracted) {
      firstAckPromise = new Promise<void>((resolve) => {
        this.lastAISpeech = this.normalizeText(ackText);
        this.ttsPlayer.src = `data:audio/mp3;base64,${preGeneratedAudio}`;
        this.ttsPlayer.onended = () => resolve();
        this.ttsPlayer.play().catch(_e => resolve());
      });
    } else if (this.isTTSEnabled) { 
      firstAckPromise = this.speakTextGCP(ackText, false); 
    }
    
    this.addMessage('assistant', ackText);
    
    // ✅ 修正: オウム返しパターンを削除し、すぐにLLMへ送信
    (async () => {
      try {
        if (firstAckPromise) await firstAckPromise;
        
        // すぐにsendMessage()を実行
        if (this.els.userInput.value.trim()) {
          this.isFromVoiceInput = true;
          this.sendMessage();
        }
      } catch (_error) {
        if (this.els.userInput.value.trim()) {
          this.isFromVoiceInput = true;
          this.sendMessage();
        }
      }
    })();
    
    this.els.voiceStatus.innerHTML = this.t('voiceStatusStopped');
    this.els.voiceStatus.className = 'voice-status stopped';
  }

  // ========================================
  // 🎯 コンシェルジュモード専用: メッセージ送信処理
  // ========================================
  protected async sendMessage() {
    let firstAckPromise: Promise<void> | null = null; 
    this.unlockAudioParams();
    const message = this.els.userInput.value.trim();
    if (!message || this.isProcessing) return;
    
    const currentSessionId = this.sessionId;
    const isTextInput = !this.isFromVoiceInput;
    
    this.isProcessing = true; 
    this.els.sendBtn.disabled = true;
    this.els.micBtn.disabled = true; 
    this.els.userInput.disabled = true;

    // ✅ テキスト入力時も「はい」だけに簡略化
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
      
      // ✅ 修正: 即答を「はい」だけに
      const ackText = this.t('ackYes');
      this.currentAISpeech = ackText;
      this.addMessage('assistant', ackText);
      
      if (this.isTTSEnabled && !isTextInput) {
        try {
          const preGeneratedAudio = this.preGeneratedAcks.get(ackText);
          if (preGeneratedAudio && this.isUserInteracted) {
            firstAckPromise = new Promise<void>((resolve) => {
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
      
      // ✅ 修正: オウム返しパターンを削除
      // (generateFallbackResponse, additionalResponse の呼び出しを削除)
    }

    this.isFromVoiceInput = false;
    
    // ✅ 待機アニメーションは6.5秒後に表示(LLM送信直前にタイマースタート)
    if (this.waitOverlayTimer) clearTimeout(this.waitOverlayTimer);
    let responseReceived = false;
    
    // タイマーセットをtry直前に移動(即答処理の後)
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
      
      // ✅ レスポンス到着フラグを立てる
      responseReceived = true;
      
      if (this.sessionId !== currentSessionId) return;
      
      // ✅ タイマーをクリアしてアニメーションを非表示
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
                  this.lastAISpeech = this.normalizeText(introText);
                  this.ttsPlayer.src = `data:audio/mp3;base64,${preGeneratedIntro}`;
                  this.ttsPlayer.onended = () => resolve();
                  this.ttsPlayer.play();
                });
              } else { 
                introPart2Promise = this.speakTextGCP(introText, false, false, isTextInput); 
              }
            }

            let firstShopAudioPromise: Promise<string | null> | null = null;
            let remainingAudioPromise: Promise<string | null> | null = null;
            let firstShopAudioBase64: string | null = null;
            let restShopAudioBase64: string | null = null;
            const shopLangConfig = this.LANGUAGE_CODE_MAP[this.currentLanguage];

            if (shopLines.length > 0 && this.isTTSEnabled && this.isUserInteracted && !isTextInput) {
              const firstShop = shopLines[0];
              const restShops = shopLines.slice(1).join('\n\n');
              firstShopAudioPromise = (async () => {
                const cleanText = this.stripMarkdown(firstShop);
                const response = await fetch(`${this.apiBase}/api/tts/synthesize`, {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({
                    text: cleanText, language_code: shopLangConfig.tts, voice_name: shopLangConfig.voice,
                    session_id: this.sessionId
                  })
                });
                const result = await response.json();
                if (result.success && result.audio) {
                  firstShopAudioBase64 = result.audio;
                  return `data:audio/mp3;base64,${result.audio}`;
                }
                return null;
              })();

              if (restShops) {
                remainingAudioPromise = (async () => {
                  const cleanText = this.stripMarkdown(restShops);
                  const response = await fetch(`${this.apiBase}/api/tts/synthesize`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                      text: cleanText, language_code: shopLangConfig.tts, voice_name: shopLangConfig.voice,
                      session_id: this.sessionId
                    })
                  });
                  const result = await response.json();
                  if (result.success && result.audio) {
                    restShopAudioBase64 = result.audio;
                    return `data:audio/mp3;base64,${result.audio}`;
                  }
                  return null;
                })();
              }
            }

            if (introPart2Promise) await introPart2Promise;
            
            if (firstShopAudioPromise) {
              const firstShopAudio = await firstShopAudioPromise;
              if (firstShopAudio && firstShopAudioBase64) {
                const firstShopText = this.stripMarkdown(shopLines[0]);
                this.lastAISpeech = this.normalizeText(firstShopText);
                const restShops = shopLines.slice(1).join('\n\n');

                // ★ リップシンク: 表情データを取得（タイムアウト時はリップシンクなしで続行）
                await Promise.race([
                  this.sendAudioToExpression(firstShopAudioBase64, true, !restShops),
                  new Promise<void>(resolve => setTimeout(resolve, this.EXPRESSION_API_TIMEOUT_MS))
                ]);

                if (!isTextInput && this.isTTSEnabled) {
                  this.stopCurrentAudio();
                }

                // ★ 最初のショップ再生中に残りの表情を先行取得
                let remainingExpPromise: Promise<void> | null = null;
                if (restShopAudioBase64) {
                  remainingExpPromise = this.fetchExpressionFrames(restShopAudioBase64);
                }

                this.ttsPlayer.src = firstShopAudio;
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

                if (remainingAudioPromise) {
                  const remainingAudio = await remainingAudioPromise;
                  if (remainingAudio && restShopAudioBase64) {
                    const restShopsText = this.stripMarkdown(shopLines.slice(1).join('\n\n'));
                    this.lastAISpeech = this.normalizeText(restShopsText);

                    // ★ 先行取得した表情フレームをキューに入れる（待ち時間ほぼゼロ）
                    if (remainingExpPromise) await remainingExpPromise;
                    const lamController = (window as any).lamAvatarController;
                    if (lamController && typeof lamController.clearFrameBuffer === 'function') {
                      lamController.clearFrameBuffer();
                    }
                    if (this.pendingExpressionFrames) {
                      lamController?.queueExpressionFrames?.(this.pendingExpressionFrames.frames, this.pendingExpressionFrames.frameRate);
                      this.pendingExpressionFrames = null;
                    }

                    if (!isTextInput && this.isTTSEnabled) {
                      this.stopCurrentAudio();
                    }

                    this.ttsPlayer.src = remainingAudio;
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
            }
            this.isAISpeaking = false;
          } catch (_e) { this.isAISpeaking = false; }
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
            // ★並行処理フローを適用
            this.speakResponseInChunks(data.response, isTextInput);
          } else {
            // ★並行処理フローを適用
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
