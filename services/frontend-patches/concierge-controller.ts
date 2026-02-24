

// src/scripts/chat/concierge-controller.ts
import { CoreController } from './core-controller';
import { AudioManager } from './audio-manager';

declare const io: any;

export class ConciergeController extends CoreController {
  // Audio2Expression はバックエンドTTSエンドポイント経由で統合済み
  private pendingAckPromise: Promise<void> | null = null;

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
    // LAMAvatar が後から初期化される可能性があるため、即時 + 遅延リトライでリンク
    let linked = false;
    let linkAttempts = 0;
    const linkTtsPlayer = () => {
      if (linked) return true;
      linkAttempts++;
      const lam = (window as any).lamAvatarController;
      if (lam && typeof lam.setExternalTtsPlayer === 'function') {
        lam.setExternalTtsPlayer(this.ttsPlayer);
        linked = true;
        console.log(`[Concierge] TTS player linked with LAMAvatar (attempt #${linkAttempts})`);
        return true;
      }
      console.log(`[Concierge] LAMAvatar not ready yet (attempt #${linkAttempts})`);
      return false;
    };
    if (!linkTtsPlayer()) {
      // 遅延リトライ: 500ms, 1000ms, 2000ms, 4000ms
      const retryDelays = [500, 1000, 2000, 4000];
      retryDelays.forEach((delay) => {
        setTimeout(() => linkTtsPlayer(), delay);
      });
    }

    // ★ 診断用: ブラウザコンソールから __testLipSync() で呼び出し可能
    (window as any).__testLipSync = () => this.runLipSyncDiagnostic();
  }

  /**
   * レンダラー診断テスト
   * ブラウザコンソールから __testLipSync() で実行
   *
   * 日本語5母音（あいうえお）の既知blendshapeパターンを
   * 無音音声と同期再生し、レンダラーが52次元データを正しく描画できるか判定する
   *
   * 判定基準:
   *   - あ: 口が大きく開く (jawOpen高)
   *   - い: 口角が横に広がる (mouthSmile高)
   *   - う: 口がすぼまる (mouthFunnel/Pucker高)
   *   - え: 口が横に広がり中程度に開く (mouthStretch高)
   *   - お: 口が丸くなる (mouthFunnel高 + jawOpen中)
   *
   * 結果:
   *   ✓ 5母音で明らかに異なる口形状 → レンダラーは52次元対応
   *   ✗ jawの開閉しか見えない → レンダラーはjawOpen単次元のみ
   */
  private runLipSyncDiagnostic(): void {
    const lam = (window as any).lamAvatarController;
    if (!lam) {
      console.error('[DIAG] lamAvatarController not found');
      return;
    }

    // 日本語5母音のARKitブレンドシェイプパターン
    const base: { [k: string]: number } = {};  // 全て0で初期化
    const vowelPatterns: { [vowel: string]: { [k: string]: number } } = {
      'あ(a)': { jawOpen: 0.7, mouthLowerDownLeft: 0.5, mouthLowerDownRight: 0.5, mouthUpperUpLeft: 0.2, mouthUpperUpRight: 0.2 },
      'い(i)': { jawOpen: 0.2, mouthSmileLeft: 0.6, mouthSmileRight: 0.6, mouthStretchLeft: 0.4, mouthStretchRight: 0.4 },
      'う(u)': { jawOpen: 0.15, mouthFunnel: 0.6, mouthPucker: 0.5 },
      'え(e)': { jawOpen: 0.4, mouthStretchLeft: 0.5, mouthStretchRight: 0.5, mouthSmileLeft: 0.3, mouthSmileRight: 0.3, mouthLowerDownLeft: 0.3, mouthLowerDownRight: 0.3 },
      'お(o)': { jawOpen: 0.5, mouthFunnel: 0.5, mouthPucker: 0.3, mouthLowerDownLeft: 0.2, mouthLowerDownRight: 0.2 },
    };

    // フレーム生成: neutral(15) → 各母音(20frames=0.67s) → neutral(15)
    const frameRate = 30;
    const frames: { [k: string]: number }[] = [];
    const addFrames = (pattern: { [k: string]: number }, count: number, label?: string) => {
      for (let i = 0; i < count; i++) {
        frames.push({ ...base, ...pattern });
      }
      if (label) console.log(`[DIAG] ${label}: frames ${frames.length - count}-${frames.length - 1}`);
    };

    addFrames(base, 15, 'neutral (start)');
    for (const [vowel, pattern] of Object.entries(vowelPatterns)) {
      addFrames(pattern, 20, vowel);
    }
    addFrames(base, 15, 'neutral (end)');

    const totalFrames = frames.length;
    const durationSec = totalFrames / frameRate + 0.5;

    // 無音WAVを生成（ttsPlayer経由で再生して同期トリガー）
    const sampleRate = 8000;
    const numSamples = Math.floor(durationSec * sampleRate);
    const wavBuf = new ArrayBuffer(44 + numSamples * 2);
    const dv = new DataView(wavBuf);
    const ws = (off: number, s: string) => { for (let i = 0; i < s.length; i++) dv.setUint8(off + i, s.charCodeAt(i)); };
    ws(0, 'RIFF');
    dv.setUint32(4, 36 + numSamples * 2, true);
    ws(8, 'WAVE'); ws(12, 'fmt ');
    dv.setUint32(16, 16, true);
    dv.setUint16(20, 1, true); dv.setUint16(22, 1, true);
    dv.setUint32(24, sampleRate, true); dv.setUint32(28, sampleRate * 2, true);
    dv.setUint16(32, 2, true); dv.setUint16(34, 16, true);
    ws(36, 'data');
    dv.setUint32(40, numSamples * 2, true);

    const wavUrl = URL.createObjectURL(new Blob([wavBuf], { type: 'audio/wav' }));

    // LAMAvatarにフレーム投入 + 再生
    lam.clearFrameBuffer();
    lam.queueExpressionFrames(frames, frameRate);

    this.ttsPlayer.src = wavUrl;
    this.ttsPlayer.play().then(() => {
      console.log(`[DIAG] ▶ Playing: ${totalFrames} frames, ${durationSec.toFixed(1)}s`);
      console.log('[DIAG] 0.5s neutral → 0.67s あ → 0.67s い → 0.67s う → 0.67s え → 0.67s お → 0.5s neutral');
      console.log('[DIAG] ✓ 5母音で口形状が変われば → レンダラーは52次元blendshape対応');
      console.log('[DIAG] ✗ jawの開閉のみ → レンダラーはjawOpen単次元');
    }).catch((e: any) => {
      console.error('[DIAG] Play failed:', e);
      console.log('[DIAG] ユーザー操作後に再試行してください（autoplay制限）');
    });
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

      // リップシンク: バックエンドTTSエンドポイント経由で表情データ取得（追加接続不要）

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
        // ★ TTS応答に同梱されたExpressionを即バッファ投入（遅延ゼロ）
        if (data.expression) {
          this.applyExpressionFromTts(data.expression);
        } else {
          console.warn(`[Concierge] TTS response has NO expression data (session=${this.sessionId})`);
        }
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

  // ★ 口周りblendshapeのスケール係数
  //
  // 【重要】__testLipSync() 診断で判明:
  // SDK (gaussian-splat-renderer-for-lam) が jawOpen と mouthLowerDown の2値しか
  // FLAME メッシュ変形に使わない。LAMAvatar.astro は全52次元を返しているが SDK が無視。
  // → LAMAvatar.astro 側で remapForSdkLimitation() を追加し、
  //   SDK全52ch対応判明: remapForSdkLimitation 廃止、全チャンネル直接パス。
  //   MOUTH_AMPLIFY は A2E 出力の調整のみ（SDKは全ch適用する）。
  //
  // A2Eモデルの出力特性: jawOpen弱(avg~0.05), lowerDown強(raw~0.84)
  // SDK は全52チャンネルを morphTargetDictionary 経由で boneTexture に書き込み、
  // シェーダーが for(i < bsCount) で全チャンネル適用する。
  // → 増幅は最小限にし、A2E の自然な出力を活かす
  // ※全値は BLENDSHAPE_SAFE_MAX(0.7) でクランプ（FLAME LBS 数値安定のため）
  //
  // 母音分化 + 振幅チューニング:
  //   動的範囲圧縮（双方向）がピークを抑え、谷を引き上げるため、
  //   amplifyを高めに設定しても安全。圧縮なしだとピーク爆発するが、
  //   圧縮ありで平均方向に引き寄せられる。
  //
  //   日本語母音の視認性: jaw+lowerDown で口の開き、
  //   smile/funnel/pucker/stretch で形状分化
  private static readonly MOUTH_AMPLIFY: { [key: string]: number } = {
    // --- 主要チャンネル（口の開き） ---
    'jawOpen': 2.0,                // 強ブースト: raw avg~0.06→eff~0.12。圧縮でピーク抑制
    'mouthLowerDownLeft': 0.5,     // 中抑制: raw~0.84→eff~0.42。圧縮でピーク→0.30程度
    'mouthLowerDownRight': 0.5,    // 中抑制: 同上
    // --- 母音チャンネル: 日本語5母音の形状分化 ---
    'mouthFunnel': 2.5,            // 強ブースト: う・お の唇突き出し
    'mouthPucker': 1.5,            // 軽ブースト: う のすぼめ
    'mouthSmileLeft': 3.5,         // 強ブースト: い の口角引き
    'mouthSmileRight': 3.5,        // 強ブースト: い の口角引き
    'mouthStretchLeft': 2.0,       // ブースト: え の口横伸ばし
    'mouthStretchRight': 2.0,      // ブースト: え の口横伸ばし
    // --- 補助チャンネル ---
    'mouthClose': 1.0,
    'mouthUpperUpLeft': 1.0,
    'mouthUpperUpRight': 1.0,
    'mouthDimpleLeft': 1.0,
    'mouthDimpleRight': 1.0,
    'mouthRollLower': 1.0,
    'mouthRollUpper': 1.0,
    'mouthShrugLower': 1.0,
    'mouthShrugUpper': 1.0,
  };

  // FLAME LBS の安全範囲。これを超えるとメッシュが破綻（数値爆発）する
  private static readonly BLENDSHAPE_SAFE_MAX = 0.7;

  /**
   * TTS応答に同梱されたExpressionデータをバッファに即投入（遅延ゼロ）
   * 同期方式: バックエンドがTTS+audio2expを同期実行し、結果を同梱して返す
   *
   * ★品質改善:
   * 1. 口周りblendshapeの増幅 → 日本語母音の可視性向上
   * 2. フレーム補間 (30fps→60fps) → レンダラーの60fps描画に滑らかに追従
   * 3. 診断ログ → jawOpen/mouthFunnel等の統計値で品質を確認可能
   */
  private applyExpressionFromTts(expression: any): void {
    const lamController = (window as any).lamAvatarController;
    if (!lamController) {
      console.warn('[Concierge] lamAvatarController not found - expression data dropped');
      return;
    }

    // 新セグメント開始時は必ずバッファクリア（前セグメントのフレーム混入防止）
    if (typeof lamController.clearFrameBuffer === 'function') {
      lamController.clearFrameBuffer();
    }

    if (expression?.names && expression?.frames?.length > 0) {
      const srcFrameRate = expression.frame_rate || 30;

      // Step 1: バックエンド形式 → LAMAvatar形式に変換 + blendshape増幅
      // ★ 新旧両フォーマット対応:
      //   旧 (FastAPI): frames = [{"weights": [0.1, ...]}, ...]
      //   新 (Flask):   frames = [[0.1, ...], ...]
      const rawFrames = expression.frames.map((f: any) => {
        const frame: { [key: string]: number } = {};
        // フレームがArrayなら直接使用、objectなら.weightsから取得
        const values: number[] = Array.isArray(f) ? f : (f.weights || []);
        expression.names.forEach((name: string, i: number) => {
          let val = values[i] || 0;
          // 口周りblendshapeを増幅（日本語母音の可視性向上）
          const amp = ConciergeController.MOUTH_AMPLIFY[name];
          if (amp) {
            // FLAME LBS 安全範囲でクランプ（>0.7 で数値不安定→メッシュ破綻）
            val = Math.min(ConciergeController.BLENDSHAPE_SAFE_MAX, val * amp);
          }
          frame[name] = val;
        });
        return frame;
      });

      // Step 2: フレーム補間 (30fps → 60fps) — 線形補間で滑らかに
      const interpolatedFrames: { [key: string]: number }[] = [];
      for (let i = 0; i < rawFrames.length; i++) {
        interpolatedFrames.push(rawFrames[i]);
        if (i < rawFrames.length - 1) {
          const curr = rawFrames[i];
          const next = rawFrames[i + 1];
          const mid: { [key: string]: number } = {};
          for (const key of Object.keys(curr)) {
            mid[key] = (curr[key] + next[key]) * 0.5;
          }
          interpolatedFrames.push(mid);
        }
      }
      const outputFrameRate = srcFrameRate * 2; // 30→60fps

      // Step 2.3: 双方向動的範囲圧縮（振幅ムラを均一化）
      // A2E出力の乱高下を平準化:
      //   弱フレーム → 平均方向に引き上げ（ごにょごにょ防止）
      //   強フレーム → 平均方向に引き下げ（ピーク抑制）
      // → amplifyを高めに設定しても安全。全体的に均一な口の動きになる
      const mouthKeys = Object.keys(ConciergeController.MOUTH_AMPLIFY);
      const compressionFactor = 0.5; // 50%を平均方向に寄せる（強めの圧縮）
      for (const key of mouthKeys) {
        const values = interpolatedFrames.map(f => f[key] || 0);
        const activeValues = values.filter(v => v > 0.01);
        if (activeValues.length < 3) continue;
        const mean = activeValues.reduce((a, b) => a + b) / activeValues.length;
        for (const frame of interpolatedFrames) {
          if (frame[key] !== undefined && frame[key] > 0.01) {
            // 双方向: 弱も強も平均方向に寄せる
            frame[key] = frame[key] * (1 - compressionFactor) + mean * compressionFactor;
          }
        }
      }

      // Step 2.5: 非対称EMAスムージング
      // 口の開き（attack）は素早く応答、閉じ（decay）はゆっくり減衰。
      // → 短い"死区間"（A2Eが0に落ちる瞬間）で口が急に閉じるのを防ぐ
      const attackAlpha = 0.82; // 開方向: 素早い応答（音素遷移を維持）
      const decayAlpha = 0.45;  // 閉方向: ゆるやかな減衰（急な閉口を防止）
      for (let i = 1; i < interpolatedFrames.length; i++) {
        const prev = interpolatedFrames[i - 1];
        const curr = interpolatedFrames[i];
        for (const key of mouthKeys) {
          if (curr[key] !== undefined && prev[key] !== undefined) {
            const alpha = curr[key] >= prev[key] ? attackAlpha : decayAlpha;
            curr[key] = prev[key] * (1 - alpha) + curr[key] * alpha;
          }
        }
      }

      // Step 3: LAMAvatarにキュー投入
      lamController.queueExpressionFrames(interpolatedFrames, outputFrameRate);

      // Step 4: 診断ログ（全母音チャンネルの統計値）
      const stat = (key: string) => {
        const vals = rawFrames.map((f: { [k: string]: number }) => f[key] || 0);
        return { max: Math.max(...vals), avg: vals.reduce((a: number, b: number) => a + b, 0) / vals.length };
      };
      const jaw = stat('jawOpen');
      const funnel = stat('mouthFunnel');
      const pucker = stat('mouthPucker');
      const smile = stat('mouthSmileLeft');
      const stretch = stat('mouthStretchLeft');
      const lowerDown = stat('mouthLowerDownLeft');
      console.log(
        `[Concierge] Expression: ${rawFrames.length}→${interpolatedFrames.length} frames (${srcFrameRate}→${outputFrameRate}fps)\n` +
        `  jaw: max=${jaw.max.toFixed(3)} avg=${jaw.avg.toFixed(3)} | lowerDown: max=${lowerDown.max.toFixed(3)}\n` +
        `  funnel: max=${funnel.max.toFixed(3)} | pucker: max=${pucker.max.toFixed(3)} | smile: max=${smile.max.toFixed(3)} | stretch: max=${stretch.max.toFixed(3)}`
      );
    } else {
      console.warn(`[Concierge] No expression frames in TTS response (names=${!!expression?.names}, frames=${expression?.frames?.length || 0})`);
    }
  }

  // アバターアニメーション停止
  private stopAvatarAnimation() {
    if (this.els.avatarContainer) {
      this.els.avatarContainer.classList.remove('speaking');
    }
    // ※ LAMAvatar の状態は ttsPlayer イベント（ended/pause）で管理
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
    // TTS無効の場合はスキップ（テキスト入力でもコンシェルジュモードではTTS再生する）
    if (!this.isTTSEnabled) {
      return;
    }

    try {
      // ★ ack再生中ならttsPlayer解放を待つ（並行処理の同期ポイント）
      if (this.pendingAckPromise) {
        await this.pendingAckPromise;
        this.pendingAckPromise = null;
      }
      this.stopCurrentAudio(); // ttsPlayer確実解放

      this.isAISpeaking = true;
      if (this.isRecording) {
        this.stopStreamingSTT();
      }

      // センテンス分割
      const sentences = this.splitIntoSentences(response, this.currentLanguage);

      // 1センテンスしかない場合は従来通り（skipAudio=false: コンシェルジュでは常に再生）
      if (sentences.length <= 1) {
        await this.speakTextGCP(response, true, false, false);
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

        // ★ 最初のTTSが返ったら即再生（Expression同梱済み）
        const firstTtsResult = await firstTtsPromise;
        if (firstTtsResult.success && firstTtsResult.audio) {
          // ★ TTS応答に同梱されたExpressionを即バッファ投入（遅延ゼロ）
          if (firstTtsResult.expression) this.applyExpressionFromTts(firstTtsResult.expression);

          this.lastAISpeech = this.normalizeText(cleanFirst);
          this.stopCurrentAudio();
          this.ttsPlayer.src = `data:audio/mp3;base64,${firstTtsResult.audio}`;

          // 残りのTTS結果を先に取得（TTS応答にExpression同梱済み）
          let remainingTtsResult: any = null;
          if (remainingTtsPromise) {
            remainingTtsResult = await remainingTtsPromise;
          }

          // 最初のセンテンス再生
          await new Promise<void>((resolve) => {
            this.ttsPlayer.onended = () => {
              this.els.voiceStatus.innerHTML = this.t('voiceStatusStopped');
              this.els.voiceStatus.className = 'voice-status stopped';
              resolve();
            };
            this.ttsPlayer.onerror = () => {
              console.error('[TTS] First sentence play error');
              resolve();
            };
            this.els.voiceStatus.innerHTML = this.t('voiceStatusSpeaking');
            this.els.voiceStatus.className = 'voice-status speaking';
            this.ttsPlayer.play().catch((e: any) => {
              console.error('[TTS] First sentence play() rejected:', e);
              resolve();
            });
          });

          // ★ 残りのセンテンスを続けて再生（Expression同梱済み）
          if (remainingTtsResult?.success && remainingTtsResult?.audio) {
            this.lastAISpeech = this.normalizeText(cleanRemaining || '');

            // ★ TTS応答に同梱されたExpressionを即バッファ投入
            if (remainingTtsResult.expression) this.applyExpressionFromTts(remainingTtsResult.expression);

            this.stopCurrentAudio();
            this.ttsPlayer.src = `data:audio/mp3;base64,${remainingTtsResult.audio}`;

            await new Promise<void>((resolve) => {
              this.ttsPlayer.onended = () => {
                this.els.voiceStatus.innerHTML = this.t('voiceStatusStopped');
                this.els.voiceStatus.className = 'voice-status stopped';
                resolve();
              };
              this.ttsPlayer.onerror = () => {
                console.error('[TTS] Remaining sentence play error');
                resolve();
              };
              this.els.voiceStatus.innerHTML = this.t('voiceStatusSpeaking');
              this.els.voiceStatus.className = 'voice-status speaking';
              this.ttsPlayer.play().catch((e: any) => {
                console.error('[TTS] Remaining sentence play() rejected:', e);
                resolve();
              });
            });
          }
        }
      }

      this.isAISpeaking = false;
    } catch (error) {
      console.error('[TTS並行処理エラー]', error);
      this.isAISpeaking = false;
      // エラー時はフォールバック（skipAudio=false: コンシェルジュでは常に再生）
      await this.speakTextGCP(response, true, false, false);
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

    // 即答を再生（ttsPlayerで）
    if (preGeneratedAudio && this.isTTSEnabled && this.isUserInteracted) {
      this.pendingAckPromise = new Promise<void>((resolve) => {
        this.lastAISpeech = this.normalizeText(ackText);
        this.ttsPlayer.src = `data:audio/mp3;base64,${preGeneratedAudio}`;
        let resolved = false;
        const done = () => { if (!resolved) { resolved = true; resolve(); } };
        this.ttsPlayer.onended = done;
        this.ttsPlayer.onpause = done; // ★ pause時もresolve（src変更やstop時のデッドロック防止）
        this.ttsPlayer.play().catch(_e => done());
      });
    } else if (this.isTTSEnabled) {
      this.pendingAckPromise = this.speakTextGCP(ackText, false);
    }

    this.addMessage('assistant', ackText);

    // ★ 並行処理: ack再生完了を待たず、即LLMリクエスト開始（~700ms短縮）
    //   pendingAckPromiseはsendMessage内でTTS再生前にawaitされる
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
      
      if (this.isTTSEnabled) {
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
            // ★ ack再生中ならttsPlayer解放を待つ（並行処理の同期ポイント）
            if (this.pendingAckPromise) {
              await this.pendingAckPromise;
              this.pendingAckPromise = null;
            }
            this.stopCurrentAudio(); // ttsPlayer確実解放

            this.isAISpeaking = true;
            if (this.isRecording) { this.stopStreamingSTT(); }

            await this.speakTextGCP(this.t('ttsIntro'), true, false, false);
            
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
                introPart2Promise = this.speakTextGCP(introText, false, false, false);
              }
            }

            let firstShopTtsPromise: Promise<any> | null = null;
            let remainingShopTtsPromise: Promise<any> | null = null;
            const shopLangConfig = this.LANGUAGE_CODE_MAP[this.currentLanguage];

            if (shopLines.length > 0 && this.isTTSEnabled && this.isUserInteracted) {
              const firstShop = shopLines[0];
              const restShops = shopLines.slice(1).join('\n\n');

              // ★ 1行目先行: 最初のショップと残りのTTSを並行開始
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

                // ★ TTS応答に同梱されたExpressionを即バッファ投入
                if (firstResult.expression) this.applyExpressionFromTts(firstResult.expression);

                this.stopCurrentAudio();

                this.ttsPlayer.src = `data:audio/mp3;base64,${firstResult.audio}`;

                // 残りのTTS結果を先に取得（Expression同梱済み）
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
                  this.ttsPlayer.onerror = () => resolve();
                  this.els.voiceStatus.innerHTML = this.t('voiceStatusSpeaking');
                  this.els.voiceStatus.className = 'voice-status speaking';
                  this.ttsPlayer.play().catch(() => resolve());
                });

                if (remainingResult?.success && remainingResult?.audio) {
                    const restShopsText = this.stripMarkdown(shopLines.slice(1).join('\n\n'));
                    this.lastAISpeech = this.normalizeText(restShopsText);

                    // ★ TTS応答に同梱されたExpressionを即バッファ投入
                    if (remainingResult.expression) this.applyExpressionFromTts(remainingResult.expression);

                    this.stopCurrentAudio();

                    this.ttsPlayer.src = `data:audio/mp3;base64,${remainingResult.audio}`;
                    await new Promise<void>((resolve) => {
                      this.ttsPlayer.onended = () => {
                        this.els.voiceStatus.innerHTML = this.t('voiceStatusStopped');
                        this.els.voiceStatus.className = 'voice-status stopped';
                        resolve();
                      };
                      this.ttsPlayer.onerror = () => resolve();
                      this.els.voiceStatus.innerHTML = this.t('voiceStatusSpeaking');
                      this.els.voiceStatus.className = 'voice-status speaking';
                      this.ttsPlayer.play().catch(() => resolve());
                    });
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
