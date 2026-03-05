/**
 * Live API 用オーディオ I/O
 *
 * ★★★ 重要 ★★★
 * このファイルのマイク制御コード（getUserMedia, AudioContext, AudioWorklet）は
 * iPhone 16/17 の iOS 18-19 セキュリティ制限への対策コードである。
 * 既存の AudioManager (gourmet-sp) のパターンをそのまま踏襲すること。
 * Claude Code の知識ベースで書き換えると iPhone で動かなくなる（実証済み）。
 *
 * Live API 経路のオーディオフロー:
 *   [マイク] → getUserMedia → AudioWorklet (48kHz→16kHz) → PCM base64 → WebSocket送信
 *   [スピーカー] ← WebSocket受信 ← PCM 24kHz base64 → AudioBuffer → AudioContext.destination
 *
 * REST 経路のオーディオフロー (既存 AudioManager がそのまま担当):
 *   [マイク] → AudioManager → Socket.IO → STT
 *   [スピーカー] ← ttsPlayer (HTMLAudioElement) ← MP3 base64 ← TTS API
 */

import type { LiveWSClient } from './live-ws-client';

export interface AudioIOOptions {
  /** PCM 送信先の LiveWSClient */
  wsClient: LiveWSClient;
  /** マイク入力のサンプルレート (デフォルト: 16000) */
  sendSampleRate?: number;
  /** 受信音声のサンプルレート (デフォルト: 24000) */
  receiveSampleRate?: number;
  /** 送信チャンクサイズ (ms) (デフォルト: 100) */
  chunkDurationMs?: number;
}

/**
 * Live API 用のオーディオ入出力マネージャー
 *
 * ★ マイク制御（startMic/stopMic）の内部実装は
 *   gourmet-sp の AudioManager を移植して使用すること。
 *   以下のスタブ実装は構造を示すためのもので、
 *   iPhone 16/17 固有の対策コードは含まれていない。
 */
export class LiveAudioIO {
  private wsClient: LiveWSClient;
  private sendSampleRate: number;
  private receiveSampleRate: number;
  private chunkDurationMs: number;

  // マイク入力
  private audioContext: AudioContext | null = null;
  private mediaStream: MediaStream | null = null;
  private workletNode: AudioWorkletNode | null = null;
  private isMicActive = false;

  // 音声出力 (PCM 24kHz)
  private playbackContext: AudioContext | null = null;
  private playbackQueue: ArrayBuffer[] = [];
  private isPlaying = false;
  private nextPlayTime = 0;

  constructor(options: AudioIOOptions) {
    this.wsClient = options.wsClient;
    this.sendSampleRate = options.sendSampleRate ?? 16000;
    this.receiveSampleRate = options.receiveSampleRate ?? 24000;
    this.chunkDurationMs = options.chunkDurationMs ?? 100;
  }

  /**
   * マイクを開始し、PCM 16kHz を WebSocket 経由で送信する
   *
   * ★★★ iPhone 対策注意事項 ★★★
   * - getUserMedia はユーザーインタラクション（tap/click）後にのみ呼ぶこと
   * - AudioContext の resume() はユーザーインタラクション内で行うこと
   * - iOS の autoplay policy により、音声入出力は同一インタラクション起点が必要
   * - 既存 AudioManager の startRecording() パターンを必ず踏襲すること
   */
  async startMic(): Promise<void> {
    if (this.isMicActive) return;

    try {
      // AudioContext 初期化
      this.audioContext = new AudioContext({ sampleRate: 48000 });

      // マイク取得
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 48000,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      // AudioWorklet でダウンサンプリング (48kHz → 16kHz)
      await this.audioContext.audioWorklet.addModule(
        this.createDownsampleWorkletUrl()
      );

      this.workletNode = new AudioWorkletNode(
        this.audioContext,
        'downsample-processor',
        {
          processorOptions: {
            targetSampleRate: this.sendSampleRate,
            chunkDurationMs: this.chunkDurationMs,
          },
        }
      );

      // Worklet → WebSocket 送信
      this.workletNode.port.onmessage = (event) => {
        if (event.data.type === 'pcm-chunk') {
          const pcmData: Int16Array = event.data.data;
          const base64 = this.int16ArrayToBase64(pcmData);
          this.wsClient.sendAudio(base64);
        }
      };

      // マイクストリームを接続
      const source = this.audioContext.createMediaStreamSource(this.mediaStream);
      source.connect(this.workletNode);

      this.isMicActive = true;
      console.log('[LiveAudioIO] Mic started (48kHz → 16kHz)');
    } catch (e) {
      console.error('[LiveAudioIO] Failed to start mic:', e);
      this.stopMic();
      throw e;
    }
  }

  /**
   * マイクを停止
   */
  stopMic(): void {
    if (this.workletNode) {
      this.workletNode.disconnect();
      this.workletNode = null;
    }
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((t) => t.stop());
      this.mediaStream = null;
    }
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
    this.isMicActive = false;
    console.log('[LiveAudioIO] Mic stopped');
  }

  /**
   * PCM 24kHz 音声をキューに追加して再生
   * relay.py から受信した base64 PCM をデコードして再生する
   */
  queuePlayback(base64Pcm: string): void {
    const pcmBytes = this.base64ToArrayBuffer(base64Pcm);
    this.playbackQueue.push(pcmBytes);

    if (!this.isPlaying) {
      this.processPlaybackQueue();
    }
  }

  /**
   * 再生を停止（barge-in / 割り込み時）
   */
  stopPlayback(): void {
    this.playbackQueue = [];
    this.isPlaying = false;
    this.nextPlayTime = 0;
    // AudioContext は保持（次の再生で再利用）
    console.log('[LiveAudioIO] Playback stopped (barge-in)');
  }

  /**
   * 全リソース解放
   */
  destroy(): void {
    this.stopMic();
    this.stopPlayback();
    if (this.playbackContext) {
      this.playbackContext.close();
      this.playbackContext = null;
    }
  }

  get micActive(): boolean {
    return this.isMicActive;
  }

  // --- private ---

  private async processPlaybackQueue(): Promise<void> {
    if (!this.playbackContext) {
      this.playbackContext = new AudioContext({ sampleRate: this.receiveSampleRate });
    }

    this.isPlaying = true;

    while (this.playbackQueue.length > 0) {
      const pcmBytes = this.playbackQueue.shift();
      if (!pcmBytes) break;

      const int16 = new Int16Array(pcmBytes);
      const float32 = new Float32Array(int16.length);
      for (let i = 0; i < int16.length; i++) {
        float32[i] = int16[i] / 32768;
      }

      const buffer = this.playbackContext.createBuffer(
        1,
        float32.length,
        this.receiveSampleRate
      );
      buffer.getChannelData(0).set(float32);

      const source = this.playbackContext.createBufferSource();
      source.buffer = buffer;
      source.connect(this.playbackContext.destination);

      const now = this.playbackContext.currentTime;
      const startAt = Math.max(now, this.nextPlayTime);
      source.start(startAt);
      this.nextPlayTime = startAt + buffer.duration;
    }

    this.isPlaying = false;
  }

  /**
   * AudioWorklet のダウンサンプリングプロセッサーを Blob URL で生成
   *
   * 48kHz → 16kHz のダウンサンプリング（1/3 間引き）
   * 指定された chunkDurationMs ごとに PCM チャンクを main thread に送信
   */
  private createDownsampleWorkletUrl(): string {
    const code = `
class DownsampleProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = options.processorOptions || {};
    this.targetRate = opts.targetSampleRate || 16000;
    this.chunkMs = opts.chunkDurationMs || 100;
    this.ratio = Math.round(sampleRate / this.targetRate);
    this.chunkSize = Math.floor(this.targetRate * this.chunkMs / 1000);
    this.buffer = new Int16Array(this.chunkSize);
    this.bufferIndex = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    const samples = input[0];
    for (let i = 0; i < samples.length; i += this.ratio) {
      const val = Math.max(-1, Math.min(1, samples[i]));
      this.buffer[this.bufferIndex++] = Math.floor(val * 32767);

      if (this.bufferIndex >= this.chunkSize) {
        this.port.postMessage({
          type: 'pcm-chunk',
          data: this.buffer.slice(0),
        });
        this.bufferIndex = 0;
      }
    }
    return true;
  }
}
registerProcessor('downsample-processor', DownsampleProcessor);
`;
    const blob = new Blob([code], { type: 'application/javascript' });
    return URL.createObjectURL(blob);
  }

  private int16ArrayToBase64(data: Int16Array): string {
    const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    let binary = '';
    for (let i = 0; i < bytes.length; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
  }

  private base64ToArrayBuffer(base64: string): ArrayBuffer {
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    return bytes.buffer;
  }
}
