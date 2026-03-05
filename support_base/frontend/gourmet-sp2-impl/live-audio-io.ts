/**
 * Live API 用オーディオ I/O
 *
 * ★★★ 重要 ★★★
 * マイク制御コード（getUserMedia, AudioContext, AudioWorklet）は
 * iPhone 16/17 の iOS 18-19 セキュリティ制限への対策コードである。
 * 既存の AudioManager (gourmet-sp2) のパターンをそのまま踏襲すること。
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

export interface LiveAudioIOOptions {
  wsClient: LiveWSClient;
  sendSampleRate?: number;
  receiveSampleRate?: number;
  chunkDurationMs?: number;
}

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

  // 再生中の currentTime を外部から参照可能にする（アバター同期用）
  private _playbackStartTime = 0;

  private isIOS = /iPhone|iPad|iPod/i.test(navigator.userAgent);

  constructor(options: LiveAudioIOOptions) {
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
   * - 既存 AudioManager の startRecording() パターンを踏襲
   */
  async startMic(): Promise<void> {
    if (this.isMicActive) return;

    try {
      // @ts-ignore
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      this.audioContext = new AudioContextClass({
        latencyHint: 'interactive',
        sampleRate: 48000,
      });

      if (this.audioContext.state === 'suspended') {
        await this.audioContext.resume();
      }

      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 48000,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      const nativeSampleRate = this.audioContext.sampleRate;
      const downsampleRatio = nativeSampleRate / this.sendSampleRate;
      const chunkSize = Math.floor(this.sendSampleRate * this.chunkDurationMs / 1000);

      // AudioWorklet でダウンサンプリング (48kHz → 16kHz)
      // iOS固有のprocessor名にタイムスタンプを付加（AudioManager パターン踏襲）
      const processorName = this.isIOS
        ? 'live-downsample-ios-' + Date.now()
        : 'live-downsample-processor';

      const code = `
class LiveDownsampleProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = options.processorOptions || {};
    this.ratio = ${downsampleRatio};
    this.chunkSize = ${chunkSize};
    this.buffer = new Int16Array(this.chunkSize);
    this.bufferIndex = 0;
    this.inputSampleCount = 0;
    ${this.isIOS ? 'this.lastFlushTime = Date.now();' : ''}
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;
    const samples = input[0];

    for (let i = 0; i < samples.length; i++) {
      this.inputSampleCount++;
      if (this.inputSampleCount >= this.ratio) {
        this.inputSampleCount -= this.ratio;
        const val = Math.max(-1, Math.min(1, samples[i]));
        this.buffer[this.bufferIndex++] = Math.floor(val * 32767);

        if (this.bufferIndex >= this.chunkSize${this.isIOS ? ' || (this.bufferIndex > 0 && Date.now() - this.lastFlushTime > 500)' : ''}) {
          const chunk = this.buffer.slice(0, this.bufferIndex);
          this.port.postMessage({ type: 'pcm-chunk', data: chunk }, [chunk.buffer]);
          this.buffer = new Int16Array(this.chunkSize);
          this.bufferIndex = 0;
          ${this.isIOS ? 'this.lastFlushTime = Date.now();' : ''}
        }
      }
    }
    return true;
  }
}
registerProcessor('${processorName}', LiveDownsampleProcessor);
`;

      const blob = new Blob([code], { type: 'application/javascript' });
      const processorUrl = URL.createObjectURL(blob);
      await this.audioContext.audioWorklet.addModule(processorUrl);
      URL.revokeObjectURL(processorUrl);

      this.workletNode = new AudioWorkletNode(this.audioContext, processorName);

      // Worklet → WebSocket 送信
      this.workletNode.port.onmessage = (event) => {
        if (event.data.type === 'pcm-chunk') {
          const pcmData: Int16Array = event.data.data;
          const base64 = this.int16ArrayToBase64(pcmData);
          this.wsClient.sendAudio(base64);
        }
      };

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

  stopMic(): void {
    if (this.workletNode) {
      this.workletNode.port.onmessage = null;
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
    console.log('[LiveAudioIO] Playback stopped (barge-in)');
  }

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

  /** 再生開始からの経過時間（アバター同期用） */
  get playbackCurrentTime(): number {
    if (!this.playbackContext) return 0;
    return this.playbackContext.currentTime - this._playbackStartTime;
  }

  private async processPlaybackQueue(): Promise<void> {
    if (!this.playbackContext) {
      // @ts-ignore
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      this.playbackContext = new AudioContextClass({ sampleRate: this.receiveSampleRate });
    }

    if (this.playbackContext.state === 'suspended') {
      await this.playbackContext.resume();
    }

    this.isPlaying = true;
    this._playbackStartTime = this.playbackContext.currentTime;

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
