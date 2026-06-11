/**
 * LAM WebSocket Manager
 * OpenAvatarChatのバックエンドと通信してリップシンクデータを受信
 *
 * Official synchronization approach:
 * - Server sends BUNDLED audio+expression in JBIN format
 * - Client plays audio and syncs expression based on playback position
 */

import { AudioSyncPlayer } from './audio-sync-player';
import type { AudioSample } from './audio-sync-player';

// JBIN形式のバイナリデータをパース
export interface MotionDataDescription {
  data_records: {
    arkit_face?: {
      shape: number[];
      data_type: string;
      sample_rate: number;
      data_offset: number;
      channel_names: string[];
    };
    avatar_audio?: {
      shape: number[];
      data_type: string;
      sample_rate: number;
      data_offset: number;
    };
  };
  batch_id: number;
  batch_name: string;
  start_of_batch: boolean;
  end_of_batch: boolean;
}

export interface MotionData {
  description: MotionDataDescription;
  arkitFace: Float32Array | null;
  audio: Int16Array | null;
}

export interface ExpressionData {
  [key: string]: number;
}

export interface ExpressionFrameData {
  frames: ExpressionData[];  // All frames for this audio chunk
  frameRate: number;         // Frames per second
  frameCount: number;        // Total number of frames
}

// Bundled motion data group (official sync approach)
export interface MotionDataGroup {
  batchId: number;
  arkitFaceArrays: Float32Array[];  // Expression frames for each audio chunk
  channelNames: string[];
  sampleRate: number;  // Expression frame rate
  arkitFaceShape: number;  // Number of channels per frame (52)
}

/**
 * JBIN形式のバイナリデータをパース
 */
export function parseMotionData(buffer: ArrayBuffer): MotionData {
  const view = new DataView(buffer);

  // マジックナンバー確認 "JBIN"
  const fourcc = String.fromCharCode(
    view.getUint8(0),
    view.getUint8(1),
    view.getUint8(2),
    view.getUint8(3)
  );

  if (fourcc !== 'JBIN') {
    throw new Error(`Invalid JBIN format: ${fourcc}`);
  }

  // ヘッダーサイズ読み取り (Little Endian)
  const jsonSize = view.getUint32(4, true);
  const binSize = view.getUint32(8, true);

  // JSON部分をデコード
  const jsonBytes = new Uint8Array(buffer, 12, jsonSize);
  const jsonString = new TextDecoder().decode(jsonBytes);
  const description: MotionDataDescription = JSON.parse(jsonString);

  // バイナリデータ開始位置
  const binaryOffset = 12 + jsonSize;

  // ARKit顔表情データの抽出
  let arkitFace: Float32Array | null = null;
  if (description.data_records.arkit_face) {
    const faceRecord = description.data_records.arkit_face;
    const faceOffset = binaryOffset + faceRecord.data_offset;
    const faceLength = faceRecord.shape.reduce((a, b) => a * b, 1);
    arkitFace = new Float32Array(buffer, faceOffset, faceLength);
  }

  // オーディオデータの抽出
  let audio: Int16Array | null = null;
  if (description.data_records.avatar_audio) {
    const audioRecord = description.data_records.avatar_audio;
    const audioOffset = binaryOffset + audioRecord.data_offset;
    const audioLength = audioRecord.shape.reduce((a, b) => a * b, 1);
    audio = new Int16Array(buffer, audioOffset, audioLength);
  }

  return { description, arkitFace, audio };
}

/**
 * ARKit表情データをExpressionDataに変換
 */
export function convertToExpressionData(
  arkitFace: Float32Array,
  channelNames: string[]
): ExpressionData {
  const expressionData: ExpressionData = {};
  channelNames.forEach((name, index) => {
    if (index < arkitFace.length) {
      expressionData[name] = arkitFace[index];
    }
  });
  return expressionData;
}

/**
 * LAM WebSocket Manager
 * Handles bundled audio+expression data with official sync approach
 */
export class LAMWebSocketManager {
  private ws: WebSocket | null = null;
  private definition: MotionDataDescription | null = null;
  private channelNames: string[] = [];
  private onExpressionUpdate: ((data: ExpressionData) => void) | null = null;
  private onExpressionFrames: ((data: ExpressionFrameData) => void) | null = null;
  private onAudioData: ((audio: Int16Array) => void) | null = null;
  private onConnectionChange: ((connected: boolean) => void) | null = null;
  private onBatchStarted: ((batchId: number) => void) | null = null;
  private onBatchEnded: ((batchId: number) => void) | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private pingInterval: ReturnType<typeof setInterval> | null = null;
  private currentWsUrl: string = '';

  // Official sync: AudioSyncPlayer + motion data groups
  private audioPlayer: AudioSyncPlayer;
  private motionDataGroups: MotionDataGroup[] = [];
  private currentBatchId: number = -1;
  private arkitFaceShape: number = 52;
  private arkitFaceSampleRate: number = 30;

  constructor(options?: {
    onExpressionUpdate?: (data: ExpressionData) => void;
    onExpressionFrames?: (data: ExpressionFrameData) => void;
    onAudioData?: (audio: Int16Array) => void;
    onConnectionChange?: (connected: boolean) => void;
    onBatchStarted?: (batchId: number) => void;
    onBatchEnded?: (batchId: number) => void;
  }) {
    if (options) {
      this.onExpressionUpdate = options.onExpressionUpdate || null;
      this.onExpressionFrames = options.onExpressionFrames || null;
      this.onAudioData = options.onAudioData || null;
      this.onConnectionChange = options.onConnectionChange || null;
      this.onBatchStarted = options.onBatchStarted || null;
      this.onBatchEnded = options.onBatchEnded || null;
    }

    // Initialize AudioSyncPlayer
    this.audioPlayer = new AudioSyncPlayer({
      sampleRate: 16000,
      onStarted: (batchId) => {
        console.log(`[LAM WebSocket] Audio playback started for batch ${batchId}`);
        this.onBatchStarted?.(batchId);
      },
      onEnded: (batchId) => {
        console.log(`[LAM WebSocket] Audio playback ended for batch ${batchId}`);
        this.onBatchEnded?.(batchId);
        // Clean up old motion data groups
        this.motionDataGroups = this.motionDataGroups.filter(g => g.batchId > batchId);
      }
    });
  }

  /**
   * WebSocket接続を開始
   */
  connect(wsUrl: string): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(wsUrl);
        this.ws.binaryType = 'arraybuffer';

        this.ws.onopen = () => {
          console.log('[LAM WebSocket] Connected');
          this.reconnectAttempts = 0;
          this.currentWsUrl = wsUrl;
          this.onConnectionChange?.(true);
          this.startPing();
          resolve();
        };

        this.ws.onmessage = (event) => {
          this.handleMessage(event);
        };

        this.ws.onclose = (event) => {
          console.log('[LAM WebSocket] Disconnected', event.code, event.reason);
          this.stopPing();
          this.onConnectionChange?.(false);
          this.attemptReconnect(this.currentWsUrl);
        };

        this.ws.onerror = (error) => {
          console.error('[LAM WebSocket] Error:', error);
          reject(error);
        };
      } catch (error) {
        reject(error);
      }
    });
  }

  /**
   * メッセージ処理
   */
  private handleMessage(event: MessageEvent): void {
    if (!(event.data instanceof ArrayBuffer)) {
      // JSON形式のメッセージ（レガシー対応）
      try {
        const msg = JSON.parse(event.data);

        // audio2exp-service からの表情データ（複数フレーム対応）- レガシーJSON形式
        if (msg.type === 'expression' && msg.channels && msg.weights) {
          const frameRate = msg.frame_rate || 30;
          const frameCount = msg.frame_count || msg.weights.length;

          // 複数フレームがある場合はフレームデータとして送信
          if (msg.weights.length > 1 && this.onExpressionFrames) {
            const frames: ExpressionData[] = msg.weights.map((frameWeights: number[]) => {
              const frame: ExpressionData = {};
              msg.channels.forEach((name: string, index: number) => {
                if (index < frameWeights.length) {
                  frame[name] = frameWeights[index];
                }
              });
              return frame;
            });

            this.onExpressionFrames({
              frames,
              frameRate,
              frameCount
            });
            console.log(`[LAM WebSocket] Expression frames received (legacy): ${frameCount} frames at ${frameRate}fps`);
          } else {
            // 1フレームの場合は従来通り
            const expressionData: ExpressionData = {};
            msg.channels.forEach((name: string, index: number) => {
              if (msg.weights[0] && index < msg.weights[0].length) {
                expressionData[name] = msg.weights[0][index];
              }
            });
            this.onExpressionUpdate?.(expressionData);
          }
          return;
        }

        // pong応答
        if (msg.type === 'pong') {
          return;
        }

        console.log('[LAM WebSocket] JSON message:', msg);
      } catch (e) {
        console.warn('[LAM WebSocket] Unknown text message:', event.data);
      }
      return;
    }

    // JBIN形式のバンドルデータを処理（公式同期アプローチ）
    try {
      const motionData = parseMotionData(event.data);
      const desc = motionData.description;

      // チャンネル名を保存
      if (desc.data_records.arkit_face?.channel_names) {
        this.channelNames = desc.data_records.arkit_face.channel_names;
        this.arkitFaceSampleRate = desc.data_records.arkit_face.sample_rate || 30;
        this.arkitFaceShape = desc.data_records.arkit_face.shape?.[1] || 52;
      }

      const batchId = desc.batch_id || 0;

      // 新しいバッチの場合はmotion data groupをリセット
      if (desc.start_of_batch || batchId !== this.currentBatchId) {
        this.currentBatchId = batchId;
        // 新しいグループを作成
        this.motionDataGroups = this.motionDataGroups.filter(g => g.batchId !== batchId);
        this.motionDataGroups.push({
          batchId,
          arkitFaceArrays: [],
          channelNames: this.channelNames,
          sampleRate: this.arkitFaceSampleRate,
          arkitFaceShape: this.arkitFaceShape
        });
      }

      // 表情データを保存
      if (motionData.arkitFace) {
        const group = this.motionDataGroups.find(g => g.batchId === batchId);
        if (group) {
          group.arkitFaceArrays.push(motionData.arkitFace);
        }
      }

      // オーディオデータをプレーヤーに送信
      if (motionData.audio) {
        const audioSample: AudioSample = {
          audioData: motionData.audio,
          sampleRate: desc.data_records.avatar_audio?.sample_rate || 16000,
          batchId,
          endOfBatch: desc.end_of_batch
        };
        this.audioPlayer.feed(audioSample);

        // レガシーコールバックも呼び出し
        this.onAudioData?.(motionData.audio);
      }

      console.log(`[LAM WebSocket] JBIN bundle received: batch=${batchId}, start=${desc.start_of_batch}, end=${desc.end_of_batch}`);

    } catch (error) {
      console.error('[LAM WebSocket] JBIN parse error:', error);
    }
  }

  /**
   * Get current expression frame based on audio playback position
   * This is the official OpenAvatarChat synchronization method
   */
  getCurrentExpressionFrame(): ExpressionData | null {
    const offsetMs = this.audioPlayer.getCurrentPlaybackOffset();
    if (offsetMs < 0) {
      return null;
    }

    // Find the motion data group for current batch
    const group = this.motionDataGroups.find(g => g.batchId === this.audioPlayer.currentBatchId);
    if (!group || group.arkitFaceArrays.length === 0) {
      return null;
    }

    // Get the sample index based on playback position
    const { sampleIndex, subOffsetMs } = this.audioPlayer.getSampleIndexForOffset(offsetMs);
    if (sampleIndex < 0 || sampleIndex >= group.arkitFaceArrays.length) {
      return null;
    }

    // Calculate frame index within the sample
    const frameOffset = Math.floor((subOffsetMs / 1000) * group.sampleRate);
    const arkitFaceArray = group.arkitFaceArrays[sampleIndex];

    // Extract frame data
    const startIdx = frameOffset * group.arkitFaceShape;
    const endIdx = startIdx + group.arkitFaceShape;

    if (startIdx >= arkitFaceArray.length) {
      // Use last frame if we're past the end
      const lastFrameStart = Math.max(0, arkitFaceArray.length - group.arkitFaceShape);
      const frameData = arkitFaceArray.slice(lastFrameStart, lastFrameStart + group.arkitFaceShape);
      return this.arrayToExpressionData(frameData, group.channelNames);
    }

    const frameData = arkitFaceArray.slice(startIdx, endIdx);
    return this.arrayToExpressionData(frameData, group.channelNames);
  }

  /**
   * Convert Float32Array to ExpressionData object
   */
  private arrayToExpressionData(frameData: Float32Array, channelNames: string[]): ExpressionData {
    const result: ExpressionData = {};
    channelNames.forEach((name, index) => {
      if (index < frameData.length) {
        result[name] = frameData[index];
      }
    });
    return result;
  }

  /**
   * Check if audio is currently playing
   */
  isAudioPlaying(): boolean {
    return this.audioPlayer.isPlaying;
  }

  /**
   * Stop audio playback
   */
  stopAudio(): void {
    this.audioPlayer.stop();
  }

  /**
   * Set audio mute state
   */
  setAudioMute(muted: boolean): void {
    this.audioPlayer.setMute(muted);
  }

  /**
   * Initialize audio player (call after user interaction)
   */
  async initializeAudio(): Promise<void> {
    await this.audioPlayer.initialize();
  }

  /**
   * 再接続を試みる
   */
  private attemptReconnect(wsUrl: string): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('[LAM WebSocket] Max reconnect attempts reached');
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
    console.log(`[LAM WebSocket] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);

    setTimeout(() => {
      this.connect(wsUrl).catch(console.error);
    }, delay);
  }

  /**
   * スピーチ終了を通知
   */
  sendEndSpeech(): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        header: { name: 'EndSpeech' }
      }));
    }
  }

  /**
   * 接続を閉じる
   */
  disconnect(): void {
    this.stopPing();
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.definition = null;
    this.channelNames = [];
    this.audioPlayer.stop();
    this.motionDataGroups = [];
  }

  /**
   * Destroy the manager and clean up resources
   */
  destroy(): void {
    this.disconnect();
    this.audioPlayer.destroy();
  }

  /**
   * Ping送信を開始（キープアライブ）
   */
  private startPing(): void {
    this.stopPing();
    this.pingInterval = setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: 'ping' }));
      }
    }, 5000); // 5秒間隔でping
  }

  /**
   * Ping送信を停止
   */
  private stopPing(): void {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }

  /**
   * 接続状態を確認
   */
  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  /**
   * チャンネル名一覧を取得
   */
  getChannelNames(): string[] {
    return this.channelNames;
  }
}

/**
 * ARKit 52チャンネル名（標準）
 */
export const ARKIT_CHANNEL_NAMES = [
  'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight',
  'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight',
  'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft', 'eyeLookDownRight',
  'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight',
  'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight',
  'eyeWideLeft', 'eyeWideRight',
  'jawForward', 'jawLeft', 'jawOpen', 'jawRight',
  'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft', 'mouthFrownRight',
  'mouthFunnel', 'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight',
  'mouthPressLeft', 'mouthPressRight', 'mouthPucker', 'mouthRight',
  'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper',
  'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight',
  'mouthUpperUpLeft', 'mouthUpperUpRight',
  'noseSneerLeft', 'noseSneerRight',
  'tongueOut'
];
