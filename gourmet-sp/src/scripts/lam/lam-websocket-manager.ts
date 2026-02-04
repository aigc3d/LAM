/**
 * LAM WebSocket Manager
 * OpenAvatarChatのバックエンドと通信してリップシンクデータを受信
 */

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
 */
export class LAMWebSocketManager {
  private ws: WebSocket | null = null;
  private definition: MotionDataDescription | null = null;
  private channelNames: string[] = [];
  private onExpressionUpdate: ((data: ExpressionData) => void) | null = null;
  private onExpressionFrames: ((data: ExpressionFrameData) => void) | null = null;
  private onAudioData: ((audio: Int16Array) => void) | null = null;
  private onConnectionChange: ((connected: boolean) => void) | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private pingInterval: ReturnType<typeof setInterval> | null = null;
  private currentWsUrl: string = '';

  constructor(options?: {
    onExpressionUpdate?: (data: ExpressionData) => void;
    onExpressionFrames?: (data: ExpressionFrameData) => void;
    onAudioData?: (audio: Int16Array) => void;
    onConnectionChange?: (connected: boolean) => void;
  }) {
    if (options) {
      this.onExpressionUpdate = options.onExpressionUpdate || null;
      this.onExpressionFrames = options.onExpressionFrames || null;
      this.onAudioData = options.onAudioData || null;
      this.onConnectionChange = options.onConnectionChange || null;
    }
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
      // JSON形式のメッセージ
      try {
        const msg = JSON.parse(event.data);

        // audio2exp-service からの表情データ（複数フレーム対応）
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
            console.log(`[LAM WebSocket] Expression frames received: ${frameCount} frames at ${frameRate}fps`);
          } else {
            // 1フレームの場合は従来通り
            const expressionData: ExpressionData = {};
            msg.channels.forEach((name: string, index: number) => {
              if (msg.weights[0] && index < msg.weights[0].length) {
                expressionData[name] = msg.weights[0][index];
              }
            });
            this.onExpressionUpdate?.(expressionData);
            console.log('[LAM WebSocket] Expression update from audio2exp (single frame)');
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

    try {
      const motionData = parseMotionData(event.data);

      // 最初のメッセージは定義情報
      if (!this.definition && motionData.description.data_records.arkit_face) {
        this.definition = motionData.description;
        this.channelNames = motionData.description.data_records.arkit_face.channel_names || [];
        console.log('[LAM WebSocket] Definition received:', this.channelNames.length, 'channels');
        return;
      }

      // 表情データの処理
      if (motionData.arkitFace && this.channelNames.length > 0) {
        const expressionData = convertToExpressionData(motionData.arkitFace, this.channelNames);
        this.onExpressionUpdate?.(expressionData);
      }

      // オーディオデータの処理
      if (motionData.audio) {
        this.onAudioData?.(motionData.audio);
      }
    } catch (error) {
      console.error('[LAM WebSocket] Parse error:', error);
    }
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
