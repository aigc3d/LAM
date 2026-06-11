/**
 * VRM Expression Manager - A2Eブレンドシェイプ→ボーン変換
 *
 * A2Eサービスから受け取った52次元ARKitブレンドシェイプ係数を
 * GVRMのボーンシステムにマッピングする。
 *
 * 現状のGVRMレンダラーはGaussian Splattingベースのボーン変形を使用:
 *   - Index 22: Jaw (口の開閉)
 *   - Index 15: Head (頭の微細な動き)
 *   - Index 9: Chest (呼吸)
 *
 * A2Eの52次元出力のうち、リップシンクに重要なブレンドシェイプを
 * 既存のボーンシステムにマッピングして、従来のFFT音量ベースよりも
 * 正確なリップシンクを実現する。
 *
 * 使い方 (concierge-controller.ts):
 *   import { ExpressionManager } from './vrm-expression-manager';
 *   const exprMgr = new ExpressionManager(this.guavaRenderer);
 *   exprMgr.playExpressionFrames(expressionData, audioElement);
 */

// A2Eサービスからのレスポンス型
export interface ExpressionData {
  names: string[];       // 52個のARKitブレンドシェイプ名
  frames: number[][];    // フレームごとの52次元係数
  frame_rate: number;    // fps (通常30)
}

// ARKitブレンドシェイプ名→インデックスのマップ
const ARKIT_INDEX: Record<string, number> = {
  eyeBlinkLeft: 0, eyeLookDownLeft: 1, eyeLookInLeft: 2, eyeLookOutLeft: 3,
  eyeLookUpLeft: 4, eyeSquintLeft: 5, eyeWideLeft: 6,
  eyeBlinkRight: 7, eyeLookDownRight: 8, eyeLookInRight: 9, eyeLookOutRight: 10,
  eyeLookUpRight: 11, eyeSquintRight: 12, eyeWideRight: 13,
  jawForward: 14, jawLeft: 15, jawRight: 16, jawOpen: 17,
  mouthClose: 18, mouthFunnel: 19, mouthPucker: 20, mouthLeft: 21, mouthRight: 22,
  mouthSmileLeft: 23, mouthSmileRight: 24, mouthFrownLeft: 25, mouthFrownRight: 26,
  mouthDimpleLeft: 27, mouthDimpleRight: 28, mouthStretchLeft: 29, mouthStretchRight: 30,
  mouthRollLower: 31, mouthRollUpper: 32, mouthShrugLower: 33, mouthShrugUpper: 34,
  mouthPressLeft: 35, mouthPressRight: 36, mouthLowerDownLeft: 37, mouthLowerDownRight: 38,
  mouthUpperUpLeft: 39, mouthUpperUpRight: 40,
  browDownLeft: 41, browDownRight: 42, browInnerUp: 43, browOuterUpLeft: 44, browOuterUpRight: 45,
  cheekPuff: 46, cheekSquintLeft: 47, cheekSquintRight: 48,
  noseSneerLeft: 49, noseSneerRight: 50,
  tongueOut: 51,
};

export class ExpressionManager {
  private renderer: any;  // GVRM instance
  private currentFrames: number[][] | null = null;
  private frameRate: number = 30;
  private frameIndex: number = 0;
  private animationFrameId: number | null = null;
  private startTime: number = 0;
  private audioElement: HTMLAudioElement | null = null;
  private isPlaying: boolean = false;

  constructor(renderer: any) {
    this.renderer = renderer;
  }

  /**
   * A2E expressionデータを使って音声と同期したリップシンクを再生
   *
   * @param expression A2Eサービスからのレスポンス
   * @param audioElement 音声再生用のHTML Audio要素
   */
  public playExpressionFrames(expression: ExpressionData, audioElement: HTMLAudioElement) {
    this.stop();

    this.currentFrames = expression.frames;
    this.frameRate = expression.frame_rate || 30;
    this.frameIndex = 0;
    this.audioElement = audioElement;
    this.isPlaying = true;

    // 音声再生に同期
    this.startTime = performance.now();
    this.tick();
  }

  /**
   * フレーム更新ループ
   * 音声の現在の再生位置に合わせてフレームを選択
   */
  private tick = () => {
    if (!this.isPlaying || !this.currentFrames || !this.audioElement) {
      this.applyLipSyncLevel(0);
      return;
    }

    // 音声が終了した場合
    if (this.audioElement.paused || this.audioElement.ended) {
      if (this.audioElement.ended) {
        this.applyLipSyncLevel(0);
        this.isPlaying = false;
        return;
      }
    }

    // 音声の再生時間からフレームインデックスを計算
    const currentTime = this.audioElement.currentTime;
    const frameIdx = Math.floor(currentTime * this.frameRate);

    if (frameIdx >= 0 && frameIdx < this.currentFrames.length) {
      const coefficients = this.currentFrames[frameIdx];
      this.applyBlendshapes(coefficients);
    } else if (frameIdx >= this.currentFrames.length) {
      // フレーム切れ → 口を閉じる
      this.applyLipSyncLevel(0);
    }

    this.animationFrameId = requestAnimationFrame(this.tick);
  };

  /**
   * 52次元ブレンドシェイプ係数をボーンシステムにマッピング
   *
   * 現状のGVRMは主にJawボーン(index 22)の回転でリップシンクを実現。
   * A2Eの詳細なブレンドシェイプを、このボーンの回転強度に変換する。
   *
   * 将来的にGVRMがブレンドシェイプ対応すれば、より詳細なマッピングが可能。
   */
  private applyBlendshapes(coefficients: number[]) {
    if (!this.renderer) return;

    // ========================================
    // Step 1: リップシンクレベルの合成
    // 複数のブレンドシェイプから統合的な口の開き度を計算
    // ========================================

    const jawOpen = coefficients[ARKIT_INDEX.jawOpen] || 0;
    const mouthFunnel = coefficients[ARKIT_INDEX.mouthFunnel] || 0;
    const mouthPucker = coefficients[ARKIT_INDEX.mouthPucker] || 0;
    const mouthLowerDownL = coefficients[ARKIT_INDEX.mouthLowerDownLeft] || 0;
    const mouthLowerDownR = coefficients[ARKIT_INDEX.mouthLowerDownRight] || 0;
    const mouthUpperUpL = coefficients[ARKIT_INDEX.mouthUpperUpLeft] || 0;
    const mouthUpperUpR = coefficients[ARKIT_INDEX.mouthUpperUpRight] || 0;

    // 口の開き度 = jawOpen(メイン) + 補助ブレンドシェイプ
    const mouthOpenness = Math.min(1.0,
      jawOpen * 0.6 +
      ((mouthLowerDownL + mouthLowerDownR) / 2) * 0.2 +
      ((mouthUpperUpL + mouthUpperUpR) / 2) * 0.1 +
      mouthFunnel * 0.05 +
      mouthPucker * 0.05
    );

    // GVRMのupdateLipSyncに渡す（0.0〜1.0）
    this.renderer.updateLipSync(mouthOpenness);

    // ========================================
    // Step 2: (将来拡張) 追加ボーンマッピング
    // 現在のVRMManagerにsetLipSync以外のAPIを追加すれば、
    // 以下の情報も活用できる:
    //
    // - mouthSmileLeft/Right → 口角の上げ (表情)
    // - browInnerUp → 眉の動き
    // - cheekPuff → 頬の膨らみ
    // - eyeBlinkLeft/Right → 瞬き
    // ========================================
  }

  /**
   * シンプルなリップシンクレベル適用（フォールバック用）
   */
  private applyLipSyncLevel(level: number) {
    if (this.renderer) {
      this.renderer.updateLipSync(level);
    }
  }

  /**
   * 再生停止
   */
  public stop() {
    this.isPlaying = false;
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
    this.currentFrames = null;
    this.applyLipSyncLevel(0);
  }

  /**
   * expressionデータが有効かどうか
   */
  public static isValid(expression: any): expression is ExpressionData {
    return (
      expression &&
      Array.isArray(expression.names) &&
      Array.isArray(expression.frames) &&
      expression.frames.length > 0 &&
      typeof expression.frame_rate === 'number'
    );
  }
}
