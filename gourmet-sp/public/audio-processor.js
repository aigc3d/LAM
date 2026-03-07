/**
 * AudioWorklet Processor for Real-time PCM Extraction
 * iPhone完全最適化版
 */

class AudioProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    // ★★★ さらにバッファを小さく（遅延最小化） ★★★
    this.bufferSize = 1024; // 2048 → 1024（約0.064秒）
    this.buffer = new Int16Array(this.bufferSize);
    this.bufferIndex = 0;
    this.sampleCount = 0;
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];

    // ★★★ 入力がない場合もカウント（デバッグ用） ★★★
    if (!input || input.length === 0) {
      return true;
    }

    const channelData = input[0];
    if (!channelData || channelData.length === 0) {
      return true;
    }

    // Float32Array を Int16Array に変換
    for (let i = 0; i < channelData.length; i++) {
      this.sampleCount++;
      
      // Float32 (-1.0 ~ 1.0) を Int16 (-32768 ~ 32767) に変換
      const s = Math.max(-1, Math.min(1, channelData[i]));
      const int16Value = Math.round(s < 0 ? s * 0x8000 : s * 0x7FFF);
      
      // バッファに書き込み
      this.buffer[this.bufferIndex++] = int16Value;

      // バッファサイズに達したら送信
      if (this.bufferIndex >= this.bufferSize) {
        // ★★★ コピーではなく新しいバッファを作成 ★★★
        const chunk = new Int16Array(this.buffer);
        this.port.postMessage({ audioChunk: chunk });
        
        // バッファリセット
        this.bufferIndex = 0;
      }
    }

    return true;
  }
}

registerProcessor('audio-processor', AudioProcessor);
