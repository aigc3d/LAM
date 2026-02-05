/**
 * AudioSyncPlayer - Audio playback with precise timing for expression sync
 *
 * Official OpenAvatarChat synchronization approach:
 * - Audio and expression data are bundled together from server
 * - This player plays audio and tracks playback position
 * - Expression frames are indexed based on audio playback time
 */

export interface AudioSample {
  audioData: Int16Array | Float32Array;
  sampleRate: number;
  startTime?: number;  // Playback start time in seconds
  batchId: number;
  endOfBatch: boolean;
}

export interface AudioSyncPlayerOptions {
  sampleRate?: number;
  onEnded?: (batchId: number) => void;
  onStarted?: (batchId: number) => void;
}

export class AudioSyncPlayer {
  private audioContext: AudioContext | null = null;
  private gainNode: GainNode | null = null;
  private sampleRate: number;
  private isMuted: boolean = false;

  // Playback tracking
  private _firstStartAbsoluteTime: number | null = null;  // When playback started (Date.now())
  private _samplesList: AudioSample[] = [];
  private _currentBatchId: number = -1;
  private _isPlaying: boolean = false;

  // Callbacks
  private onEnded: ((batchId: number) => void) | null = null;
  private onStarted: ((batchId: number) => void) | null = null;

  // Queued audio sources
  private scheduledSources: AudioBufferSourceNode[] = [];
  private nextStartTime: number = 0;

  constructor(options: AudioSyncPlayerOptions = {}) {
    this.sampleRate = options.sampleRate || 16000;
    this.onEnded = options.onEnded || null;
    this.onStarted = options.onStarted || null;
  }

  /**
   * Initialize audio context (must be called after user interaction)
   */
  async initialize(): Promise<void> {
    if (this.audioContext) return;

    this.audioContext = new AudioContext({ sampleRate: this.sampleRate });
    this.gainNode = this.audioContext.createGain();
    this.gainNode.connect(this.audioContext.destination);
    this.gainNode.gain.value = this.isMuted ? 0 : 1;

    // Resume context if suspended
    if (this.audioContext.state === 'suspended') {
      await this.audioContext.resume();
    }
  }

  /**
   * Get the absolute time when playback started
   */
  get firstStartAbsoluteTime(): number | null {
    return this._firstStartAbsoluteTime;
  }

  /**
   * Get all samples list with their start times
   */
  get samplesList(): AudioSample[] {
    return this._samplesList;
  }

  /**
   * Get current batch ID
   */
  get currentBatchId(): number {
    return this._currentBatchId;
  }

  /**
   * Check if currently playing
   */
  get isPlaying(): boolean {
    return this._isPlaying;
  }

  /**
   * Feed audio data for playback
   */
  async feed(sample: AudioSample): Promise<void> {
    if (!this.audioContext || !this.gainNode) {
      await this.initialize();
    }

    const ctx = this.audioContext!;
    const gain = this.gainNode!;

    // Check if this is a new batch (new speech)
    if (sample.batchId !== this._currentBatchId) {
      // New batch - reset timing
      this._currentBatchId = sample.batchId;
      this._firstStartAbsoluteTime = null;
      this._samplesList = [];
      this.nextStartTime = ctx.currentTime;

      // Cancel any scheduled sources from previous batch
      this.cancelScheduledSources();
    }

    // Convert Int16 to Float32 if needed
    let audioFloat: Float32Array;
    if (sample.audioData instanceof Int16Array) {
      audioFloat = new Float32Array(sample.audioData.length);
      for (let i = 0; i < sample.audioData.length; i++) {
        audioFloat[i] = sample.audioData[i] / 32768.0;
      }
    } else {
      audioFloat = sample.audioData;
    }

    // Create audio buffer
    const buffer = ctx.createBuffer(1, audioFloat.length, sample.sampleRate);
    buffer.copyToChannel(audioFloat, 0);

    // Create source node
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(gain);

    // Calculate start time
    const startTime = Math.max(ctx.currentTime, this.nextStartTime);
    const duration = audioFloat.length / sample.sampleRate;

    // Record sample info with start time
    const sampleInfo: AudioSample = {
      ...sample,
      startTime: startTime - (this.nextStartTime === ctx.currentTime ? 0 : this.nextStartTime - ctx.currentTime)
    };
    this._samplesList.push(sampleInfo);

    // Track first start time
    if (this._firstStartAbsoluteTime === null) {
      this._firstStartAbsoluteTime = Date.now();
      this._isPlaying = true;
      this.onStarted?.(sample.batchId);
      console.log(`[AudioSyncPlayer] Started batch ${sample.batchId}`);
    }

    // Schedule playback
    source.start(startTime);
    this.scheduledSources.push(source);
    this.nextStartTime = startTime + duration;

    // Handle end of batch
    if (sample.endOfBatch) {
      source.onended = () => {
        this._isPlaying = false;
        console.log(`[AudioSyncPlayer] Ended batch ${sample.batchId}`);
        this.onEnded?.(sample.batchId);
      };
    }

    console.log(`[AudioSyncPlayer] Queued ${duration.toFixed(2)}s audio, batch=${sample.batchId}, end=${sample.endOfBatch}`);
  }

  /**
   * Cancel all scheduled audio sources
   */
  private cancelScheduledSources(): void {
    for (const source of this.scheduledSources) {
      try {
        source.stop();
        source.disconnect();
      } catch (e) {
        // Ignore errors from already stopped sources
      }
    }
    this.scheduledSources = [];
  }

  /**
   * Stop playback and clear queue
   */
  stop(): void {
    this.cancelScheduledSources();
    this._isPlaying = false;
    this._firstStartAbsoluteTime = null;
    this._samplesList = [];
    this.nextStartTime = this.audioContext?.currentTime || 0;
  }

  /**
   * Set mute state
   */
  setMute(muted: boolean): void {
    this.isMuted = muted;
    if (this.gainNode) {
      this.gainNode.gain.value = muted ? 0 : 1;
    }
  }

  /**
   * Destroy the player
   */
  destroy(): void {
    this.stop();
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
    this.gainNode = null;
  }

  /**
   * Calculate current playback offset in milliseconds
   * Used for expression frame synchronization
   */
  getCurrentPlaybackOffset(): number {
    if (!this._firstStartAbsoluteTime || !this._isPlaying) {
      return -1;
    }
    return Date.now() - this._firstStartAbsoluteTime;
  }

  /**
   * Get the sample index for a given offset time
   */
  getSampleIndexForOffset(offsetMs: number): { sampleIndex: number; subOffsetMs: number } {
    if (this._samplesList.length === 0) {
      return { sampleIndex: -1, subOffsetMs: 0 };
    }

    let lastIndex = 0;
    let firstSampleStartTime: number | undefined;

    for (let i = 0; i < this._samplesList.length; i++) {
      const sample = this._samplesList[i];
      if (firstSampleStartTime === undefined && sample.startTime !== undefined) {
        firstSampleStartTime = sample.startTime;
      }
      if (sample.startTime !== undefined &&
          (sample.startTime - (firstSampleStartTime || 0)) * 1000 <= offsetMs) {
        lastIndex = i;
      }
    }

    const sample = this._samplesList[lastIndex];
    const subOffsetMs = offsetMs - (sample.startTime || 0) * 1000;

    return { sampleIndex: lastIndex, subOffsetMs };
  }
}
