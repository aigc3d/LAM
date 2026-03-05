// src/scripts/chat/chat-controller.ts
import { CoreController } from './core-controller';
import { AudioManager } from './audio-manager'; 

export class ChatController extends CoreController {
  
  constructor(container: HTMLElement, apiBase: string) {
    super(container, apiBase);
    this.audioManager = new AudioManager(4500);
    // チャットモードに設定
    this.currentMode = 'chat';
    this.init();
  }

  // 初期化プロセスをオーバーライド
  protected async init() {
    // 親クラスの初期化を実行
    await super.init();
    
    // チャットモード固有の要素とイベントを追加
    const query = (sel: string) => this.container.querySelector(sel) as HTMLElement;
    this.els.modeSwitch = query('#modeSwitch') as HTMLInputElement;
    
    // モードスイッチの初期状態を設定(チャットモード = unchecked)
    if (this.els.modeSwitch) {
      this.els.modeSwitch.checked = false;
      
      // モードスイッチのイベントリスナー追加
      this.els.modeSwitch.addEventListener('change', () => {
        this.toggleMode();
      });
    }
  }

  // モード切り替え処理 - ページ遷移
  private toggleMode() {
    const isChecked = this.els.modeSwitch?.checked;
    if (isChecked) {
      // コンシェルジュモードへページ遷移
      console.log('[ChatController] Switching to Concierge mode...');
      window.location.href = '/concierge';
    }
    // チャットモードは既に現在のページなので何もしない
  }
}
