/**
 * グルメモード固有ロジック
 *
 * concierge-controller.ts のグルメ固有部分を抽出。
 * - ショップ表示イベント
 * - センテンス分割（splitIntoSentences）
 * - 並行TTS処理（speakResponseInChunks のパターン）
 *
 * PlatformController から呼び出される。
 */

export interface ShopData {
  name: string;
  area?: string;
  description?: string;
  photo_url?: string;
  rating?: number;
  place_id?: string;
}

/**
 * ショップリストをDOMにディスパッチ
 * concierge-controller.ts L873-874
 */
export function dispatchShopDisplay(shops: ShopData[], language: string): void {
  document.dispatchEvent(
    new CustomEvent('displayShops', {
      detail: { shops, language },
    })
  );
}

/**
 * センテンス単位でテキストを分割
 * concierge-controller.ts L526-546 をそのまま移植
 */
export function splitIntoSentences(text: string, language: string): string[] {
  let separator: RegExp;

  if (language === 'ja' || language === 'zh') {
    separator = /。/;
  } else {
    separator = /\.\s+/;
  }

  const sentences = text.split(separator).filter((s) => s.trim().length > 0);

  return sentences.map((s, idx) => {
    if (
      idx < sentences.length - 1 ||
      text.endsWith('。') ||
      text.endsWith('. ')
    ) {
      return language === 'ja' || language === 'zh' ? s + '。' : s + '. ';
    }
    return s;
  });
}

/**
 * レスポンスからショップ情報を抽出（フォールバック）
 * concierge-controller.ts L1015-1023
 */
export function extractShopsFromResponse(response: string): ShopData[] {
  const shops: ShopData[] = [];
  const lines = response.split('\n');

  for (const line of lines) {
    // "1. **店名**(エリア): 説明" パターン
    const match = line.match(/^\d+\.\s*\*\*(.+?)\*\*(?:\((.+?)\))?\s*[:：]\s*(.+)/);
    if (match) {
      shops.push({
        name: match[1],
        area: match[2] || undefined,
        description: match[3],
      });
    }
  }

  return shops;
}
