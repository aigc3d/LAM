"""
モードレジストリ

モードプラグインの登録・取得を管理する。
新モード追加時はプラグインクラスを作成してここに登録するだけ。
"""

import logging

from platform.modes.base_mode import BaseModePlugin

logger = logging.getLogger(__name__)


class ModeRegistry:
    """モードプラグインのレジストリ"""

    def __init__(self):
        self._modes: dict[str, BaseModePlugin] = {}

    def register(self, plugin: BaseModePlugin) -> None:
        """モードプラグインを登録"""
        self._modes[plugin.name] = plugin
        logger.info(f"[ModeRegistry] Registered: {plugin.name} ({plugin.display_name})")

    def get(self, name: str) -> BaseModePlugin | None:
        """モードプラグインを取得"""
        return self._modes.get(name)

    def list_modes(self) -> list[dict]:
        """登録済みモードのリスト"""
        return [
            {"name": m.name, "display_name": m.display_name}
            for m in self._modes.values()
        ]

    def has(self, name: str) -> bool:
        """指定モードが登録済みか"""
        return name in self._modes
