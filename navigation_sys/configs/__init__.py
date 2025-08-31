# storage/config/__init__.py
import threading
from typing import Any, Optional

from .manager import ConfigManager
from .schema import RootConfig
from ..services.logger_service import LoggerService


class ConfigProvider:
    """
    Единый «глобальный» объект, который:
    * читает конфиг один раз (lazy‑load);
    * валидирует его через pydantic;
    * предоставляет удобный API (get, attribute‑access);
    * умеет «перезагружать» конфиг в рантайме (для тестов, hot‑reload);
    * потокобезопасен.
    """

    _instance_lock = threading.Lock()
    _instance: Optional["ConfigProvider"] = None

    # ---------- Синглтон ----------
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._instance_lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    # ---------- Инициализация ----------
    def __init__(self, config_path: Optional[str] = None):
        # Инициализация будет выполнена только один раз
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self._logger = LoggerService.get_logger(self.__class__.__name__)

        self._config_path = config_path
        self._manager = ConfigManager(config_path=config_path)
        self._reload_lock = threading.RLock()
        self._validated: Optional[RootConfig] = None
        self._load_and_validate()

    # ---------- Приватный метод ----------
    def _load_and_validate(self) -> None:
        """Читает `raw`‑словарь и валидирует его."""
        with self._reload_lock:
            raw = self._manager.raw
            try:
                self._validated = RootConfig(**raw)
                self._logger.info("Configuration validated successfully.")
            except Exception as exc:          # pydantic.ValidationError
                self._logger.error(f"Configuration validation failed: {exc}")
                raise

    # ---------- Публичный API ----------
    def reload(self, config_path: Optional[str] = None) -> None:
        """
        Перезагружает конфигурацию из (возможного) нового файла.
        Полезно в тестах или при «hot‑reload».
        """
        with self._reload_lock:
            if config_path:
                self._config_path = config_path
                self._manager = ConfigManager(config_path=config_path)
            else:
                # переиспользуем уже существующий менеджер, но заставляем его перечитать файл
                self._manager._load()
            self._load_and_validate()

    @property
    def raw(self) -> dict:
        """Сырой словарь без валидации (редко нужен)."""
        return self._manager.raw

    @property
    def data(self) -> RootConfig:
        """Валидированный объект pydantic. Доступ через атрибуты."""
        return self._validated  # type: ignore[return-value]

    # ----- Удобный «get» по dotted‑path -----
    def get(self, dotted_path: str, default: Any = None) -> Any:
        """
        Пример: config.get('gps.validation.max_hdop')
        """
        parts = dotted_path.split(".")
        cur: Any = self._validated
        for part in parts:
            if isinstance(cur, dict):
                cur = cur.get(part, default)
            else:
                cur = getattr(cur, part, default)
            if cur is default:
                break
        return cur

    # ----- Позволяем обращаться как к dict (для старого)
    def __getitem__(self, key: str) -> Any:
        return getattr(self._validated, key)

    def __repr__(self) -> str:
        return f"<ConfigProvider path={self._config_path}>"


# Экспортируем готовый объект, который будет использоваться везде
CONFIG = ConfigProvider()
