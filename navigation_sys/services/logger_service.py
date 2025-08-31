# navigation_sys/services/logger_service.py
import logging
import logging.handlers
import os
import sys
from typing import Optional, ClassVar


class LoggerService:
    """
    Централизованный сервис логирования.
    Позволяет один раз сконфигурировать корневой логгер и получать
    дочерние логгеры по имени модуля.
    """

    # ---------- Параметры конфигурации (по умолчанию) ----------
    _initialized: ClassVar[bool] = False
    _log_level: ClassVar[int] = logging.DEBUG
    _log_format: ClassVar[str] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    _log_directory: ClassVar[str] = "logs_dir"
    _max_file_size_bytes: ClassVar[int] = 10 * 1024 * 1024   # 10 МБ
    _backup_count: ClassVar[int] = 5

    # ---------- Публичный API ----------
    @classmethod
    def configure(
        cls,
        *,
        log_level: int = _log_level,
        log_directory: str = "logs_dir",
        log_format: Optional[str] = None,
        max_file_size_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
    ) -> None:
        """
        Вызывается один раз при старте приложения.
        """
        if cls._initialized:
            # Если уже сконфигурировано – просто пишем предупреждение
            cls.get_logger(__name__).warning(
                "LoggerService уже инициализирован; повторная конфигурация игнорируется."
            )
            return

        # Сохраняем параметры
        cls._log_level = log_level
        cls._log_directory = log_directory
        if log_format:
            cls._log_format = log_format
        cls._max_file_size_bytes = max_file_size_bytes
        cls._backup_count = backup_count

        # Создаём директорию, если её нет
        os.makedirs(cls._log_directory, mode=0o755, exist_ok=True)

        # ---------- Настройка корневого логгера ----------
        root_logger = logging.getLogger()
        root_logger.setLevel(cls._log_level)

        # Чтобы не добавить обработчики дважды
        if not root_logger.handlers:
            formatter = logging.Formatter(cls._log_format)

            # Файловый обработчик с ротацией
            file_path = os.path.join(cls._log_directory, "position_module.log")
            file_handler = logging.handlers.RotatingFileHandler(
                file_path,
                maxBytes=cls._max_file_size_bytes,
                backupCount=cls._backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(cls._log_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

            # Консольный вывод (stdout)
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(cls._log_level)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)

        cls._initialized = True

        # Сообщаем о завершении конфигурации
        cls.get_logger(__name__).info("LoggerService сконфигурирован.")
        cls.get_logger(__name__).debug(
            f"Конфигурация: level={log_level}, dir={log_directory}"
        )

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Возвращает (или создаёт) логгер с указанным именем.
        Если сервис ещё не сконфигурирован – вызываем configure()
        с параметрами‑по‑умолчанию.
        """
        if not cls._initialized:
            # Автоматическая инициализация «на лету», если кто‑то забыл вызвать configure()
            cls.configure()

        # Дочерний логгер будет наследовать все обработчики от root‑логгера
        return logging.getLogger(name)