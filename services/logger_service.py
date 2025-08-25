# /services/logger_service.py
import logging
import logging.handlers
import os
import sys
from typing import Optional


class LoggerService:
    """
    Централизованный сервис логирования для модуля позиционирования.
    Обеспечивает единообразное логирование по всему приложению.
    """
    _initialized = False
    _log_level = logging.INFO
    _log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    _log_directory = "/var/log/position_module"
    _max_file_size_bytes = 10 * 1024 * 1024  # 10 MB
    _backup_count = 5  # Хранить до 5 архивных файлов

    @classmethod
    def configure(cls,
                  log_level: int = logging.INFO,
                  log_directory: str = "/var/log/position_module",
                  log_format: Optional[str] = None,
                  max_file_size_bytes: int = 10 * 1024 * 1024,
                  backup_count: int = 5):
        """
        Настраивает глобальные параметры логирования.
        Должен быть вызван один раз при старте приложения, до получения любых логгеров.

        Args:
            log_level (int): Уровень логирования (например, logging.DEBUG, logging.INFO).
                             По умолчанию logging.INFO.
            log_directory (str): Директория для сохранения log-файлов.
                                 По умолчанию "/var/log/position_module".
            log_format (str, optional): Формат строки лога.
                                        Если None, используется формат по умолчанию.
            max_file_size_bytes (int): Максимальный размер одного log-файла в байтах.
                                       По умоланию 10 MB.
            backup_count (int): Количество архивных файлов для хранения.
                                По умолчанию 5.
        """
        if cls._initialized:
            # Или можно выбрасывать исключение, если повторная настройка недопустима
            cls.get_logger(__name__).warning(
                "LoggerService уже инициализирован. Повторная настройка игнорируется.")
            return

        cls._log_level = log_level
        cls._log_directory = log_directory
        if log_format:
            cls._log_format = log_format
        cls._max_file_size_bytes = max_file_size_bytes
        cls._backup_count = backup_count

        # Создаем директорию для логов, если её нет
        os.makedirs(cls._log_directory, mode=0o755, exist_ok=True)

        # Настройка корневого логгера
        root_logger = logging.getLogger()
        root_logger.setLevel(cls._log_level)

        # Предотвращаем повторное добавление обработчиков, если они уже есть
        if not root_logger.handlers:

            # Создаем форматтер
            formatter = logging.Formatter(cls._log_format)

            # Обработчик для файла с ротацией
            log_file_path = os.path.join(
                cls._log_directory, "position_module.log")
            file_handler = logging.handlers.RotatingFileHandler(
                log_file_path,
                maxBytes=cls._max_file_size_bytes,
                backupCount=cls._backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(cls._log_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

            # Обработчик для консоли (stdout/stderr)
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(cls._log_level)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)

        cls._initialized = True
        cls.get_logger(__name__).info("LoggerService сконфигурирован.")
        cls.get_logger(__name__).debug(
            f"Конфигурация: уровень={log_level}, директория={log_directory}")

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Получает экземпляр логгера с заданным именем.
        Если сервис не был явно сконфигурирован, применяются настройки по умолчанию.

        Args:
            name (str): Имя логгера (обычно __name__ модуля).

        Returns:
            logging.Logger: Настроенный экземпляр логгера.
        """
        if not cls._initialized:
            # Автоматическая инициализация с параметрами по умолчанию, если это необходимо
            # Это позволяет использовать логирование без явного вызова configure()
            # но рекомендуется явная настройка в main()
            cls.configure()  # Используются значения по умолчанию из аргументов метода configure

        # Возвращаем логгер с указанным именем
        # Он будет использовать обработчики, установленные для корневого логгера
        return logging.getLogger(name)

# Пример использования (обычно это происходит в main.py или аналоге):
# if __name__ == "__main__":
#     # Настройка логирования при запуске приложения
#     LoggerService.configure(
#         log_level=logging.DEBUG,
#         log_directory="/tmp/position_module_logs"
#     )
#
#     # Получение логгера в модуле
#     logger = LoggerService.get_logger(__name__)
#     logger.info("Это информационное сообщение.")
#     logger.debug("Это сообщение отладки.")
#     logger.warning("Это предупреждение.")
#     logger.error("Это сообщение об ошибке.")
