# /storage/config_manager.py

import json
import os
from typing import Dict, Any
import logging
# Предполагается, что LoggerService уже реализован
from ..services.logger_service import LoggerService


class ConfigManager:
    """
    Управляет загрузкой и предоставлением конфигурации модуля.
    """

    # Путь к конфигурационному файлу по умолчанию (относительно этого файла)
    DEFAULT_CONFIG_PATH = os.path.join(
        os.path.dirname(__file__), "..", "..", "config.json")

    def __init__(self, config_path: str = None):
        """
        Инициализирует менеджер конфигурации.

        Args:
            config_path (str, optional): Путь к файлу config.json.
                                         Если не указан, используется DEFAULT_CONFIG_PATH.
        """
        self.logger = LoggerService.get_logger(self.__class__.__name__)
        self._config_path = config_path if config_path else self.DEFAULT_CONFIG_PATH
        self._config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self):
        """
        Загружает конфигурацию из файла JSON.
        """
        try:
            if not os.path.exists(self._config_path):
                # Попытка использовать путь по умолчанию, если основной файл не найден
                if self._config_path != self.DEFAULT_CONFIG_PATH and os.path.exists(self.DEFAULT_CONFIG_PATH):
                    self.logger.warning(
                        f"Конфигурационный файл '{self._config_path}' не найден. Используется файл по умолчанию '{self.DEFAULT_CONFIG_PATH}'.")
                    self._config_path = self.DEFAULT_CONFIG_PATH
                else:
                    self.logger.error(
                        f"Конфигурационный файл '{self._config_path}' не найден.")
                    raise FileNotFoundError(
                        f"Файл конфигурации не найден: {self._config_path}")

            with open(self._config_path, 'r', encoding='utf-8') as f:
                self._config = json.load(f)
            self.logger.info(
                f"Конфигурация успешно загружена из '{self._config_path}'.")

        except json.JSONDecodeError as e:
            self.logger.error(
                f"Ошибка декодирования JSON в файле конфигурации '{self._config_path}': {e}")
            raise  # Передаем исключение дальше
        except Exception as e:
            self.logger.error(
                f"Неожиданная ошибка при загрузке конфигурации из '{self._config_path}': {e}")
            raise

    def get_config(self) -> Dict[str, Any]:
        """
        Возвращает полную конфигурацию.

        Returns:
            dict: Словарь с конфигурацией.
        """
        # Возвращаем копию, чтобы избежать случайного изменения оригинала
        return self._config.copy()

    def get(self, key_path: str, default=None):
        """
        Получает значение из конфигурации по пути ключа.

        Args:
            key_path (str): Путь к ключу, например, 'database.path' или 'gps.max_hdop'.
            default: Значение по умолчанию, если ключ не найден.

        Returns:
            Значение ключа или значение по умолчанию.
        """
        keys = key_path.split('.')
        value = self._config
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            self.logger.debug(
                f"Ключ конфигурации '{key_path}' не найден, возвращается значение по умолчанию.")
            return default

# --- Пример config.json для тестирования ---
# {
#   "database": {
#     "path": "/tmp/test_sensor_data.db",
#     "retention_minutes": 1
#   },
#   "gps": {
#     "sensors": [
#       { "id": 1, "offset": { "x": 1.0, "y": 0.0, "z": 1.5 } },
#       { "id": 2, "offset": { "x": -1.0, "y": 0.0, "z": 1.5 } }
#     ],
#     "max_hdop": 2.5,
#     "max_speed_ms": 20
#   },
#   "imu": {
#     "offset": { "x": 0.0, "y": 0.0, "z": 1.2 },
#     "gyro_units": "rad_s",
#     "calibration": {
#       "gyro_interval_min": 30,
#       "required_still_time_sec": 15,
#       "magnetometer_duration_sec": 120
#     }
#   },
#   "fusion": {
#     "kf_process_variance": 0.1,
#     "kf_gps_measurement_variance": 0.5,
#     "complementary_filter_alpha": 0.98
#   },
#   "interpolation": {
#     "time_window_ms": 1000,
#     "max_points": 10
#   },
#   "extrapolation": {
#     "max_imu_time_sec": 120,
#     "max_gps_only_time_sec": 15
#   },
#   "power_management": {
#     "inactivity_timeout_to_standby_sec": 30,
#     "inactivity_timeout_to_off_sec": 300
#   },
#   "system": {
#     "request_timeout_ms": 1000,
#     "default_confidence_threshold": 0.1
#   }
# }
