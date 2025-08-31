# storage/config/manager.py
import json
import os
from typing import Dict, Any
from ..services.logger_service import LoggerService


class ConfigManager:
    """
    Низкоуровневый загрузчик JSON‑файла.
    Не делает валидацию – это задача `ConfigProvider`.
    """

    DEFAULT_CONFIG_PATH = os.path.join(
        os.path.dirname(__file__), "default_config.json"
    )

    def __init__(self, config_path: str | None = None):
        self.logger = LoggerService.get_logger(self.__class__.__name__)
        self._config_path = config_path or self.DEFAULT_CONFIG_PATH
        self._raw: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        """Читает файл, поднимает исключения, пишет в лог."""
        if not os.path.exists(self._config_path):
            self.logger.error(f"Config file not found: {self._config_path}")
            # Можно решить: fallback к defaults.json
            raise FileNotFoundError(self._config_path)

        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                self._raw = json.load(f)
            self.logger.info(f"Config loaded from {self._config_path}")
        except json.JSONDecodeError as exc:
            self.logger.error(f"JSON decode error in {self._config_path}: {exc}")
            raise

    @property
    def raw(self) -> Dict[str, Any]:
        """Возвращает «сырой» словарь без копирования (для внутреннего использования)."""
        return self._raw
    



class GPSValidationConfig:
    """Конфигурация параметров валидации GPS."""

    def __init__(self, config: Dict[str, Any]):
        gps_config = config.get("gps", {})
        validation_config = gps_config.get("validation", {})

        self.max_hdop = validation_config.get("max_hdop", 2.5)
        self.max_speed_ms = validation_config.get("max_speed_ms", 20.0)
        self.max_position_jump_speed = validation_config.get(
            "max_position_jump_speed", 100.0)
        self.max_stale_time_ms = validation_config.get(
            "max_stale_time_ms", 5000)
        self.min_time_between_points_ms = validation_config.get(
            "min_time_between_points_ms", 100)
        self.outlier_check_enabled = validation_config.get(
            "outlier_check_enabled", True)
        
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
