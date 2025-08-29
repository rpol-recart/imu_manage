# /processing/position_fuser.py
"""
Модуль для фузии данных GPS и IMU для получения точной позиции и ориентации.
"""

from typing import Dict, List, Any, Optional, Tuple
from ..services.logger_service import LoggerService
from ..utils.interpolation import interpolate_gps_data, interpolate_orientation_data
from ..utils.geometry import calculate_machine_center_and_azimuth


class PositionFuser:
    """
    Объединяет данные GPS и IMU для получения финальной оценки позиции, ориентации и достоверности.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация фьюзера позиции.

        Args:
            config (dict): Конфигурация модуля.
        """
        self.logger = LoggerService.get_logger(self.__class__.__name__)
        self.config = config
        self.gps_config = config.get("gps", {})
        self.interp_config = config.get("interpolation", {})
        self.logger.info("PositionFuser инициализирован")

    def fuse_position_data(self, sensor_data: Dict[str, List[Dict]], target_timestamp: int) -> Dict[str, Any]:
        """
        Фузия данных сенсоров для получения состояния на заданное время.

        Args:
            sensor_data (dict): Словарь с данными сенсоров, полученными из БД.
                                {
                                    'gps_data': [gps_record1, gps_record2, ...],
                                    'imu_orientation_data': [orient_record1, ...]
                                }
                                Каждая запись содержит 'sensor_id', 'timestamp', 'data_json'.
            target_timestamp (int): Целевое время (Unix timestamp в мс).

        Returns:
            dict: Результат фузии.
                {
                  "lat": float | null,
                  "lon": float | null,
                  "alt": float | null,
                  "azimuth": float | null,
                  "timestamp": int,
                  "confidence": float,
                  "status": str,
                  "source": str,
                }
        """
        try:
            gps_records = sensor_data.get('gps_data', [])
            imu_records = sensor_data.get('imu_orientation_data', [])

            # --- 1. Интерполяция/экстраполяция данных ---
            interpolated_gps = {}
            gps_timestamps = {}
            for record in gps_records:
                sid = record['sensor_id']
                if sid not in interpolated_gps:
                    interpolated_gps[sid] = []
                    gps_timestamps[sid] = []
                interpolated_gps[sid].append(
                    (record['timestamp'], record['data_json']))
                gps_timestamps[sid].append(record['timestamp'])

            # Интерполируем GPS данные для каждого датчика
            gps_data_at_target = {}
            for sensor_id, data_list in interpolated_gps.items():
                if data_list:
                    interp_result = interpolate_gps_data(
                        data_list, target_timestamp)
                    if interp_result:
                        gps_data_at_target[sensor_id] = interp_result
                    else:
                        self.logger.debug(
                            f"Не удалось интерполировать данные GPS {sensor_id} для времени {target_timestamp}")

            # Интерполируем ориентацию IMU
            orientation_at_target = None
            if imu_records:
                imu_data_list = [(r['timestamp'], r['data_json'])
                                 for r in imu_records]
                orientation_at_target = interpolate_orientation_data(
                    imu_data_list, target_timestamp)
                if not orientation_at_target:
                    self.logger.debug(
                        f"Не удалось интерполировать данные ориентации IMU для времени {target_timestamp}")

            # --- 2. Расчет центра и азимута ---
            machine_center, machine_azimuth, azimuth_source = calculate_machine_center_and_azimuth(
                gps_data_at_target, orientation_at_target, self.config
            )

            # --- 3. Формирование результата ---
            result = {
                "lat": machine_center["lat"] if machine_center else None,
                "lon": machine_center["lon"] if machine_center else None,
                "alt": machine_center["alt"] if machine_center else None,
                "azimuth": machine_azimuth,
                "timestamp": target_timestamp,
                "confidence": 0.0,  # Будет рассчитано в PositionEstimator
                "status": "unknown",  # Будет установлено в PositionEstimator
                "source": azimuth_source,  # Источник азимута
            }

            self.logger.debug(
                f"Фузия завершена для времени {target_timestamp}: {result}")
            return result

        except Exception as e:
            self.logger.error(f"Ошибка фузии данных: {e}")
            # Возвращаем минимально возможное состояние
            return {
                "lat": None,
                "lon": None,
                "alt": None,
                "azimuth": None,
                "timestamp": target_timestamp,
                "confidence": 0.0,
                "status": "fusion_error",
                "source": "none",
            }
