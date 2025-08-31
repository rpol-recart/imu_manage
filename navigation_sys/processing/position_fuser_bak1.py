# /processing/position_fuser.py
"""
Модуль для фузии данных GPS и IMU для получения точной позиции и ориентации.
"""

from typing import Dict, List, Any, Optional, Tuple
from ..services.logger_service import LoggerService
from ..utils.interpolation import interpolate_gps_data, interpolate_orientation_data
from ..utils.geometry import calculate_machine_center_and_azimuth
from ..configs import CONFIG
from geopy.distance import geodesic
from geographiclib.geodesic import Geodesic


class PositionFuser:
    """
    Объединяет данные GPS и IMU для получения финальной оценки позиции, ориентации и достоверности.
    """

    def __init__(self):
        """
        Инициализация фьюзера позиции.

        Args:
            config (dict): Конфигурация модуля.
        """
        self.logger = LoggerService.get_logger(self.__class__.__name__)
        self.config = CONFIG
        self.gps_config = self.config.data.gps
        self.interp_config = self.config.data.interpolation
        self.logger.info("PositionFuser инициализирован")

    def fuse_position_data(self, sensor_data: Dict[str, List[Dict]], target_timestamp: int) -> Dict[str, Any]:
        """
        Фузия данных сенсоров для получения состояния на заданное время.

        Args:
            sensor_data (dict): Словарь с данными сенсоров, полученными из БД.
                                {
                                    'gps': [gps_record1, gps_record2, ...],
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
            gps_records = sensor_data.get('gps', [])
            imu_records = sensor_data.get('imu_orientation_data', [])

            if not gps_records:
                self.logger.info('Нет данных GPS ')
            # --- 1. Определение конфигурации датчиков
            aggregated_gps, gps_timestamps = self.aggregate_gps(gps_records)

            sensors_count = len(aggregated_gps.keys())
            
            # --- 1.1 Работает 1 датчик
            if sensors_count == 1:
                # Если работает 1 датчик
                # Устанавливаем азмут
                if not imu_records:
                    
                    last_point_and_azimuth = self.extract_last_point_one_gps(
                        aggregated_gps)
                    
                    azimuth_source = 'gps'
                else:
                    imu_azimuth = self.set_azimuth_one_gps_imu(
                        aggregated_gps, imu_records)
                    last_point_and_azimuth = self.extract_last_point_one_gps(
                        aggregated_gps, imu_azimuth)
                    azimuth_source = 'imu'

                result = {
                    "lat": last_point_and_azimuth["lat"] if last_point_and_azimuth else None,
                    "lon": last_point_and_azimuth["lon"] if last_point_and_azimuth else None,
                    "alt": last_point_and_azimuth["alt"] if last_point_and_azimuth else None,
                    "azimuth": last_point_and_azimuth['azimuth'],
                    "timestamp": target_timestamp,
                    "confidence": 0.0,  # Будет рассчитано в PositionEstimator
                    "status": "unknown",  # Будет установлено в PositionEstimator
                    "source": azimuth_source,  # Источник азимута
                }
            if sensors_count == 2:
                
                # Проверяем насколько свежие данные
                max_timestamps = {}

                for key, records in aggregated_gps.items():
                    # Извлекаем все значения 'timestamp' из словарей внутри кортежей
                    timestamps = [record[0] for record in records]
                    max_timestamps[key] = max(timestamps)
                print(max_timestamps, target_timestamp)
            # --- 1. Интерполяция/экстраполяция данных ---

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

    def aggregate_gps(self, gps_records):
        aggregated_gps = {}
        gps_timestamps = {}
        for record in gps_records:
            sid = record['sensor_id']
            if sid not in aggregated_gps:
                aggregated_gps[sid] = []
                gps_timestamps[sid] = []
            aggregated_gps[sid].append(
                (record['timestamp'], record['data']))
            gps_timestamps[sid].append(record['timestamp'])
        return aggregated_gps, gps_timestamps

    def extract_last_point_one_gps(self, aggregated_gps, azimuth=None):
        gps_list = list(aggregated_gps.items())[0][1]
        # Берём две последние по времени точки (первые в списке)
        _, prev_point = gps_list[1]  # предпоследняя
        _, last_point = gps_list[0]  # последняя
        if azimuth is None:
            azimuth = get_azimuth_by_points(last_point, prev_point)

        return {'timestamp': last_point['timestamp'],
                'lat': last_point['lat'],
                'lon': last_point['lon'],
                'alt': last_point['alt'],
                'azimuth': azimuth}


def get_azimuth_by_points(last_point, prev_point):
    # Координаты: (широта, долгота)
    point1 = (prev_point['lat'], prev_point['lon'])
    point2 = (last_point['lat'], last_point['lon'])

    # Вычисляем начальный азимут (в градусах) 
    # Calculate initial bearing
    geod = Geodesic.WGS84
    result = geod.Inverse(point1[0], point1[1], point2[0], point2[1])
    azimuth = result['azi1']
    # Округляем
    return round(azimuth, 2)
