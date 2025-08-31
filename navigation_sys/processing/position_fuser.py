# /processing/position_fuser.py
"""
Модуль для фузии данных GPS и IMU для получения точной позиции и ориентации.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Protocol
import logging

from ..services.logger_service import LoggerService
from ..utils.interpolation import interpolate_gps_data, interpolate_orientation_data
from ..utils.geometry import calculate_machine_center_and_azimuth
from ..configs import CONFIG
from geopy.distance import geodesic
from geographiclib.geodesic import Geodesic


class AzimuthSource(Enum):
    """Источники данных азимута."""
    GPS = "gps"
    IMU = "imu"
    NONE = "none"


class FusionStatus(Enum):
    """Статусы результата фузии."""
    SUCCESS = "success"
    UNKNOWN = "unknown"
    FUSION_ERROR = "fusion_error"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass(frozen=True)
class GpsPoint:
    """Точка GPS с координатами и временной меткой."""
    timestamp: int
    lat: float
    lon: float
    alt: float


@dataclass(frozen=True)
class SensorRecord:
    """Запись с сенсора."""
    sensor_id: str
    timestamp: int
    data: Dict[str, Any]


@dataclass(frozen=True)
class FusionResult:
    """Результат фузии данных позиционирования."""
    lat: Optional[float]
    lon: Optional[float]
    alt: Optional[float]
    azimuth: Optional[float]
    timestamp: int
    confidence: float
    status: FusionStatus
    source: AzimuthSource

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь для совместимости с существующим API."""
        return {
            "lat": self.lat,
            "lon": self.lon,
            "alt": self.alt,
            "azimuth": self.azimuth,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "status": self.status.value,
            "source": self.source.value,
        }


class AzimuthCalculator(Protocol):
    """Протокол для вычисления азимута."""
    
    def calculate_azimuth(self, point1: GpsPoint, point2: GpsPoint) -> float:
        """Вычисляет азимут между двумя точками GPS."""
        ...


class GeodeticAzimuthCalculator:
    """Калькулятор азимута на основе геодезических вычислений."""
    
    def __init__(self):
        self._geod = Geodesic.WGS84
    
    def calculate_azimuth(self, point1: GpsPoint, point2: GpsPoint) -> float:
        """
        Вычисляет азимут между двумя точками GPS.
        
        Args:
            point1: Начальная точка (более ранняя по времени)
            point2: Конечная точка (более поздняя по времени)
            
        Returns:
            Азимут в градусах
        """
        result = self._geod.Inverse(
            point1.lat, point1.lon,
            point2.lat, point2.lon
        )
        return round(result['azi1'], 2)


class SensorDataAggregator:
    """Агрегатор данных сенсоров."""
    
    def __init__(self, logger: logging.Logger):
        self._logger = logger
    
    def aggregate_gps_data(self, gps_records: List[SensorRecord]) -> Tuple[Dict[str, List[Tuple[int, Dict[str, Any]]]], Dict[str, List[int]]]:
        """
        Агрегирует данные GPS по sensor_id.
        
        Args:
            gps_records: Список записей GPS
            
        Returns:
            Кортеж из агрегированных данных GPS и временных меток
        """
        aggregated_gps: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
        gps_timestamps: Dict[str, List[int]] = {}
        
        for record in gps_records:
            sensor_id = record.sensor_id
            if sensor_id not in aggregated_gps:
                aggregated_gps[sensor_id] = []
                gps_timestamps[sensor_id] = []
            
            aggregated_gps[sensor_id].append((record.timestamp, record.data))
            gps_timestamps[sensor_id].append(record.timestamp)
        
        self._logger.debug(f"Агрегированы данные GPS от {len(aggregated_gps)} датчиков")
        return aggregated_gps, gps_timestamps


class SingleSensorProcessor:
    """Процессор данных для случая с одним GPS датчиком."""
    
    def __init__(self, azimuth_calculator: AzimuthCalculator, logger: logging.Logger):
        self._azimuth_calculator = azimuth_calculator
        self._logger = logger
    
    def process_single_gps_sensor(
        self, 
        aggregated_gps: Dict[str, List[Tuple[int, Dict[str, Any]]]], 
        imu_records: List[SensorRecord],
        target_timestamp: int
    ) -> FusionResult:
        """
        Обрабатывает данные одного GPS датчика.
        
        Args:
            aggregated_gps: Агрегированные данные GPS
            imu_records: Записи IMU
            target_timestamp: Целевое время
            
        Returns:
            Результат фузии
        """
        try:
            gps_data = list(aggregated_gps.values())[0]
            
            if len(gps_data) < 2:
                self._logger.warning("Недостаточно точек GPS для вычисления азимута")
                return self._create_insufficient_data_result(target_timestamp)
            
            # Сортируем по времени (убывание - последние точки первыми)
            gps_data.sort(key=lambda x: x[0], reverse=True)
            
            last_point_data = gps_data[0][1]
            prev_point_data = gps_data[1][1]
            
            last_point = self._create_gps_point(gps_data[0][0], last_point_data)
            prev_point = self._create_gps_point(gps_data[1][0], prev_point_data)
            
            # Определяем источник азимута
            if imu_records:
                azimuth = self._extract_imu_azimuth(imu_records, target_timestamp)
                azimuth_source = AzimuthSource.IMU
            else:
                azimuth = self._azimuth_calculator.calculate_azimuth(prev_point, last_point)
                azimuth_source = AzimuthSource.GPS
            
            return FusionResult(
                lat=last_point.lat,
                lon=last_point.lon,
                alt=last_point.alt,
                azimuth=azimuth,
                timestamp=target_timestamp,
                confidence=0.0,  # Будет рассчитано в PositionEstimator
                status=FusionStatus.SUCCESS,
                source=azimuth_source
            )
            
        except Exception as e:
            self._logger.error(f"Ошибка обработки одного GPS датчика: {e}")
            return self._create_error_result(target_timestamp)
    
    def _create_gps_point(self, timestamp: int, data: Dict[str, Any]) -> GpsPoint:
        """Создает объект GpsPoint из данных."""
        return GpsPoint(
            timestamp=timestamp,
            lat=data['lat'],
            lon=data['lon'],
            alt=data['alt']
        )
    
    def _extract_imu_azimuth(self, imu_records: List[SensorRecord], target_timestamp: int) -> Optional[float]:
        """Извлекает азимут из данных IMU."""
        # TODO: Реализовать логику извлечения азимута из IMU
        # Пока возвращаем None как заглушку
        self._logger.debug("Извлечение азимута из IMU не реализовано")
        return None
    
    def _create_insufficient_data_result(self, target_timestamp: int) -> FusionResult:
        """Создает результат для случая недостаточных данных."""
        return FusionResult(
            lat=None,
            lon=None,
            alt=None,
            azimuth=None,
            timestamp=target_timestamp,
            confidence=0.0,
            status=FusionStatus.INSUFFICIENT_DATA,
            source=AzimuthSource.NONE
        )
    
    def _create_error_result(self, target_timestamp: int) -> FusionResult:
        """Создает результат для случая ошибки."""
        return FusionResult(
            lat=None,
            lon=None,
            alt=None,
            azimuth=None,
            timestamp=target_timestamp,
            confidence=0.0,
            status=FusionStatus.FUSION_ERROR,
            source=AzimuthSource.NONE
        )


class DualSensorProcessor:
    """Процессор данных для случая с двумя GPS датчиками."""
    
    def __init__(self, logger: logging.Logger):
        self._logger = logger
    
    def process_dual_gps_sensors(
        self, 
        aggregated_gps: Dict[str, List[Tuple[int, Dict[str, Any]]]], 
        target_timestamp: int
    ) -> FusionResult:
        """
        Обрабатывает данные двух GPS датчиков.
        
        Args:
            aggregated_gps: Агрегированные данные GPS
            target_timestamp: Целевое время
            
        Returns:
            Результат фузии
        """
        try:
            # Проверяем свежесть данных
            max_timestamps = self._get_max_timestamps(aggregated_gps)
            self._logger.debug(f"Максимальные временные метки: {max_timestamps}, целевое время: {target_timestamp}")
            
            # TODO: Реализовать логику обработки двух датчиков
            # Пока возвращаем заглушку
            return FusionResult(
                lat=None,
                lon=None,
                alt=None,
                azimuth=None,
                timestamp=target_timestamp,
                confidence=0.0,
                status=FusionStatus.UNKNOWN,
                source=AzimuthSource.NONE
            )
            
        except Exception as e:
            self._logger.error(f"Ошибка обработки двух GPS датчиков: {e}")
            return self._create_error_result(target_timestamp)
    
    def _get_max_timestamps(self, aggregated_gps: Dict[str, List[Tuple[int, Dict[str, Any]]]]) -> Dict[str, int]:
        """Получает максимальные временные метки для каждого датчика."""
        max_timestamps = {}
        for sensor_id, records in aggregated_gps.items():
            timestamps = [record[0] for record in records]
            max_timestamps[sensor_id] = max(timestamps) if timestamps else 0
        return max_timestamps
    
    def _create_error_result(self, target_timestamp: int) -> FusionResult:
        """Создает результат для случая ошибки."""
        return FusionResult(
            lat=None,
            lon=None,
            alt=None,
            azimuth=None,
            timestamp=target_timestamp,
            confidence=0.0,
            status=FusionStatus.FUSION_ERROR,
            source=AzimuthSource.NONE
        )


class PositionFuser:
    """
    Объединяет данные GPS и IMU для получения финальной оценки позиции, ориентации и достоверности.
    
    Применяет принцип Single Responsibility - отвечает только за координацию процесса фузии,
    делегируя специфические задачи соответствующим компонентам.
    """

    def __init__(
        self,
        azimuth_calculator: Optional[AzimuthCalculator] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Инициализация фьюзера позиции.

        Args:
            azimuth_calculator: Калькулятор азимута (по умолчанию GeodeticAzimuthCalculator)
            logger: Логгер (по умолчанию из LoggerService)
        """
        self._logger = logger or LoggerService.get_logger(self.__class__.__name__)
        self._config = CONFIG
        
        # Dependency Injection для гибкости тестирования и расширения
        self._azimuth_calculator = azimuth_calculator or GeodeticAzimuthCalculator()
        self._data_aggregator = SensorDataAggregator(self._logger)
        self._single_sensor_processor = SingleSensorProcessor(self._azimuth_calculator, self._logger)
        self._dual_sensor_processor = DualSensorProcessor(self._logger)
        
        self._logger.info("PositionFuser инициализирован")

    def fuse_position_data(self, sensor_data: Dict[str, List[Dict[str, Any]]], target_timestamp: int) -> Dict[str, Any]:
        """
        Фузия данных сенсоров для получения состояния на заданное время.

        Args:
            sensor_data: Словарь с данными сенсоров, полученными из БД.
                        {
                            'gps': [gps_record1, gps_record2, ...],
                            'imu_orientation_data': [orient_record1, ...]
                        }
                        Каждая запись содержит 'sensor_id', 'timestamp', 'data_json'.
            target_timestamp: Целевое время (Unix timestamp в мс).

        Returns:
            Результат фузии в формате словаря для совместимости с существующим API.
        """
        try:
            # Преобразуем входные данные в типизированные объекты
            gps_records = self._parse_sensor_records(sensor_data.get('gps', []))
            imu_records = self._parse_sensor_records(sensor_data.get('imu_orientation_data', []))

            if not gps_records:
                self._logger.info('Нет данных GPS')
                return self._create_no_data_result(target_timestamp).to_dict()

            # Агрегируем данные GPS
            aggregated_gps, gps_timestamps = self._data_aggregator.aggregate_gps_data(gps_records)
            sensors_count = len(aggregated_gps)

            # Выбираем стратегию обработки в зависимости от количества датчиков
            result = self._process_by_sensor_count(
                sensors_count, aggregated_gps, imu_records, target_timestamp
            )

            self._logger.debug(f"Фузия завершена для времени {target_timestamp}")
            return result.to_dict()

        except Exception as e:
            self._logger.error(f"Ошибка фузии данных: {e}")
            return self._create_error_result(target_timestamp).to_dict()

    def _parse_sensor_records(self, raw_records: List[Dict[str, Any]]) -> List[SensorRecord]:
        """Преобразует сырые данные в типизированные объекты SensorRecord."""
        return [
            SensorRecord(
                sensor_id=record['sensor_id'],
                timestamp=record['timestamp'],
                data=record.get('data_json', record.get('data', {}))
            )
            for record in raw_records
        ]

    def _process_by_sensor_count(
        self, 
        sensors_count: int, 
        aggregated_gps: Dict[str, List[Tuple[int, Dict[str, Any]]]], 
        imu_records: List[SensorRecord],
        target_timestamp: int
    ) -> FusionResult:
        """
        Выбирает стратегию обработки в зависимости от количества датчиков.
        
        Применяет принцип Open/Closed - легко расширяется для новых стратегий.
        """
        if sensors_count == 1:
            return self._single_sensor_processor.process_single_gps_sensor(
                aggregated_gps, imu_records, target_timestamp
            )
        elif sensors_count == 2:
            return self._dual_sensor_processor.process_dual_gps_sensors(
                aggregated_gps, target_timestamp
            )
        else:
            self._logger.warning(f"Неподдерживаемое количество GPS датчиков: {sensors_count}")
            return self._create_insufficient_data_result(target_timestamp)

    def _create_no_data_result(self, target_timestamp: int) -> FusionResult:
        """Создает результат для случая отсутствия данных."""
        return FusionResult(
            lat=None,
            lon=None,
            alt=None,
            azimuth=None,
            timestamp=target_timestamp,
            confidence=0.0,
            status=FusionStatus.INSUFFICIENT_DATA,
            source=AzimuthSource.NONE
        )

    def _create_error_result(self, target_timestamp: int) -> FusionResult:
        """Создает результат для случая ошибки."""
        return FusionResult(
            lat=None,
            lon=None,
            alt=None,
            azimuth=None,
            timestamp=target_timestamp,
            confidence=0.0,
            status=FusionStatus.FUSION_ERROR,
            source=AzimuthSource.NONE
        )

    def _create_insufficient_data_result(self, target_timestamp: int) -> FusionResult:
        """Создает результат для случая недостаточных данных."""
        return FusionResult(
            lat=None,
            lon=None,
            alt=None,
            azimuth=None,
            timestamp=target_timestamp,
            confidence=0.0,
            status=FusionStatus.INSUFFICIENT_DATA,
            source=AzimuthSource.NONE
        )


# Backward compatibility function
def get_azimuth_by_points(last_point: Dict[str, Any], prev_point: Dict[str, Any]) -> float:
    """
    Функция для обратной совместимости. Рекомендуется использовать GeodeticAzimuthCalculator.
    
    Args:
        last_point: Последняя точка с ключами 'lat', 'lon'
        prev_point: Предыдущая точка с ключами 'lat', 'lon'
        
    Returns:
        Азимут в градусах
    """
    calculator = GeodeticAzimuthCalculator()
    
    point1 = GpsPoint(
        timestamp=0,  # Не используется в расчете
        lat=prev_point['lat'],
        lon=prev_point['lon'],
        alt=0  # Не используется в расчете
    )
    
    point2 = GpsPoint(
        timestamp=0,  # Не используется в расчете
        lat=last_point['lat'],
        lon=last_point['lon'],
        alt=0  # Не используется в расчете
    )
    
    return calculator.calculate_azimuth(point1, point2)