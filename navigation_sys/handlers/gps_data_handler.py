import time
import json
import math
import logging
from typing import Dict, Any, Optional, List, NamedTuple, Tuple
from dataclasses import dataclass
from enum import Enum
from ..services.logger_service import LoggerService
from ..storage.database_manager import DatabaseManager


class ValidationResult(Enum):
    """Результаты валидации GPS данных."""
    VALID = "valid"
    INVALID_COORDINATES = "invalid_coordinates"
    POOR_HDOP = "poor_hdop"
    EXCESSIVE_SPEED = "excessive_speed"
    POSITION_OUTLIER = "position_outlier"
    INVALID_NMEA_STATUS = "invalid_nmea_status"
    STALE_TIMESTAMP = "stale_timestamp"


@dataclass
class GPSCoordinates:
    """Структура для GPS координат."""
    lat: float
    lon: float
    alt: float

    def is_valid(self) -> bool:
        """Проверяет базовую валидность координат."""
        return all(
            isinstance(coord, (int, float)) and
            abs(coord) != float('inf') and
            coord == coord  # проверка на NaN
            for coord in [self.lat, self.lon, self.alt]
        )

    def distance_to(self, other: 'GPSCoordinates') -> float:
        """Вычисляет расстояние до другой точки в метрах (формула гаверсинусов)."""
        if not self.is_valid() or not other.is_valid():
            return float('inf')

        # Радиус Земли в метрах
        R = 6371000

        lat1_rad = math.radians(self.lat)
        lat2_rad = math.radians(other.lat)
        delta_lat = math.radians(other.lat - self.lat)
        delta_lon = math.radians(other.lon - self.lon)

        a = (math.sin(delta_lat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c


@dataclass
class GPSMetrics:
    """Дополнительные GPS метрики."""
    hdop: Optional[float] = None
    num_sats: Optional[int] = None
    speed: Optional[float] = None
    course: Optional[float] = None
    nmea_status: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует в словарь для JSON сериализации."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class GPSDataPoint:
    """Полная структура GPS данных."""
    sensor_id: int
    timestamp: int
    coordinates: GPSCoordinates
    metrics: GPSMetrics

    def to_storage_dict(self) -> Dict[str, Any]:
        """Преобразует в формат для хранения в БД."""
        result = {
            "lat": self.coordinates.lat,
            "lon": self.coordinates.lon,
            "alt": self.coordinates.alt,
            "timestamp": self.timestamp
        }
        result.update(self.metrics.to_dict())
        return result

    def estimated_speed_to(self, other: 'GPSDataPoint') -> float:
        """Вычисляет оценочную скорость между двумя точками."""
        if self.timestamp >= other.timestamp:
            return 0.0

        distance = self.coordinates.distance_to(other.coordinates)
        time_diff_s = (other.timestamp - self.timestamp) / 1000.0

        return distance / time_diff_s if time_diff_s > 0 else float('inf')


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


class GPSValidator:
    """Класс для валидации GPS данных."""

    def __init__(self, config: GPSValidationConfig):
        self.config = config
        self.logger = LoggerService.get_logger(self.__class__.__name__)

    def validate(self, data_point: GPSDataPoint, last_valid_point: Optional[GPSDataPoint] = None) -> Tuple[ValidationResult, str]:
        """
        Комплексная валидация GPS данных.

        Returns:
            Tuple[ValidationResult, str]: Результат валидации и описание причины
        """
        # 1. Базовая валидация координат
        if not data_point.coordinates.is_valid():
            return ValidationResult.INVALID_COORDINATES, "Координаты содержат NaN или Inf"

        # 2. Проверка актуальности временной метки
        current_time_ms = int(time.time() * 1000)
        if abs(current_time_ms - data_point.timestamp) > self.config.max_stale_time_ms:
            return ValidationResult.STALE_TIMESTAMP, f"Временная метка слишком старая или будущая"

        # 3. Валидация HDOP
        if (data_point.metrics.hdop is not None and
                data_point.metrics.hdop > self.config.max_hdop):
            return ValidationResult.POOR_HDOP, f"HDOP ({data_point.metrics.hdop}) выше порога ({self.config.max_hdop})"

        # 4. Валидация скорости
        if (data_point.metrics.speed is not None and
                data_point.metrics.speed > self.config.max_speed_ms):
            return ValidationResult.EXCESSIVE_SPEED, f"Скорость ({data_point.metrics.speed} м/с) выше порога ({self.config.max_speed_ms} м/с)"

        # 5. Валидация NMEA статуса
        if (data_point.metrics.nmea_status is not None and
                data_point.metrics.nmea_status != "1"):
            return ValidationResult.INVALID_NMEA_STATUS, f"Невалидный NMEA статус ({data_point.metrics.nmea_status})"

        # 6. Проверка на выбросы позиции (если есть предыдущие данные)
        if last_valid_point and self.config.outlier_check_enabled:
            outlier_result = self._validate_position_continuity(
                data_point, last_valid_point)
            if outlier_result[0] != ValidationResult.VALID:
                return outlier_result

        return ValidationResult.VALID, "Данные валидны"

    def _validate_position_continuity(self, current: GPSDataPoint, previous: GPSDataPoint) -> Tuple[ValidationResult, str]:
        """Проверяет непрерывность позиции между двумя точками."""
        time_diff_ms = current.timestamp - previous.timestamp

        # Игнорируем слишком близкие по времени точки
        if time_diff_ms < self.config.min_time_between_points_ms:
            return ValidationResult.VALID, "Точки слишком близко по времени для проверки"

        # Игнорируем слишком старые данные
        if time_diff_ms > self.config.max_stale_time_ms:
            return ValidationResult.VALID, "Предыдущие данные слишком старые для сравнения"

        estimated_speed = previous.estimated_speed_to(current)

        if estimated_speed > self.config.max_position_jump_speed:
            return (ValidationResult.POSITION_OUTLIER,
                    f"Подозрительный скачок позиции (оценочная скорость {estimated_speed:.2f} м/с)")

        return ValidationResult.VALID, "Позиция в пределах нормы"


class SensorDataCache:
    """Кэш для хранения последних данных датчиков."""

    def __init__(self, max_cache_size: int = 1000):
        self._data: Dict[int, GPSDataPoint] = {}
        self._timestamps: Dict[int, int] = {}
        self.max_cache_size = max_cache_size

    def update(self, data_point: GPSDataPoint):
        """Обновляет кэш новыми данными."""
        self._data[data_point.sensor_id] = data_point
        self._timestamps[data_point.sensor_id] = data_point.timestamp

        # Простая очистка кэша при превышении размера
        if len(self._data) > self.max_cache_size:
            self._cleanup_old_entries()

    def get_last_data(self, sensor_id: int) -> Optional[GPSDataPoint]:
        """Получает последние данные для датчика."""
        return self._data.get(sensor_id)

    def get_last_timestamp(self, sensor_id: int) -> Optional[int]:
        """Получает последнюю временную метку для датчика."""
        return self._timestamps.get(sensor_id)

    def get_active_sensors(self, time_threshold_ms: int) -> List[int]:
        """Возвращает список активных датчиков."""
        current_time = int(time.time() * 1000)
        return [
            sensor_id for sensor_id, timestamp in self._timestamps.items()
            if (current_time - timestamp) <= time_threshold_ms
        ]

    def _cleanup_old_entries(self):
        """Удаляет старые записи из кэша."""
        current_time = int(time.time() * 1000)
        old_threshold = current_time - 60000  # Удаляем данные старше 1 минуты

        sensors_to_remove = [
            sensor_id for sensor_id, timestamp in self._timestamps.items()
            if timestamp < old_threshold
        ]

        for sensor_id in sensors_to_remove:
            self._data.pop(sensor_id, None)
            self._timestamps.pop(sensor_id, None)


class GPSDataHandler:
    """
    Обработчик GPS данных с улучшенной архитектурой.
    Обеспечивает приём, валидацию, фильтрацию и сохранение данных от GPS-датчиков.
    """

    def __init__(self, config: Dict[str, Any], database_manager: DatabaseManager):
        """
        Инициализация обработчика GPS данных.

        Args:
            config: Конфигурация модуля
            database_manager: Менеджер базы данных
        """
        self.logger = LoggerService.get_logger(self.__class__.__name__)
        self.config = config
        self.db_manager = database_manager

        # Инициализация компонентов
        self.validation_config = GPSValidationConfig(config)
        self.validator = GPSValidator(self.validation_config)
        self.cache = SensorDataCache()

        # Конфигурация датчиков
        gps_config = config.get("gps", {})
        self.sensors_config = {
            s["id"]: s for s in gps_config.get("sensors", [])
        }

        # Статистика обработки
        self._stats = {
            "total_processed": 0,
            "valid_count": 0,
            "invalid_count": 0,
            "validation_errors": {}
        }

        self.logger.info(
            f"GPSDataHandler инициализирован с {len(self.sensors_config)} датчиками")

    def process_gps_data(self, sensor_id: int, timestamp: int, lat: float, lon: float, alt: float, **kwargs) -> bool:
        """
        Обрабатывает входящие GPS данные.

        Returns:
            bool: True если данные успешно обработаны и сохранены
        """
        try:
            # Создание структуры данных
            coordinates = GPSCoordinates(lat, lon, alt)
            metrics = GPSMetrics(
                hdop=kwargs.get("hdop"),
                num_sats=kwargs.get("num_sats"),
                speed=kwargs.get("speed"),
                course=kwargs.get("course"),
                nmea_status=kwargs.get("nmea_status")
            )
            data_point = GPSDataPoint(
                sensor_id, timestamp, coordinates, metrics)

            # Валидация
            last_valid_point = self.cache.get_last_data(sensor_id)
            validation_result, reason = self.validator.validate(
                data_point, last_valid_point)

            self._update_stats(validation_result)

            if validation_result != ValidationResult.VALID:
                self.logger.warning(
                    f"GPS {sensor_id} данные отклонены: {reason}")
                return False

            # Сохранение в БД
            success = self._save_to_database(data_point)
            if not success:
                return False

            # Обновление кэша
            self.cache.update(data_point)

            self.logger.debug(f"GPS {sensor_id} данные успешно обработаны")
            return True

        except Exception as e:
            self.logger.error(
                f"Ошибка обработки GPS {sensor_id} данных: {e}", exc_info=True)
            return False

    def _save_to_database(self, data_point: GPSDataPoint) -> bool:
        """Сохраняет данные в базу данных."""
        try:
            data_json = json.dumps(data_point.to_storage_dict())
            self.db_manager.add_sensor_data(
                data_point.timestamp,
                "gps",
                data_point.sensor_id,
                data_json
            )
            return True
        except Exception as e:
            self.logger.error(f"Ошибка сохранения в БД: {e}")
            return False

    def _update_stats(self, validation_result: ValidationResult):
        """Обновляет статистику обработки."""
        self._stats["total_processed"] += 1

        if validation_result == ValidationResult.VALID:
            self._stats["valid_count"] += 1
        else:
            self._stats["invalid_count"] += 1
            error_key = validation_result.value
            self._stats["validation_errors"][error_key] = (
                self._stats["validation_errors"].get(error_key, 0) + 1
            )

    def has_recent_data(self, sensor_id: Optional[int] = None, time_threshold_ms: int = 2000) -> bool:
        """
        Проверяет наличие свежих данных.

        Args:
            sensor_id: ID датчика (если None, проверяет любой датчик)
            time_threshold_ms: Порог времени в миллисекундах

        Returns:
            bool: True если есть свежие данные
        """
        if sensor_id is not None:
            last_ts = self.cache.get_last_timestamp(sensor_id)
            if last_ts:
                current_time = int(time.time() * 1000)
                return (current_time - last_ts) <= time_threshold_ms
            return False

        # Проверяем все датчики
        active_sensors = self.cache.get_active_sensors(time_threshold_ms)
        return len(active_sensors) > 0

    def get_last_valid_data(self, sensor_id: int) -> Optional[Dict[str, Any]]:
        """
        Получает последние валидные данные датчика.

        Returns:
            Dict с данными или None
        """
        data_point = self.cache.get_last_data(sensor_id)
        return data_point.to_storage_dict() if data_point else None

    def get_last_timestamp(self, sensor_id: int) -> Optional[int]:
        """Возвращает последнюю временную метку датчика."""
        return self.cache.get_last_timestamp(sensor_id)

    def get_active_sensors(self, time_threshold_ms: int = 2000) -> List[int]:
        """Возвращает список активных датчиков."""
        return self.cache.get_active_sensors(time_threshold_ms)

    def is_vehicle_moving(self, threshold_speed: float = 0.5, min_sensors: int = 1) -> bool:
        """
        Определяет движение транспорта по данным GPS.

        Args:
            threshold_speed: Минимальная скорость для определения движения (м/с)
            min_sensors: Минимальное количество датчиков для подтверждения движения

        Returns:
            bool: True если транспорт движется
        """
        moving_sensors = 0
        active_sensors = self.get_active_sensors()

        for sensor_id in active_sensors:
            data_point = self.cache.get_last_data(sensor_id)
            if (data_point and
                data_point.metrics.speed is not None and
                    data_point.metrics.speed > threshold_speed):
                moving_sensors += 1

                if moving_sensors >= min_sensors:
                    return True

        return False

    def get_processing_stats(self) -> Dict[str, Any]:
        """Возвращает статистику обработки данных."""
        stats = self._stats.copy()
        stats["success_rate"] = (
            stats["valid_count"] / stats["total_processed"] * 100
            if stats["total_processed"] > 0 else 0
        )
        return stats

    def reset_stats(self):
        """Сбрасывает статистику обработки."""
        self._stats = {
            "total_processed": 0,
            "valid_count": 0,
            "invalid_count": 0,
            "validation_errors": {}
        }
        self.logger.info("Статистика GPS обработки сброшена")
