"""
Модуль для выполнения геометрических расчетов в системах спутниковой навигации.

Содержит функции для расчета центра машины на основе данных от GPS датчиков
и их смещений, а также для вычисления векторов и азимута.
"""

import math
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass

# Константы для геодезических расчетов
EARTH_RADIUS_METERS = 6378137.0  # WGS84 радиус Земли на экваторе
METERS_PER_DEGREE_LAT = 111111.0  # Приблизительно метров на градус широты


@dataclass
class Position3D:
    """Класс для представления трехмерной позиции."""
    lat: float
    lon: float
    alt: float


@dataclass
class Offset3D:
    """Класс для представления трехмерного смещения."""
    x: float  # Вперед (север)
    y: float  # Влево (запад)
    z: float  # Вверх


class GeometryCalculator:
    """Класс для геометрических расчетов навигационной системы."""

    @staticmethod
    def calculate_machine_center(
        gps_data_list: List[Dict],
        sensor_configs: List[Dict]
    ) -> Optional[Position3D]:
        """
        Рассчитывает центр машины на основе данных от GPS датчиков.

        Args:
            gps_data_list: Список с данными GPS
            sensor_configs: Список конфигураций сенсоров

        Returns:
            Position3D с координатами центра или None при недостатке данных
        """
        if not gps_data_list:
            return None

        num_sensors = len(gps_data_list)

        if num_sensors == 1:
            return GeometryCalculator._calculate_center_single_gps(
                gps_data_list[0], sensor_configs
            )
        elif num_sensors == 2:
            return GeometryCalculator._calculate_center_dual_gps(
                gps_data_list, sensor_configs
            )
        else:
            # Для более чем 2 GPS можно реализовать усреднение или выбор лучших
            return None

    @staticmethod
    def _calculate_center_single_gps(
        gps_data: Dict,
        sensor_configs: List[Dict]
    ) -> Position3D:
        """Расчет центра для одного GPS датчика."""
        sensor_id = gps_data["sensor_id"]
        base_position = Position3D(
            gps_data["lat"],
            gps_data["lon"],
            gps_data["alt"]
        )

        # Поиск конфигурации датчика
        sensor_config = next(
            (s for s in sensor_configs if s["id"] == sensor_id),
            None
        )

        if not sensor_config:
            return base_position

        offset_data = sensor_config["offset"]
        offset = Offset3D(
            offset_data.get("x", 0.0),
            offset_data.get("y", 0.0),
            offset_data.get("z", 0.0)
        )

        return GeometryCalculator._apply_offset_to_position(
            base_position, offset
        )

    @staticmethod
    def _calculate_center_dual_gps(
        gps_data_list: List[Dict],
        sensor_configs: List[Dict]
    ) -> Position3D:
        """Расчет центра для двух GPS датчиков."""
        gps1, gps2 = gps_data_list[0], gps_data_list[1]

        # Геометрический центр между двумя точками
        center_lat = (gps1["lat"] + gps2["lat"]) / 2.0
        center_lon = (gps1["lon"] + gps2["lon"]) / 2.0
        center_alt = (gps1["alt"] + gps2["alt"]) / 2.0

        geometric_center = Position3D(center_lat, center_lon, center_alt)

        # Получение конфигураций датчиков
        config1 = GeometryCalculator._get_sensor_config(
            gps1["sensor_id"], sensor_configs
        )
        config2 = GeometryCalculator._get_sensor_config(
            gps2["sensor_id"], sensor_configs
        )

        if not (config1 and config2):
            return geometric_center

        # Для точной коррекции нужен азимут между GPS
        azimuth = GeometryCalculator.calculate_azimuth_from_gps_vector(
            gps1["lat"], gps1["lon"], gps2["lat"], gps2["lon"]
        )

        # Применение коррекции на основе конфигурации датчиков
        return GeometryCalculator._apply_dual_gps_correction(
            geometric_center, config1, config2, azimuth
        )

    @staticmethod
    def _get_sensor_config(sensor_id: int, sensor_configs: List[Dict]) -> Optional[Dict]:
        """Получение конфигурации датчика по ID."""
        return next(
            (s for s in sensor_configs if s["id"] == sensor_id),
            None
        )

    @staticmethod
    def _apply_offset_to_position(
        position: Position3D,
        offset: Offset3D
    ) -> Position3D:
        """
        Применение смещения к позиции.

        Для точного применения X и Y смещений требуется азимут ориентации машины.
        В данной реализации применяется только Z смещение для высоты.
        """
        # Простое применение только по высоте
        corrected_alt = position.alt + offset.z

        # Для X и Y смещений нужен азимут машины
        # Пока возвращаем исходные lat/lon
        return Position3D(position.lat, position.lon, corrected_alt)

    @staticmethod
    def _apply_dual_gps_correction(
        center: Position3D,
        config1: Dict,
        config2: Dict,
        azimuth: float
    ) -> Position3D:
        """
        Применение коррекции для двух GPS с учетом их смещений.

        Args:
            center: Геометрический центр
            config1, config2: Конфигурации датчиков
            azimuth: Азимут между GPS датчиками
        """
        # Получение смещений
        offset1 = Offset3D(**config1["offset"])
        offset2 = Offset3D(**config2["offset"])

        # Средняя коррекция по высоте
        avg_z_correction = (offset1.z + offset2.z) / 2.0
        corrected_alt = center.alt + avg_z_correction

        # Для точной коррекции X,Y смещений нужна более сложная геометрия
        # В базовой реализации используем геометрический центр
        return Position3D(center.lat, center.lon, corrected_alt)

    @staticmethod
    def calculate_azimuth_from_gps_vector(
        lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """
        Рассчитывает азимут по вектору между двумя GPS точками.

        Args:
            lat1, lon1: Координаты первой точки (градусы)
            lat2, lon2: Координаты второй точки (градусы)

        Returns:
            Азимут в градусах (0-360°)
        """
        if abs(lat1 - lat2) < 1e-9 and abs(lon1 - lon2) < 1e-9:
            return 0.0  # Точки совпадают

        # Преобразование в радианы
        lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
        lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)

        delta_lon = lon2_rad - lon1_rad

        # Формула азимута с учетом сферической геометрии
        y = math.sin(delta_lon) * math.cos(lat2_rad)
        x = (math.cos(lat1_rad) * math.sin(lat2_rad) -
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon))

        azimuth_rad = math.atan2(y, x)

        # Нормализация в диапазон [0, 360°)
        azimuth_deg = (math.degrees(azimuth_rad) + 360) % 360

        return azimuth_deg

    @staticmethod
    def calculate_gps_course(
        prev_position: Position3D,
        curr_position: Position3D,
        time_delta_sec: float,
        min_distance_threshold: float = 0.1
    ) -> Optional[float]:
        """
        Рассчитывает курс движения на основе изменения GPS координат.

        Args:
            prev_position: Предыдущая позиция
            curr_position: Текущая позиция  
            time_delta_sec: Интервал времени между измерениями
            min_distance_threshold: Минимальное расстояние для расчета курса

        Returns:
            Курс в градусах или None при недостатке движения
        """
        if time_delta_sec <= 0:
            return None

        # Проверка достаточности движения
        distance_moved = GeometryCalculator._calculate_haversine_distance(
            prev_position, curr_position
        )

        if distance_moved < min_distance_threshold:
            return None  # Недостаточно движения для определения курса

        return GeometryCalculator.calculate_azimuth_from_gps_vector(
            prev_position.lat, prev_position.lon,
            curr_position.lat, curr_position.lon
        )

    @staticmethod
    def _calculate_haversine_distance(pos1: Position3D, pos2: Position3D) -> float:
        """Расчет расстояния между двумя точками по формуле гаверсинусов."""
        lat1_rad = math.radians(pos1.lat)
        lon1_rad = math.radians(pos1.lon)
        lat2_rad = math.radians(pos2.lat)
        lon2_rad = math.radians(pos2.lon)

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = (math.sin(dlat/2)**2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        return EARTH_RADIUS_METERS * c

    @staticmethod
    def meters_to_coordinate_delta(
        meters_north: float,
        meters_east: float,
        reference_latitude: float
    ) -> Tuple[float, float]:
        """
        Преобразует смещение в метрах в изменение координат.

        Использует улучшенную формулу с учетом широты для лучшей точности.

        Args:
            meters_north: Смещение на север в метрах
            meters_east: Смещение на восток в метрах  
            reference_latitude: Опорная широта в градусах

        Returns:
            Кортеж (delta_lat, delta_lon) в градусах
        """
        # Преобразование смещения по широте (север-юг)
        delta_lat = meters_north / METERS_PER_DEGREE_LAT

        # Преобразование смещения по долготе с учетом широты
        meters_per_degree_lon = METERS_PER_DEGREE_LAT * math.cos(
            math.radians(reference_latitude)
        )
        delta_lon = meters_east / meters_per_degree_lon

        return delta_lat, delta_lon


# Функции совместимости с исходным API
def calculate_machine_center(
    gps_data_list: List[Dict],
    sensor_configs: List[Dict]
) -> Optional[Tuple[float, float, float]]:
    """Функция совместимости с исходным API."""
    result = GeometryCalculator.calculate_machine_center(
        gps_data_list, sensor_configs)
    return (result.lat, result.lon, result.alt) if result else None


def calculate_azimuth_from_gps_vector(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Функция совместимости с исходным API."""
    return GeometryCalculator.calculate_azimuth_from_gps_vector(lat1, lon1, lat2, lon2)


def calculate_gps_course(
    lat_prev: float, lon_prev: float, lat_curr: float, lon_curr: float,
    time_delta_sec: float
) -> Optional[float]:
    """Функция совместимости с исходным API."""
    prev_pos = Position3D(lat_prev, lon_prev, 0.0)
    curr_pos = Position3D(lat_curr, lon_curr, 0.0)
    return GeometryCalculator.calculate_gps_course(prev_pos, curr_pos, time_delta_sec)


def meters_to_lat_lon_delta(
    meters_lat: float, meters_lon: float, latitude_deg: float
) -> Tuple[float, float]:
    """Функция совместимости с исходным API."""
    return GeometryCalculator.meters_to_coordinate_delta(
        meters_lat, meters_lon, latitude_deg
    )
