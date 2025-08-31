import math
import random
import time
from navigation_sys.api import PositionEstimator


def generate_pallet_truck_gps_data(
    start_lat=55.7558,
    start_lon=37.6173,
    area_size_meters=100,
    duration_seconds=60,
    max_speed_kmh=25,
    sensor_id_1=1,
    sensor_id_2=2,
    lateral_offset_m=2.0,
    start_altitude=150.0,
    start_timestamp_ms=None,
    failure_probability_per_sec=0.02,
    min_failure_duration=1,
    max_failure_duration=10,
    independent_failures=True
):
    """
    Генерирует данные для двух GPS-сенсоров каждую секунду.
    Оба сенсора обновляются в один и тот же временной интервал (одна итерация = одна секунда).
    """
    MAX_SPEED_MS = max_speed_kmh * 1000 / 3600  # м/с
    if start_timestamp_ms is None:
        start_timestamp_ms = int(time.time() * 1000)

    lat = start_lat
    lon = start_lon
    altitude = start_altitude
    heading = random.uniform(0, 360)
    current_time_ms = start_timestamp_ms

    # Состояние сбоев
    in_failure_1 = False
    failure_end_time_ms_1 = None
    in_failure_2 = False
    failure_end_time_ms_2 = None

    def add_distance_to_coords(lat, lon, distance_m, bearing_deg):
        R = 6371000  # Радиус Земли в метрах
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        bearing_rad = math.radians(bearing_deg)

        new_lat_rad = math.asin(
            math.sin(lat_rad) * math.cos(distance_m / R) +
            math.cos(lat_rad) * math.sin(distance_m / R) * math.cos(bearing_rad)
        )
        new_lon_rad = lon_rad + math.atan2(
            math.sin(bearing_rad) * math.sin(distance_m / R) * math.cos(lat_rad),
            math.cos(distance_m / R) - math.sin(lat_rad) * math.sin(new_lat_rad)
        )
        return math.degrees(new_lat_rad), math.degrees(new_lon_rad)

    center_lat, center_lon = start_lat, start_lon
    half_size = area_size_meters / 2

    for sec in range(duration_seconds):
        current_timestamp_ms = int(time.time() * 1000)

        # === Проверка и инициализация сбоев ===
        if not in_failure_1 and random.random() < failure_probability_per_sec:
            in_failure_1 = True
            dur = random.randint(min_failure_duration, max_failure_duration)
            failure_end_time_ms_1 = current_timestamp_ms + dur * 1000

        if independent_failures and not in_failure_2 and random.random() < failure_probability_per_sec:
            in_failure_2 = True
            dur = random.randint(min_failure_duration, max_failure_duration)
            failure_end_time_ms_2 = current_timestamp_ms + dur * 1000
        elif not independent_failures:
            in_failure_2 = in_failure_1
            failure_end_time_ms_2 = failure_end_time_ms_1

        # Завершить сбои, если нужно
        if in_failure_1 and current_timestamp_ms >= failure_end_time_ms_1:
            in_failure_1 = False
        if in_failure_2 and current_timestamp_ms >= failure_end_time_ms_2:
            in_failure_2 = False

        # === Обновление движения тележки ===
        turn = random.uniform(-15, 15)
        heading = (heading + turn) % 360

        speed_ms = random.uniform(0, MAX_SPEED_MS)
        distance_m = speed_ms * 1.0

        new_lat, new_lon = add_distance_to_coords(lat, lon, distance_m, heading)

        # Проверка на выход за границы
        d_lat_m = (new_lat - center_lat) * 111319.9
        d_lon_m = (new_lon - center_lon) * (111319.9 * math.cos(math.radians(center_lat)))
        if abs(d_lat_m) > half_size or abs(d_lon_m) > half_size:
            heading = (heading + 180 + random.uniform(-45, 45)) % 360
            new_lat, new_lon = add_distance_to_coords(lat, lon, distance_m, heading)

        lat, lon = new_lat, new_lon
        alt = altitude + random.uniform(-0.5, 0.5)

        # === Вычисление позиций двух сенсоров (смещение вбок на ±1 метр от центра, итого 2 м между собой) ===
        left_lat, left_lon = add_distance_to_coords(lat, lon, lateral_offset_m / 2, (heading + 90) % 360)
        right_lat, right_lon = add_distance_to_coords(lat, lon, lateral_offset_m / 2, (heading - 90) % 360)

        alt1 = alt + random.uniform(-0.1, 0.1)
        alt2 = alt + random.uniform(-0.1, 0.1)

        # === Формируем данные для двух сенсоров с одинаковой временной меткой ===
        data_1 = {
            "sensor_id": sensor_id_1,
            "timestamp": current_timestamp_ms,
            "lat": 0.0 if in_failure_1 else left_lat,
            "lon": 0.0 if in_failure_1 else left_lon,
            "alt": 0.0 if in_failure_1 else alt1
        }

        data_2 = {
            "sensor_id": sensor_id_2,
            "timestamp": current_timestamp_ms,
            "lat": 0.0 if in_failure_2 else right_lat,
            "lon": 0.0 if in_failure_2 else right_lon,
            "alt": 0.0 if in_failure_2 else alt2
        }

        # Возвращаем оба сенсора подряд с одинаковым timestamp
        yield [data_1, data_2]
        


# === Пример использования ===
if __name__ == "__main__":
    
    print("Моделирование GPS-данных для двух сенсоров (2 м в разные стороны)...")
    estimator = PositionEstimator('/home/roman/projects/imu_manage/config.json')
    #estimator.database_manager.cleanup_old_data(1)
    for data1 in generate_pallet_truck_gps_data(duration_seconds=240, lateral_offset_m=2.0):
        #print(data1)
        for data in data1:
            if data['sensor_id']==2:
                estimator.update_gps(
                    sensor_id=data['sensor_id'],
                    timestamp=data['timestamp'],
                    lat=data['lat'],
                    lon=data['lon'],
                    alt=data['alt']
                )
        current_state = estimator.get_current_state()
        print("Current state:", current_state)
        print("stats",estimator.get_history_stats())
        # Имитация реального времени: ждём 1 секунду на каждую итерацию (т.е. после 2 сенсоров)
        time.sleep(2)