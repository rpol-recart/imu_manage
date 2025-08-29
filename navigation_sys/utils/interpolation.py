# /utils/interpolation.py
"""
Модуль для выполнения интерполяции данных.

Содержит функции для линейной интерполяции скалярных значений (например, координат)
и сферической линейной интерполяции (Slerp) для углов/ориентаций.
"""

import math
from typing import Tuple, Optional

# Константы для оптимизации
_EPSILON = 1e-10
_SLERP_THRESHOLD = 0.9995
_TWO_PI = 2 * math.pi
_PI = math.pi

def interpolate_orientation_data(
    data_list: List[Tuple[int, Dict[str, float]]],
    target_timestamp: int
) -> Optional[Dict[str, float]]:
    """
    Выполняет сферическую линейную интерполяцию (Slerp) кватернионов ориентации IMU.

    Args:
        data_list: Список (timestamp_ms, data_json), где data_json содержит кватернион.
        target_timestamp: Целевое время в миллисекундах.

    Returns:
        Интерполированный кватернион в формате {"quat_w": w, "quat_x": x, ...} или None.
    """
    if len(data_list) < 2:
        return None

    # Сортируем по времени
    sorted_data = sorted(data_list, key=lambda x: x[0])
    
    # Поиск интервала [t0, t1], содержащего target_timestamp
    t0, q0_data = None, None
    t1, q1_data = None, None

    for i in range(len(sorted_data) - 1):
        t_a, data_a = sorted_data[i]
        t_b, data_b = sorted_data[i + 1]
        if t_a <= target_timestamp <= t_b:
            # Проверяем окно интерполяции (±500 мс согласно ТЗ)
            if t_b - t_a <= 1000:  # Разница не более 1 сек (2x500 мс)
                t0, q0_data = t_a, data_a
                t1, q1_data = t_b, data_b
                break

    if t0 is None or t1 is None:
        return None

    # Извлечение кватернионов
    q0 = (q0_data["quat_w"], q0_data["quat_x"], q0_data["quat_y"], q0_data["quat_z"])
    q1 = (q1_data["quat_w"], q1_data["quat_x"], q1_data["quat_y"], q1_data["quat_z"])

    # Параметр интерполяции
    dt = t1 - t0
    if dt == 0:
        return None
    t = (target_timestamp - t0) / dt

    try:
        # Используем существующую функцию Slerp
        w, x, y, z = slerp_quaternion(q0, q1, t)
        return {"quat_w": w, "quat_x": x, "quat_y": y, "quat_z": z}
    except (ValueError, ZeroDivisionError):
        return None

def interpolate_gps_data(data_list: list, target_timestamp: int) -> dict or None:
    """
    Линейная интерполяция GPS-координат по времени.

    Args:
        data_list: Список (timestamp_ms, data_dict), отсортированный по времени.
                   data_dict = {"lat": float, "lon": float, "alt": float}
        target_timestamp: Целевое время (мс).

    Returns:
        dict с ключами "lat", "lon", "alt" или None, если интерполяция невозможна.
    """
    if len(data_list) < 2:
        return None

    # Извлечение временных меток
    timestamps = [item[0] for item in data_list]

    # Проверка: находится ли целевое время в пределах ±500 мс от диапазона данных
    min_time = min(timestamps)
    max_time = max(timestamps)
    window_ms = 500

    if target_timestamp < min_time - window_ms or target_timestamp > max_time + window_ms:
        return None

    # Поиск интервала [t0, t1], содержащего target_timestamp
    for i in range(len(data_list) - 1):
        t0, d0 = data_list[i]
        t1, data1 = data_list[i + 1]
        d1 = data1  # Это data_json

        if t0 <= target_timestamp <= t1:
            if t1 == t0:
                return None  # Защита от деления на ноль

            ratio = (target_timestamp - t0) / (t1 - t0)

            lat = d0["lat"] + (d1["lat"] - d0["lat"]) * ratio
            lon = d0["lon"] + (d1["lon"] - d0["lon"]) * ratio
            alt = d0["alt"] + (d1["alt"] - d0["alt"]) * ratio

            return {"lat": lat, "lon": lon, "alt": alt}

    # Если целевое время вне [min_time, max_time], но в пределах окна ±500 мс
    # Можно выполнить экстраполяцию? Нет — ТЗ требует интерполяции только.
    # Поэтому возвращаем None, если нет подходящего интервала
    return None


def linear_interpolate(x0: float, y0: float, x1: float, y1: float, x: float) -> Optional[float]:
    """
    Выполняет линейную интерполяцию между двумя точками (x0, y0) и (x1, y1) для заданного x.

    Args:
        x0: Координата X первой точки.
        y0: Координата Y первой точки.
        x1: Координата X второй точки.
        y1: Координата Y второй точки.
        x:  Координата X, для которой нужно найти интерполированное значение Y.

    Returns:
        Интерполированное значение Y, или None, если точки совпадают по X или x вне диапазона.
    """
    dx = x1 - x0
    if abs(dx) < _EPSILON:
        return None

    # Быстрая проверка границ
    if (x - x0) * (x - x1) > _EPSILON:
        return None

    # Оптимизированная формула интерполяции
    return y0 + (y1 - y0) * (x - x0) / dx


def _normalize_quaternion_fast(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Быстрая нормализация кватерниона с проверкой на нуль."""
    w, x, y, z = q
    norm_sq = w*w + x*x + y*y + z*z

    if norm_sq < _EPSILON:
        raise ValueError("Невозможно интерполировать нулевой кватернион")

    if abs(norm_sq - 1.0) < _EPSILON:
        return q  # Уже нормализован

    inv_norm = 1.0 / math.sqrt(norm_sq)
    return (w * inv_norm, x * inv_norm, y * inv_norm, z * inv_norm)


def _dot_product_quaternion(q0: Tuple[float, float, float, float],
                            q1: Tuple[float, float, float, float]) -> float:
    """Вычисляет скалярное произведение двух кватернионов."""
    return q0[0]*q1[0] + q0[1]*q1[1] + q0[2]*q1[2] + q0[3]*q1[3]


def slerp(q0: Tuple[float, float, float, float],
          q1: Tuple[float, float, float, float],
          t: float) -> Tuple[float, float, float, float]:
    """
    Выполняет сферическую линейную интерполяцию (SLERP) между двумя кватернионами.

    Args:
        q0: Начальный кватернион (w, x, y, z).
        q1: Конечный кватернион (w, x, y, z).
        t:  Параметр интерполяции (0.0 <= t <= 1.0).

    Returns:
        Интерполированный кватернион (w, x, y, z).

    Raises:
        ValueError: Если t не в диапазоне [0, 1] или кватернион нулевой.
    """
    if not (0.0 <= t <= 1.0):
        raise ValueError(
            "Параметр интерполяции t должен быть в диапазоне [0, 1]")

    # Быстрые проверки граничных случаев
    if t == 0.0:
        return _normalize_quaternion_fast(q0)
    if t == 1.0:
        return _normalize_quaternion_fast(q1)

    # Нормализация кватернионов
    q0_norm = _normalize_quaternion_fast(q0)
    q1_norm = _normalize_quaternion_fast(q1)

    # Скалярное произведение
    dot = _dot_product_quaternion(q0_norm, q1_norm)

    # Выбираем кратчайший путь
    if dot < 0.0:
        q1_norm = (-q1_norm[0], -q1_norm[1], -q1_norm[2], -q1_norm[3])
        dot = -dot

    # Ограничиваем dot для избежания ошибок округления
    dot = min(1.0, dot)

    # Если кватернионы очень близки, используем линейную интерполяцию
    if dot > _SLERP_THRESHOLD:
        w = (1.0 - t) * q0_norm[0] + t * q1_norm[0]
        x = (1.0 - t) * q0_norm[1] + t * q1_norm[1]
        y = (1.0 - t) * q0_norm[2] + t * q1_norm[2]
        z = (1.0 - t) * q0_norm[3] + t * q1_norm[3]
        return _normalize_quaternion_fast((w, x, y, z))

    # SLERP формула
    theta = math.acos(dot)
    sin_theta = math.sin(theta)

    if sin_theta < _EPSILON:
        # Резервный вариант линейной интерполяции
        w = (1.0 - t) * q0_norm[0] + t * q1_norm[0]
        x = (1.0 - t) * q0_norm[1] + t * q1_norm[1]
        y = (1.0 - t) * q0_norm[2] + t * q1_norm[2]
        z = (1.0 - t) * q0_norm[3] + t * q1_norm[3]
        return _normalize_quaternion_fast((w, x, y, z))

    inv_sin_theta = 1.0 / sin_theta
    ratio_a = math.sin((1.0 - t) * theta) * inv_sin_theta
    ratio_b = math.sin(t * theta) * inv_sin_theta

    return (
        ratio_a * q0_norm[0] + ratio_b * q1_norm[0],
        ratio_a * q0_norm[1] + ratio_b * q1_norm[1],
        ratio_a * q0_norm[2] + ratio_b * q1_norm[2],
        ratio_a * q0_norm[3] + ratio_b * q1_norm[3]
    )


def _normalize_angle_fast(angle: float) -> float:
    """Быстрая нормализация угла в диапазон [-π, π]."""
    if -_PI <= angle <= _PI:
        return angle

    # Приводим к диапазону [0, 2π], затем к [-π, π]
    normalized = angle % _TWO_PI
    return normalized - _TWO_PI if normalized > _PI else normalized


def slerp_angle(angle0_rad: float, angle1_rad: float, t: float) -> float:
    """
    Выполняет сферическую линейную интерполяцию между двумя углами (в радианах).

    Args:
        angle0_rad: Начальный угол в радианах.
        angle1_rad: Конечный угол в радианах.
        t:          Параметр интерполяции (0.0 <= t <= 1.0).

    Returns:
        Интерполированный угол в радианах в диапазоне [-π, π].

    Raises:
        ValueError: Если t не в диапазоне [0, 1].
    """
    if not (0.0 <= t <= 1.0):
        raise ValueError(
            "Параметр интерполяции t должен быть в диапазоне [0, 1]")

    # Быстрые проверки граничных случаев
    if t == 0.0:
        return _normalize_angle_fast(angle0_rad)
    if t == 1.0:
        return _normalize_angle_fast(angle1_rad)

    # Нормализация углов
    a0 = _normalize_angle_fast(angle0_rad)
    a1 = _normalize_angle_fast(angle1_rad)

    # Вычисление кратчайшей разности
    diff = a1 - a0
    if diff > _PI:
        diff -= _TWO_PI
    elif diff < -_PI:
        diff += _TWO_PI

    # Интерполяция и нормализация результата
    return _normalize_angle_fast(a0 + t * diff)


# Дополнительные утилиты для пакетной обработки
def linear_interpolate_batch(points: list[Tuple[float, float]], x_values: list[float]) -> list[Optional[float]]:
    """
    Выполняет линейную интерполяцию для нескольких значений x одновременно.

    Args:
        points: Список точек [(x0, y0), (x1, y1), ...] для интерполяции.
        x_values: Список x-координат для интерполяции.

    Returns:
        Список интерполированных y-значений.
    """
    if len(points) < 2:
        return [None] * len(x_values)

    # Сортируем точки по x
    sorted_points = sorted(points, key=lambda p: p[0])
    results = []

    for x in x_values:
        # Находим подходящий интервал
        interpolated = None
        for i in range(len(sorted_points) - 1):
            x0, y0 = sorted_points[i]
            x1, y1 = sorted_points[i + 1]

            if x0 <= x <= x1:
                interpolated = linear_interpolate(x0, y0, x1, y1, x)
                break

        results.append(interpolated)

    return results
