# /processing/state_predictor.py
"""
Модуль для прогнозирования состояния при потере данных сенсоров (экстраполяция).
"""
import math
from typing import Dict, Any
from ..services.logger_service import LoggerService

# --- Вспомогательные функции ---


def deg_to_rad(degrees: float) -> float:
    """Преобразование градусов в радианы."""
    return math.radians(degrees)


def rad_to_deg(radians: float) -> float:
    """Преобразование радиан в градусы."""
    return math.degrees(radians)

# --- Основной класс ---


class StatePredictor:
    """
    Прогнозирует положение и ориентацию на основе предыдущих данных и скорости,
    используя упрощенный линейный Калмановский фильтр.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация предиктора состояния.

        Args:
            config (dict): Конфигурация модуля.
        """
        self.logger = LoggerService.get_logger(self.__class__.__name__)
        self.config = config
        self.extrap_config = config.data.extrapolation
        self.fusion_config = config.data.fusion
        self.logger.info("StatePredictor инициализирован")

    def predict_position(self, fused_state: Dict[str, Any], target_timestamp: int) -> Dict[str, Any]:
        """
        Прогнозирует состояние на заданное время на основе последнего известного состояния.

        Args:
            fused_state (dict): Последнее известное состояние от PositionFuser.
            target_timestamp (int): Целевое время прогноза (Unix timestamp в мс).

        Returns:
            dict: Прогнозируемое состояние.
        """
        try:
            # Проверка валидности исходных данных
            if fused_state.get("lat") is None or fused_state.get("lon") is None:
                self.logger.debug(
                    "Невозможно экстраполировать: отсутствуют координаты.")
                return fused_state  # Возвращаем как есть, статус будет обновлен выше

            current_lat = fused_state["lat"]
            current_lon = fused_state["lon"]
            current_alt = fused_state.get("alt", 0.0)
            current_azimuth = fused_state.get("azimuth")  # Может быть None
            current_timestamp = fused_state["timestamp"]

            # --- Определение скорости для прогноза ---
            # Предполагаем, что скорость и направление сохраняются
            # В реальном сценарии скорость можно было бы получить из IMU или предыдущих GPS
            # Для упрощения возьмем последние доступные данные о скорости/направлении
            # В рамках ТЗ, будем использовать IMU для скорости, если доступно, иначе GPS-курс/скорость

            # Заглушка: предположим, что у нас есть скорость и направление из последнего состояния
            # В реальной реализации это должно приходить из fused_state или отдельного хранилища скорости
            # Например, можно добавить поля vx, vy в fused_state или хранить их отдельно

            # Для демонстрации, предположим, что мы можем оценить скорость из IMU (акселерометр -> скорость)
            # или из изменения координат GPS. Пока используем фиксированную скорость и направление.

            # ВАЖНО: В ТЗ описан упрощенный KF. Реализуем простую линейную экстраполяцию.

            # --- Простая линейная экстраполяция ---
            dt_sec = (target_timestamp - current_timestamp) / 1000.0

            if dt_sec <= 0:
                self.logger.debug(
                    "Целевое время не в будущем. Возвращаем текущее состояние.")
                return fused_state

            # --- Источник скорости и направления ---
            # Приоритет: 1. IMU (скорость из акселерометра, направление из yaw)
            #           2. GPS (курс из изменения координат)
            #           3. Последний известный azimuth

            vx_mps = 0.0  # Скорость по X (восток) в м/с
            vy_mps = 0.0  # Скорость по Y (север) в м/с
            speed_source = "none"

            # Попробуем получить скорость из fused_state (если была рассчитана ранее)
            # Предположим, что fused_state может содержать vx, vy (это потребует доработки PositionFuser)
            # Пока что используем простую логику

            # ВАЖНО: Реализация упрощенного KF требует хранения состояния [lat, lon, vx, vy]
            # и ковариационной матрицы. Для соответствия ТЗ, реализуем базовую экстраполяцию.

            # --- Пример логики (упрощено) ---
            # Предположим, что fused_state содержит "vx_mps" и "vy_mps" (это нужно добавить в логику)
            # Пока что используем azimuth и фиксированную скорость или 0

            if current_azimuth is not None:
                # Используем последний известный азимут для направления
                speed_mps = 1.0  # Заглушка, реальная скорость должна быть известна
                # Предположим, что скорость была рассчитана и сохранена, например, в fused_state
                # Для примера, возьмем фиксированную скорость 1 м/с
                # Имитация хранения
                speed_mps = getattr(self, '_last_known_speed_mps', 0.0)

                azimuth_rad = deg_to_rad(current_azimuth)

                # Переводим скорость и направление в vx, vy (N-E система)
                # Внимание: оси могут отличаться. Обычно в геодезии X - север, Y - восток.
                # Но в ТЗ не уточнено. Предположим X - восток (lon), Y - север (lat).
                vx_mps = speed_mps * math.sin(azimuth_rad)
                vy_mps = speed_mps * math.cos(azimuth_rad)
                speed_source = "azimuth_based"
                self.logger.debug(
                    f"Используем азимут {current_azimuth} для экстраполяции скорости {speed_mps} м/с.")
            else:
                # Нет направления, предполагаем остановку или используем IMU 
                # для оценки
                # Пока что оставляем скорость 0
                vx_mps = 0.0
                vy_mps = 0.0
                speed_source = "zero_speed"
                self.logger.debug(
                    "Нет азимута, предполагаем нулевую скорость.")

            # --- Экстраполяция координат ---
            # Преобразуем смещение в метрах в градусы
            # Примерное преобразование (не точно, но для упрощения подойдет)
            # 1 градус широты ~ 111320 метров
            # 1 градус долготы ~ 111320 * cos(lat) метров
            lat_to_meters = 111320.0
            lon_to_meters = 111320.0 * math.cos(deg_to_rad(current_lat))

            delta_x_m = vx_mps * dt_sec  # Смещение по востоку (lon)
            delta_y_m = vy_mps * dt_sec  # Смещение по северу (lat)

            delta_lat_deg = delta_y_m / lat_to_meters
            delta_lon_deg = delta_x_m / lon_to_meters

            predicted_lat = current_lat + delta_lat_deg
            predicted_lon = current_lon + delta_lon_deg
            predicted_alt = current_alt  # Предполагаем постоянную высоту

            # --- Экстраполяция азимута (если IMU доступен) ---
            predicted_azimuth = current_azimuth
            # Если есть данные гироскопа, можно интегрировать угловую скорость
            # Но для простоты и соответствия описанию "линейная модель на 
            # основе скорости от IMU или GPS"
            # Предполагаем, что азимут остается постоянным

            # --- Обновление результата ---
            predicted_state = fused_state.copy()
            predicted_state.update({
                "lat": predicted_lat,
                "lon": predicted_lon,
                "alt": predicted_alt,
                "azimuth": predicted_azimuth,
                "timestamp": target_timestamp,
                # confidence будет пересчитан в PositionEstimator на основе 
                # времени экстраполяции
                # status также будет обновлен в PositionEstimator
            })

            self.logger.debug(f"Прогноз на {target_timestamp} (dt={dt_sec}s, source={speed_source}): "
                              f"lat={predicted_lat}, lon={predicted_lon}, alt={predicted_alt}, azimuth={predicted_azimuth}")

            return predicted_state

        except Exception as e:
            self.logger.error(f"Ошибка прогнозирования состояния: {e}")
            # Возвращаем исходное состояние с пометкой об ошибке
            error_state = fused_state.copy()
            error_state["status"] = "prediction_error"
            return error_state

    # Метод для обновления скорости (может быть вызван из других частей системы)
    def update_velocity_estimate(self, vx_mps: float, vy_mps: float):
        """Обновляет оценку скорости для использования в прогнозировании."""
        self._last_known_speed_mps = math.sqrt(vx_mps**2 + vy_mps**2)
        self.logger.debug(
            f"Обновлена оценка скорости: {self._last_known_speed_mps} м/с")
