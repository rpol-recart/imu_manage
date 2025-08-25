# /processing/imu_processor.py
"""
Модуль для обработки данных с IMU: фильтрация, вычисление ориентации (Complementary Filter).
"""
import math
import numpy as np
from typing import Dict, Tuple, Optional, Any
from collections import deque
from ..services.logger_service import LoggerService

# --- Вспомогательные функции ---


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Умножение двух кватернионов."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])


def quaternion_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Преобразование углов Эйлера в кватернион."""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z])


def euler_from_quaternion(q: np.ndarray) -> Tuple[float, float, float]:
    """Преобразование кватерниона в углы Эйлера (roll, pitch, yaw)."""
    w, x, y, z = q
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        # use 90 degrees if out of range
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Нормализация вектора."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

# --- Основной класс ---


class IMUProcessor:
    """
    Обрабатывает сырые данные IMU, применяет фильтрацию и вычисляет ориентацию
    с использованием Complementary Filter.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация процессора IMU.

        Args:
            config (dict): Конфигурация модуля.
        """
        self.logger = LoggerService.get_logger(self.__class__.__name__)
        self.config = config.get("imu", {})
        self.fusion_config = config.get("fusion", {})

        # Параметры фильтра
        self.alpha = self.fusion_config.get("complementary_filter_alpha", 0.98)

        # Окно для медианного фильтра
        self.filter_window_size = 5
        self.accel_history = deque(maxlen=self.filter_window_size)
        self.gyro_history = deque(maxlen=self.filter_window_size)
        self.mag_history = deque(maxlen=self.filter_window_size)

        # Калибровочные параметры (загружаются извне)
        self.gyro_bias = np.array([0.0, 0.0, 0.0])
        self.mag_hard_iron = np.array([0.0, 0.0, 0.0])
        # Предполагаем единичную матрицу по умолчанию
        self.mag_soft_iron_inv = np.eye(3)

        # Состояние
        # Начальная ориентация: без поворота
        self.orientation_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.last_timestamp_s = None  # Время последнего обновления в секундах

        self.logger.info("IMUProcessor инициализирован")

    def _convert_gyro(self, gyro_dict: Dict[str, float]) -> np.ndarray:
        """Преобразует угловую скорость в deg/s."""
        gyro = np.array([gyro_dict["x"], gyro_dict["y"], gyro_dict["z"]])
        if self.gyro_units == "rad_s":
            return np.degrees(gyro)
        elif self.gyro_units == "deg_s":
            return gyro
        else:
            self.logger.warning(
                f"Неизвестные единицы: {self.gyro_units}. Используем deg/s.")
            return gyro

    def set_calibration(self, calibration_data: Dict[str, Any]):
        """
        Установка калибровочных параметров.

        Args:
            calibration_data (dict): Данные калибровки из БД.
        """
        try:
            if "imu_gyro_bias" in calibration_data:
                self.gyro_bias = np.array(calibration_data["imu_gyro_bias"])
                self.logger.debug(
                    f"Установлен bias гироскопа: {self.gyro_bias}")

            if "imu_mag_calib" in calibration_data:
                calib = calibration_data["imu_mag_calib"]
                self.mag_hard_iron = np.array(
                    calib.get("hard_iron", [0.0, 0.0, 0.0]))
                # Предполагаем, что soft_iron_inv передается как список списков
                soft_iron_data = calib.get(
                    "soft_iron_inv", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                self.mag_soft_iron_inv = np.array(soft_iron_data)
                self.logger.debug(
                    f"Установлена калибровка магнитометра: hard_iron={self.mag_hard_iron}, soft_iron_inv shape={self.mag_soft_iron_inv.shape}")
        except Exception as e:
            self.logger.error(f"Ошибка при установке калибровки IMU: {e}")

    def _median_filter(self, history: deque, new_value: np.ndarray) -> np.ndarray:
        """Применяет медианный фильтр к новому значению."""
        history.append(new_value)
        if len(history) < self.filter_window_size:
            # Если данных недостаточно, возвращаем последнее значение
            return new_value

        # Медиана по каждому компоненту
        window_array = np.array(history)
        median_filtered = np.median(window_array, axis=0)
        return median_filtered

    def process_imu_sample(self, imu_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Обрабатывает один сэмпл данных IMU: фильтрация и вычисление ориентации.

        Args:
            imu_data (dict): Сырые данные IMU.
                {
                  "timestamp": 1712345678901,
                  "accelerometer": {"x": 0.12, "y": -0.05, "z": 9.78},
                  "gyroscope": {"x": 0.5, "y": -0.3, "z": 2.1},
                  "magnetometer": {"x": 25.3, "y": -10.1, "z": 40.2},
                  "temperature": 32.5,
                  "status": "ok"
                }

        Returns:
            dict or None: Обработанные данные ориентации или None при ошибке.
                {
                  "timestamp": int,
                  "roll": float,
                  "pitch": float,
                  "yaw": float,
                  "quat_w": float,
                  "quat_x": float,
                  "quat_y": float,
                  "quat_z": float
                }
        """
        try:
            timestamp_ms = imu_data["timestamp"]
            accel = np.array([imu_data["accelerometer"]["x"],
                             imu_data["accelerometer"]["y"], imu_data["accelerometer"]["z"]])
            gyro = np.array([imu_data["gyroscope"]["x"],
                            imu_data["gyroscope"]["y"], imu_data["gyroscope"]["z"]])
            mag = np.array([imu_data["magnetometer"]["x"],
                           imu_data["magnetometer"]["y"], imu_data["magnetometer"]["z"]])

            # --- 1. Фильтрация ---
            filtered_accel = self._median_filter(self.accel_history, accel)
            filtered_gyro = self._median_filter(self.gyro_history, gyro)
            filtered_mag = self._median_filter(self.mag_history, mag)

            # --- 2. Компенсация калибровки ---
            corrected_gyro = filtered_gyro - self.gyro_bias
            # Применяем калибровку магнитометра: сначала вычитаем hard iron, затем умножаем на обратную soft iron
            corrected_mag = self.mag_soft_iron_inv @ (
                filtered_mag - self.mag_hard_iron)

            # --- 3. Расчет ориентации ---
            timestamp_s = timestamp_ms / 1000.0
            dt = 0.0
            if self.last_timestamp_s is not None:
                dt = timestamp_s - self.last_timestamp_s

            if dt <= 0 or dt > 1.0:  # Защита от нулевого или слишком большого dt
                self.logger.warning(
                    f"Недопустимый dt для IMU: {dt}. Пропуск шага интеграции.")
                self.last_timestamp_s = timestamp_s
                # Все равно возвращаем текущую ориентацию
                roll, pitch, yaw = euler_from_quaternion(self.orientation_quat)
                return {
                    "timestamp": timestamp_ms,
                    "roll": math.degrees(roll),
                    "pitch": math.degrees(pitch),
                    "yaw": math.degrees(yaw),
                    "quat_w": float(self.orientation_quat[0]),
                    "quat_x": float(self.orientation_quat[1]),
                    "quat_y": float(self.orientation_quat[2]),
                    "quat_z": float(self.orientation_quat[3]),
                }

            # a) Оценка углов из акселерометра (для roll и pitch)
            # Предполагаем, что ось Z направлена вверх
            acc_norm = normalize_vector(filtered_accel)
            if np.linalg.norm(acc_norm) > 0.01:  # Избегаем деления на ноль и шума
                acc_roll = math.atan2(acc_norm[1], acc_norm[2])
                # Используем atan2 для большей устойчивости
                acc_pitch = math.atan2(-acc_norm[0],
                                       math.sqrt(acc_norm[1]**2 + acc_norm[2]**2))
            else:
                acc_roll, acc_pitch = 0.0, 0.0

            # b) Оценка yaw из магнитометра (требует roll и pitch)
            # Проецируем магнитное поле на горизонтальную плоскость
            mag_norm = normalize_vector(corrected_mag)
            if np.linalg.norm(mag_norm) > 0.01:
                # Поворачиваем вектор магнитного поля в систему, где оси X и Y горизонтальны
                # Это упрощенная версия, предполагающая, что roll и pitch малы или уже учтены
                # Более точный способ требует матрицы поворота из текущей ориентации
                # Для простоты и соответствия описанию ТЗ, используем текущую ориентацию для проекции
                # (Это требует итерации, но для комплементарного фильтра обычно берется приблизительная ориентация)
                # Упрощение: используем только roll и pitch из акселерометра для проекции магнитометра
                cos_roll = math.cos(acc_roll)
                sin_roll = math.sin(acc_roll)
                cos_pitch = math.cos(acc_pitch)
                sin_pitch = math.sin(acc_pitch)

                # Проекция магнитного поля на горизонтальную плоскость
                mag_x_horiz = mag_norm[0] * cos_pitch + mag_norm[1] * \
                    sin_roll * sin_pitch + mag_norm[2] * cos_roll * sin_pitch
                mag_y_horiz = mag_norm[1] * cos_roll - mag_norm[2] * sin_roll
                # Вычисляем yaw из проекции
                acc_yaw = math.atan2(-mag_y_horiz, mag_x_horiz)
            else:
                acc_yaw = 0.0

            # c) Интеграция гироскопа (в кватернионах для избежания Gimbal Lock)
            # Преобразуем угловую скорость в кватернион производной
            gyro_rad = np.radians(corrected_gyro)
            # Кватернион производной: 0.5 * q_t * omega_quat
            # omega_quat = [0, wx, wy, wz]
            omega_quat = np.array([0, gyro_rad[0], gyro_rad[1], gyro_rad[2]])
            q_dot = 0.5 * \
                quaternion_multiply(self.orientation_quat, omega_quat)

            # Интегрирование: q_new = q_old + q_dot * dt
            integrated_quat = self.orientation_quat + q_dot * dt
            integrated_quat = integrated_quat / \
                np.linalg.norm(integrated_quat)  # Нормализация

            # d) Комплементарный фильтр
            # Преобразуем углы из акселерометра/магнитометра в кватернион
            # acc_quat = quaternion_from_euler(acc_roll, acc_pitch, acc_yaw)

            # Фильтрация: q_fused = alpha * q_integrated + (1 - alpha) * q_accel_mag
            # Для кватернионов это немного сложнее, используем сферическую линейную интерполяцию (SLERP)
            # Но для простоты и соответствия описанию ТЗ (Euler angles fusion),
            # преобразуем в углы, смешиваем, обратно в кватернион.
            # Это может быть менее устойчиво при больших поворотах.

            # Преобразуем интегрированный кватернион в углы
            int_roll, int_pitch, int_yaw = euler_from_quaternion(
                integrated_quat)

            # Смешиваем углы
            fused_roll = self.alpha * int_roll + (1 - self.alpha) * acc_roll
            fused_pitch = self.alpha * int_pitch + (1 - self.alpha) * acc_pitch
            # Для yaw часто используется более сложная логика из-за разрывов на 360 градусах,
            # но для простоты применим тот же подход.
            # Нужно быть осторожным с разрывами, например, от -179 до +179.
            # Простое усреднение может дать неверный результат.
            # Используем более устойчивый способ для yaw:
            delta_yaw = acc_yaw - int_yaw
            # Приводим delta_yaw к диапазону [-pi, pi]
            while delta_yaw > math.pi:
                delta_yaw -= 2 * math.pi
            while delta_yaw < -math.pi:
                delta_yaw += 2 * math.pi
            fused_yaw = int_yaw + (1 - self.alpha) * delta_yaw

            # Преобразуем обратно в кватернион
            self.orientation_quat = quaternion_from_euler(
                fused_roll, fused_pitch, fused_yaw)
            # Нормализуем на всякий случай
            self.orientation_quat = self.orientation_quat / \
                np.linalg.norm(self.orientation_quat)

            # Обновляем время
            self.last_timestamp_s = timestamp_s

            # --- 4. Формирование результата ---
            # fused_roll, fused_pitch, fused_yaw уже в радианах
            return {
                "timestamp": timestamp_ms,
                # Возвращаем в градусах как в ТЗ
                "roll": math.degrees(fused_roll),
                "pitch": math.degrees(fused_pitch),
                "yaw": math.degrees(fused_yaw),
                "quat_w": float(self.orientation_quat[0]),
                "quat_x": float(self.orientation_quat[1]),
                "quat_y": float(self.orientation_quat[2]),
                "quat_z": float(self.orientation_quat[3]),
            }

        except Exception as e:
            self.logger.error(f"Ошибка обработки IMU сэмпла: {e}")
            return None
