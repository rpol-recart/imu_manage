# /calibration/calibration_manager.py

import time
import json
import threading
import math
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import logging

# Предполагается, что LoggerService находится в ../services/logger_service.py
# и что DatabaseManager передается в конструктор.
# from ..services.logger_service import LoggerService

# Для демонстрации используем стандартный logging.
# В реальной реализации следует заменить на LoggerService.
import logging
LoggerService = logging


class CalibrationManager:
    """
    Управление калибровкой датчиков, особенно магнитометра и гироскопа.
    Отвечает за запуск процедур калибровки, сбор данных, расчет коэффициентов
    и сохранение результатов в базу данных.
    """

    def __init__(self, config: Dict[str, Any], database_manager):
        """
        Инициализация менеджера калибровки.

        Args:
            config (dict): Конфигурация модуля из config.json.
            database_manager: Экземпляр DatabaseManager для сохранения калибровочных данных.
        """
        self.logger = LoggerService.get_logger(__name__)
        self.config = config
        self.db_manager = database_manager

        # Для хранения данных калибровки магнитометра во время сбора
        self._mag_calibration_data: List[Tuple[float, float, float]] = []
        self._mag_calibration_lock = threading.Lock()
        self._is_mag_calibrating = False
        self._mag_calibration_start_time: Optional[float] = None

        self.logger.info("CalibrationManager инициализирован.")

    def _calculate_hard_iron_offset(self, data_points: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
        """
        Вычисляет смещение жесткой железа (hard iron offset) как центр эллипсоида.
        Простой метод: находит средние минимум и максимум по каждой оси.
        """
        if len(data_points) < 10:  # Минимальное количество точек
            self.logger.warning(
                "Недостаточно точек данных для калибровки магнитометра.")
            return 0.0, 0.0, 0.0

        try:
            x_vals = [p[0] for p in data_points]
            y_vals = [p[1] for p in data_points]
            z_vals = [p[2] for p in data_points]

            x_offset = (min(x_vals) + max(x_vals)) / 2.0
            y_offset = (min(y_vals) + max(y_vals)) / 2.0
            z_offset = (min(z_vals) + max(z_vals)) / 2.0

            return x_offset, y_offset, z_offset
        except Exception as e:
            self.logger.error(f"Ошибка при расчете hard iron offset: {e}")
            return 0.0, 0.0, 0.0

    # def _calculate_ellipsoid_fit(self, data_points: List[Tuple[float, float, float]]) -> Dict[str, Any]:
    #     """
    #     (Опционально) Более сложный метод подгонки эллипсоида для soft iron калибровки.
    #     Требует библиотеки типа numpy/scipy.
    #     """
    #     # Реализация метода наименьших квадратов для подгонки эллипсоида.
    #     # Это сложнее и требует дополнительных зависимостей.
    #     # Возвращаемый словарь может содержать offset (hard iron) и scale/rotation (soft iron).
    #     pass

    def perform_magnetometer_calibration(self, duration_sec: int = 120) -> Dict[str, Any]:
        """
        Запускает процедуру калибровки магнитометра.
        Собирает данные в течение заданного времени.

        Args:
            duration_sec (int): Длительность сбора данных в секундах.

        Returns:
            dict: Результаты калибровки (коэффициенты) или пустой словарь при ошибке.
        """
        self.logger.info(
            f"Начало калибровки магнитометра на {duration_sec} секунд.")

        with self._mag_calibration_lock:
            if self._is_mag_calibrating:
                self.logger.warning("Калибровка магнитометра уже выполняется.")
                return {}

            self._is_mag_calibrating = True
            self._mag_calibration_data = []
            self._mag_calibration_start_time = time.time()

        try:
            # Ожидание завершения сбора данных
            time.sleep(duration_sec)

            # После ожидания, данные должны были быть собраны
            # обработчиком IMU (например, в update_imu, если состояние CALIBRATING_MAG)
            # или через отдельный callback. Для простоты, предположим,
            # что данные уже собраны в self._mag_calibration_data.

            with self._mag_calibration_lock:
                if not self._is_mag_calibrating:  # Может быть сброшено вручную или по ошибке
                    self.logger.warning(
                        "Калибровка магнитометра была прервана.")
                    return {}

                collected_data = self._mag_calibration_data.copy()
                self._is_mag_calibrating = False
                self._mag_calibration_data = []
                self._mag_calibration_start_time = None

            self.logger.info(
                f"Собрано {len(collected_data)} точек данных для калибровки магнитометра.")

            # Вычисление калибровочных коэффициентов
            # Hard iron offset
            x_offset, y_offset, z_offset = self._calculate_hard_iron_offset(
                collected_data)

            calibration_result = {
                "hard_iron_offset": {
                    "x": x_offset,
                    "y": y_offset,
                    "z": z_offset
                },
                # "soft_iron_matrix": {...} # Если бы использовался _calculate_ellipsoid_fit
            }

            # Сохранение результатов в БД
            self.save_calibration("imu_mag_hard_iron", calibration_result)
            self.logger.info(
                "Калибровка магнитометра завершена и данные сохранены.")
            return calibration_result

        except Exception as e:
            self.logger.error(f"Ошибка во время калибровки магнитометра: {e}")
            with self._mag_calibration_lock:
                self._is_mag_calibrating = False
                self._mag_calibration_data = []
                self._mag_calibration_start_time = None
            return {}

    def is_mag_calibrating(self) -> bool:
        """Проверяет, идет ли в данный момент калибровка магнитометра."""
        with self._mag_calibration_lock:
            return self._is_mag_calibrating

    def add_magnetometer_sample(self, mag_x: float, mag_y: float, mag_z: float):
        """
        (Вызывается извне, например, из IMUDataHandler)
        Добавляет один отсчет магнитометра в буфер для калибровки.
        """
        if not self.is_mag_calibrating():
            return

        with self._mag_calibration_lock:
            # Дополнительная проверка внутри блокировки
            if not self._is_mag_calibrating:
                return
            self._mag_calibration_data.append((mag_x, mag_y, mag_z))
            # Ограничение размера буфера для предотвращения переполнения памяти
            # при очень длинной калибровке или высокой частоте данных.
            max_samples = self.config.get("imu", {}).get(
                "calibration", {}).get("max_calibration_samples", 10000)
            if len(self._mag_calibration_data) > max_samples:
                self.logger.warning(
                    "Превышен лимит точек данных для калибровки магнитометра. Очистка буфера.")
                self._mag_calibration_data = self._mag_calibration_data[-max_samples:]

    def save_calibration(self, sensor_name: str, parameters: Dict[str, Any]):
        """
        Сохраняет калибровочные параметры через DatabaseManager.save_calibration.
        """
        try:
            timestamp_ms = int(time.time() * 1000)
            # Исправлено: используем правильный метод
            self.db_manager.save_calibration(
                sensor_name=sensor_name, parameters=parameters, date_calibrated=timestamp_ms)
            self.logger.debug(f"Калибровка '{sensor_name}' сохранена.")
        except Exception as e:
            self.logger.error(
                f"Ошибка сохранения калибровки '{sensor_name}': {e}")
            return False
        return True

    def load_calibration(self, sensor_name: str) -> Optional[Dict[str, Any]]:
        """
        Загружает калибровочные параметры через DatabaseManager.get_calibration.
        """
        try:
            # Исправлено: используем единый интерфейс
            result = self.db_manager.get_calibration(sensor_name)
            if result is not None:
                return result  # Уже dict, т.к. DatabaseManager возвращает распарсенные данные
            self.logger.debug(f"Калибровка для '{sensor_name}' не найдена.")
            return None
        except Exception as e:
            self.logger.error(
                f"Ошибка загрузки калибровки '{sensor_name}': {e}")
            return None

    def perform_auto_gyro_calibration(self, imu_data_samples: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
        """
        Выполняет автоматическую калибровку смещения гироскопа (bias) на основе
        набора образцов данных, когда транспортное средство неподвижно.

        Args:
            imu_data_samples (list): Список словарей с данными IMU (должны содержать 'gyroscope' и 'timestamp').

        Returns:
            dict | None: Словарь со смещениями по осям {'x': bias_x, 'y': bias_y, 'z': bias_z} или None при ошибке.
        """
        if not imu_data_samples:
            self.logger.warning("Нет данных IMU для калибровки гироскопа.")
            return None

        required_still_time_sec = self.config.get("imu", {}).get(
            "calibration", {}).get("required_still_time_sec", 15)

        # Проверка времени сбора данных
        if len(imu_data_samples) < 2:
            self.logger.warning(
                "Недостаточно данных IMU для калибровки гироскопа.")
            return None

        # Предполагается, что timestamp в мс
        start_time = imu_data_samples[0].get("timestamp", 0) / 1000.0
        end_time = imu_data_samples[-1].get("timestamp", 0) / 1000.0
        actual_duration = end_time - start_time

        if actual_duration < required_still_time_sec:
            self.logger.info(
                f"Недостаточное время неподвижности для калибровки гироскопа: {actual_duration:.2f}с < {required_still_time_sec}с")
            return None

        self.logger.info(
            f"Начало автоматической калибровки гироскопа. Время сбора: {actual_duration:.2f}с")

        try:
            # Сбор данных гироскопа
            gyro_x_values = [sample['gyroscope']['x']
                             for sample in imu_data_samples if 'gyroscope' in sample and 'x' in sample['gyroscope']]
            gyro_y_values = [sample['gyroscope']['y']
                             for sample in imu_data_samples if 'gyroscope' in sample and 'y' in sample['gyroscope']]
            gyro_z_values = [sample['gyroscope']['z']
                             for sample in imu_data_samples if 'gyroscope' in sample and 'z' in sample['gyroscope']]

            if not all([gyro_x_values, gyro_y_values, gyro_z_values]):
                self.logger.error(
                    "Ошибка: Неполные данные гироскопа в образцах.")
                return None

            # Расчет среднего значения (bias)
            bias_x = sum(gyro_x_values) / len(gyro_x_values)
            bias_y = sum(gyro_y_values) / len(gyro_y_values)
            bias_z = sum(gyro_z_values) / len(gyro_z_values)

            gyro_bias = {
                "x": bias_x,
                "y": bias_y,
                "z": bias_z
            }

            # Сохранение результатов
            self.save_calibration("imu_gyro_bias", gyro_bias)
            self.logger.info(
                f"Автоматическая калибровка гироскопа завершена. Bias: X={bias_x:.4f}, Y={bias_y:.4f}, Z={bias_z:.4f}")
            return gyro_bias

        except Exception as e:
            self.logger.error(
                f"Ошибка при автоматической калибровке гироскопа: {e}")
            return None

    def get_calibration_status(self) -> Dict[str, Any]:
        """
        Возвращает текущий статус калибровки.

        Returns:
            dict: Информация о статусе калибровки.
        """
        with self._mag_calibration_lock:
            return {
                "mag_calibrating": self._is_mag_calibrating,
                "mag_calibration_points": len(self._mag_calibration_data) if self._is_mag_calibrating else 0,
                "mag_calibration_elapsed_sec": (time.time() - self.mag_calibration_start_time) if self._mag_calibration_start_time else 0
            }
