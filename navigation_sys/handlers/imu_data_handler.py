# /handlers/imu_data_handler.py

import time
import json
import math
from typing import Dict, Any, Optional
from collections import deque
from ..services.logger_service import LoggerService
from ..storage.database_manager import DatabaseManager


class IMUDataHandler:
    """
    Приём, валидация и сохранение данных от IMU-датчика.
    Расчёт простых статистик для фильтрации и поддержания истории.
    """

    def __init__(self, config: Dict[str, Any], database_manager: DatabaseManager):
        """
        Инициализация обработчика IMU данных.

        Args:
            config (dict): Конфигурация модуля.
            database_manager (DatabaseManager): Менеджер базы данных для записи.
        """
        self.logger = LoggerService.get_logger(self.__class__.__name__)
        self.config = config
        self.db_manager = database_manager

        # Кэширование последних данных
        self._last_valid_data = None  # Последние валидные сырые данные
        self._last_timestamp = None

        # Для определения состояния покоя
        # Окно для накопления данных об ускорении
        self._stillness_window = deque(maxlen=100)

        # Извлечение конфигурации IMU
        self.imu_config = self.config.get("imu", {})

        self.logger.info("IMUDataHandler инициализирован")

    def process_imu_data(self, imu_data: Dict[str, Any]):
        """
        Приём, валидация и сохранение данных от IMU-датчика.

        Args:
            imu_data (dict): Данные IMU.
        """
        try:
            timestamp = imu_data.get("timestamp")
            if not timestamp:
                self.logger.warning(
                    "IMU данные без временной метки, отбракованы.")
                return

            # 1. Валидация данных
            validation_result = self._validate_data(imu_data)
            if not validation_result["valid"]:
                self.logger.warning(
                    f"IMU данные отбракованы: {validation_result['reason']}")
                return  # Не обрабатываем невалидные данные

            # 2. Подготовка структуры данных для хранения
            # Сохраняем сырые данные
            raw_data_to_store = {
                k: v for k, v in imu_data.items()
                if k in ["accelerometer", "gyroscope", "magnetometer", "temperature", "status"]
            }

            # 3. Запись в базу данных
            raw_data_json = json.dumps(raw_data_to_store)
            self.db_manager.add_sensor_data(
                timestamp, "imu_raw", None, raw_data_json)

            # 4. Обновление внутреннего кэша
            self._last_valid_data = imu_data
            # Добавляем timestamp в кэш
            self._last_valid_data["timestamp"] = timestamp
            self._last_timestamp = timestamp

            # 5. Обновление данных для определения покоя
            self._update_stillness_data(imu_data)

            self.logger.debug(f"IMU сырые данные обработаны и сохранены.")

        except Exception as e:
            self.logger.error(
                f"Ошибка обработки IMU данных (ts: {imu_data.get('timestamp')}): {e}", exc_info=True)

    def _validate_data(self, imu_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Валидация входных IMU данных согласно ТЗ.

        Args:
            imu_data (dict): Данные IMU.

        Returns:
            dict: Словарь с ключами 'valid' (bool) и 'reason' (str).
        """
        # Проверка на NaN/inf
        for sensor_type in ["accelerometer", "gyroscope", "magnetometer"]:
            sensor_data = imu_data.get(sensor_type)
            if sensor_data:
                for axis, value in sensor_data.items():
                    if not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
                        return {"valid": False, "reason": f"{sensor_type}.{axis} содержит NaN или Inf"}

        # Проверка температуры
        temp = imu_data.get("temperature")
        if temp is not None and temp > 80:
            return {"valid": False, "reason": f"Перегрев IMU (температура {temp}°C > 80°C)"}

        # Проверка диапазонов (примерные, могут быть в конфиге)
        # accel = imu_data.get("accelerometer", {})
        # gyro = imu_data.get("gyroscope", {})
        # Проверки диапазонов опущены для краткости, но логика аналогична GPS

        # Проверка статуса датчика
        status = imu_data.get("status")
        if status is not None and status != "ok":
            return {"valid": False, "reason": f"Статус датчика '{status}'"}

        return {"valid": True, "reason": "OK"}

    def has_recent_data(self, time_threshold_ms: int = 200) -> bool:
        """
        Проверяет, были ли получены недавние данные от IMU.

        Args:
            time_threshold_ms (int): Порог времени в миллисекундах.

        Returns:
            bool: True, если данные свежие.
        """
        if not self._last_timestamp:
            return False
        current_time_ms = int(time.time() * 1000)
        return (current_time_ms - self._last_timestamp) <= time_threshold_ms

    def get_last_valid_data(self) -> Optional[Dict[str, Any]]:
        """
        Получает последние валидные сырые данные.

        Returns:
            dict or None: Сырые данные или None.
        """
        # Возвращаем копию, чтобы избежать изменений извне
        return self._last_valid_data.copy() if self._last_valid_data else None

    def get_last_timestamp(self) -> Optional[int]:
        """
        Получает временную метку последних данных.

        Returns:
            int or None: Временная метка или None.
        """
        return self._last_timestamp

    def _update_stillness_data(self, imu_data: Dict[str, Any]):
        """
        Обновляет данные для определения состояния покоя транспорта.

        Args:
            imu_data (dict): Сырые данные IMU.
        """
        accel_data = imu_data.get("accelerometer")
        if not accel_data:
            return

        timestamp = imu_data.get("timestamp")
        if not timestamp:
            return

        # Рассчитываем модуль ускорения (без учета гравитации, приблизительно)
        ax, ay, az = accel_data.get("x", 0), accel_data.get(
            "y", 0), accel_data.get("z", 0)
        # Вычитаем примерное значение g (9.81 м/с^2) из Z-компоненты, если датчик ориентирован стандартно
        magnitude = math.sqrt(ax**2 + ay**2 + (az - 9.81)**2)
        self._stillness_window.append((timestamp, magnitude))

    def is_vehicle_still(self, window_duration_sec: float = 2.0, magnitude_threshold: float = 0.5) -> bool:
        """
        Определяет, неподвижен ли транспорт, анализируя данные акселерометра.

        Args:
            window_duration_sec (float): Длительность окна анализа (сек).
            magnitude_threshold (float): Порог модуля ускорения для определения покоя (м/с^2).

        Returns:
            bool: True, если транспорт неподвижен.
        """
        if len(self._stillness_window) < 2:
            return False  # Недостаточно данных

        current_time_ms = int(time.time() * 1000)
        window_start_time_ms = current_time_ms - \
            int(window_duration_sec * 1000)

        magnitudes_in_window = [
            mag for ts, mag in self._stillness_window if ts >= window_start_time_ms
        ]

        if not magnitudes_in_window:
            return False  # Нет данных в окне

        avg_magnitude = sum(magnitudes_in_window) / len(magnitudes_in_window)
        self.logger.debug(
            f"Среднее ускорение за последние {window_duration_sec}с: {avg_magnitude:.4f} м/с^2")
        return avg_magnitude < magnitude_threshold

    def is_vehicle_moving(self, threshold_accel_magnitude: float = 1.0) -> bool:
        """
        Определяет, движется ли транспорт, основываясь на данных IMU.

        Args:
            threshold_accel_magnitude (float): Порог модуля ускорения для определения движения (м/с^2).

        Returns:
            bool: True, если транспорт движется.
        """
        # Простая проверка: если есть недавние данные и среднее ускорение выше порога
        if not self._stillness_window:
            return False

        # Берем последнее значение из окна
        _, last_magnitude = self._stillness_window[-1]
        return last_magnitude > threshold_accel_magnitude

    def calculate_gyro_bias(self, window_duration_sec: float = 15.0) -> Dict[str, float]:
        """
        Рассчитывает смещение (bias) гироскопа, когда транспорт неподвижен.

        Args:
            window_duration_sec (float): Длительность окна анализа (сек).

        Returns:
            dict: Словарь смещений по осям {"x": bias_x, "y": bias_y, "z": bias_z}.
        """
        if len(self._stillness_window) < 2:
            self.logger.warning(
                "Недостаточно данных для расчета bias гироскопа.")
            return {"x": 0.0, "y": 0.0, "z": 0.0}

        # Проверяем, неподвижен ли транспорт
        # Более строгий порог для калибровки
        if not self.is_vehicle_still(window_duration_sec, 0.1):
            self.logger.warning(
                "Транспорт не неподвижен, расчет bias гироскопа пропущен.")
            return {"x": 0.0, "y": 0.0, "z": 0.0}

        # current_time_ms = int(time.time() * 1000)
        # window_start_time_ms = current_time_ms - \
        #    int(window_duration_sec * 1000)

        # Для точного расчета нужно хранить сырые данные IMU.
        # В текущей реализации упрощаем: если транспорт неподвижен,
        # то предполагаем, что среднее значение гироскопа = bias.
        # Это требует хранения последних N сырых данных IMU.
        # TODO: Хранить очередь сырых IMU данных для точного расчета bias.

        # Заглушка: возвращаем нули, так как сырые данные гироскопа не хранятся в _stillness_window.
        # Реализация будет в CalibrationManager, который будет запрашивать сырые данные из БД.
        self.logger.warning(
            "Расчет gyro bias требует буфера сырых данных гироскопа, возвращаются нули.")
        return {"x": 0.0, "y": 0.0, "z": 0.0}
