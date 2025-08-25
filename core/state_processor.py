import time
import threading
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from ..services.logger_service import LoggerService

class StateProcessor:
    """
    Управление состоянием модуля позиционирования.
    Отслеживает текущее состояние, вычисляет confidence,
    управляет переходами между состояниями.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация процессора состояний.
        
        Args:
            config (dict): Конфигурация модуля
        """
        self.logger = LoggerService.get_logger(__name__)
        self.config = config
        
        # Текущее состояние модуля
        self._current_state = "BOOTING"
        self._previous_state = None
        self._state_timestamp = int(time.time() * 1000)
        
        # Временные метки последних данных
        self._last_gps_time = None
        self._last_imu_time = None
        self._last_valid_position_time = None
        
        # Отслеживание состояния GPS датчиков
        self._gps_last_timestamps = {}  # sensor_id -> timestamp
        self._gps_configured_sensors = self._get_configured_gps_sensors()
        
        # Флаг наличия IMU данных
        self._has_imu_data = False
        
        # Статус транспортного средства
        self._vehicle_status = "off"
        
        # Блокировка для потокобезопасности
        self._lock = threading.RLock()
        
        self.logger.info("StateProcessor инициализирован")
    
    def _get_configured_gps_sensors(self) -> List[int]:
        """Получение списка сконфигурированных GPS датчиков"""
        gps_config = self.config.get("gps", {}).get("sensors", [])
        return [sensor.get("id") for sensor in gps_config]
    
    @property
    def current_state(self) -> str:
        """Текущее состояние модуля"""
        with self._lock:
            return self._current_state
    
    @property
    def vehicle_status(self) -> str:
        """Текущий статус транспортного средства"""
        with self._lock:
            return self._vehicle_status
    
    @property
    def has_imu_data(self) -> bool:
        """Флаг наличия данных от IMU"""
        with self._lock:
            return self._has_imu_data
    
    def set_imu_data_available(self, available: bool):
        """
        Установка флага наличия данных от IMU.
        
        Args:
            available (bool): True если IMU данные доступны
        """
        with self._lock:
            self._has_imu_data = available
            self._update_state()
    
    def update_gps_timestamp(self, sensor_id: int, timestamp: int):
        """
        Обновление временной метки последнего получения данных от GPS датчика.
        
        Args:
            sensor_id (int): ID GPS датчика
            timestamp (int): Временная метка в миллисекундах
        """
        with self._lock:
            self._gps_last_timestamps[sensor_id] = timestamp
            self._last_gps_time = timestamp
            self._update_state()
    
    def update_imu_timestamp(self, timestamp: int):
        """
        Обновление временной метки последнего получения данных от IMU.
        
        Args:
            timestamp (int): Временная метка в миллисекундах
        """
        with self._lock:
            self._last_imu_time = timestamp
            # Отмечаем, что IMU данные получены
            if not self._has_imu_data:
                self._has_imu_data = True
            self._update_state()
    
    def update_valid_position_timestamp(self, timestamp: int):
        """
        Обновление временной метки последней валидной позиции.
        
        Args:
            timestamp (int): Временная метка в миллисекундах
        """
        with self._lock:
            self._last_valid_position_time = timestamp
            self._update_state()
    
    def _get_active_gps_sensors(self, time_threshold_ms: int = 2000) -> List[int]:
        """
        Получение списка активных GPS датчиков.
        
        Args:
            time_threshold_ms (int): Порог времени для определения активности (мс)
            
        Returns:
            List[int]: Список ID активных датчиков
        """
        current_time = int(time.time() * 1000)
        active_sensors = []
        
        for sensor_id, last_timestamp in self._gps_last_timestamps.items():
            if last_timestamp and (current_time - last_timestamp) <= time_threshold_ms:
                active_sensors.append(sensor_id)
        
        return active_sensors
    
    def _update_state(self):
        """Обновление состояния модуля на основе текущих данных"""
        with self._lock:
            previous_state = self._current_state
            self._previous_state = previous_state
            
            # Определение наличия данных
            current_time = int(time.time() * 1000)
            has_recent_gps = self._has_recent_data(self._last_gps_time, current_time, 2000)  # 2 секунды
            has_recent_imu = self._has_imu_data and self._has_recent_data(self._last_imu_time, current_time, 100)   # 100 мс
            
            # Получение активных GPS датчиков
            active_gps_sensors = self._get_active_gps_sensors()
            num_active_gps = len(active_gps_sensors)
            num_configured_gps = len(self._gps_configured_sensors)
            
            # Определение статуса транспортного средства
            self._vehicle_status = self._determine_vehicle_status(current_time)
            
            # Определение состояния модуля
            if self._current_state == "CALIBRATING_MAG":
                # Не меняем состояние во время калибровки
                pass
            elif self._vehicle_status == "off":
                self._current_state = "POWER_OFF"
            elif self._is_data_stale(current_time):
                self._current_state = "DEGRADED"
            elif has_recent_gps and has_recent_imu:
                self._current_state = "GPS_IMU_FUSION"
            elif has_recent_imu:
                self._current_state = "IMU_DEAD_RECKONING"
            elif has_recent_gps:
                # Проверяем конфигурацию GPS
                if num_configured_gps >= 2:
                    if num_active_gps >= 1:
                        self._current_state = "GPS_ONLY"
                        if num_active_gps < num_configured_gps:
                            self.logger.info(f"Работает {num_active_gps} из {num_configured_gps} GPS датчиков: {active_gps_sensors}")
                    else:
                        self._current_state = "STANDBY"
                else:
                    # Один датчик сконфигурирован
                    if num_active_gps >= 1:
                        self._current_state = "GPS_ONLY"
                    else:
                        self._current_state = "STANDBY"
            elif self._last_gps_time is None and self._last_imu_time is None:
                self._current_state = "BOOTING"
            else:
                self._current_state = "STANDBY"
            
            # Обновление временной метки состояния
            self._state_timestamp = current_time
            
            # Логирование изменения состояния
            if previous_state != self._current_state:
                self.logger.info(f"Состояние модуля изменено: {previous_state} -> {self._current_state}")
                if num_configured_gps >= 2 and num_active_gps < num_configured_gps:
                    self.logger.info(f"GPS датчики: активные {active_gps_sensors}, всего {num_configured_gps}")
    
    def _has_recent_data(self, last_time: Optional[int], current_time: int, threshold_ms: int) -> bool:
        """
        Проверка, были ли получены данные недавно.
        
        Args:
            last_time (int): Время последних данных
            current_time (int): Текущее время
            threshold_ms (int): Порог в миллисекундах
            
        Returns:
            bool: True если данные свежие
        """
        if last_time is None:
            return False
        return (current_time - last_time) <= threshold_ms
    
    def _determine_vehicle_status(self, current_time: int) -> str:
        """
        Определение статуса транспортного средства.
        
        Args:
            current_time (int): Текущее время в миллисекундах
            
        Returns:
            str: Статус транспортного средства
        """
        # Проверка времени последней активности
        last_activity_time = max(
            filter(None, [self._last_gps_time, self._last_imu_time]),
            default=0
        )
        
        if last_activity_time == 0:
            return "off"
        
        inactivity_time = current_time - last_activity_time
        
        off_timeout = self.config.get("power_management", {}).get("inactivity_timeout_to_off_sec", 300) * 1000
        standby_timeout = self.config.get("power_management", {}).get("inactivity_timeout_to_standby_sec", 30) * 1000
        
        if inactivity_time > off_timeout:
            return "off"
        elif inactivity_time > standby_timeout:
            return "standby"
        else:
            return "running"
    
    def _is_data_stale(self, current_time: int) -> bool:
        """
        Проверка, устарели ли данные.
        
        Args:
            current_time (int): Текущее время в миллисекундах
            
        Returns:
            bool: True если данные устарели
        """
        if self._last_valid_position_time is None:
            return True
        
        stale_threshold = self._get_stale_threshold()
        time_since_last_valid = current_time - self._last_valid_position_time
        
        return time_since_last_valid > stale_threshold
    
    def _get_stale_threshold(self) -> int:
        """
        Получение порога устаревания данных в зависимости от текущего состояния.
        
        Returns:
            int: Порог в миллисекундах
        """
        has_imu = self._has_imu_data and self._has_recent_data(self._last_imu_time, int(time.time() * 1000), 100)
        
        if has_imu:
            # С IMU данные считаются устаревшими через 2 минуты
            return self.config.get("extrapolation", {}).get("max_imu_time_sec", 120) * 1000
        else:
            # Без IMU данные считаются устаревшими через 15 секунд
            return self.config.get("extrapolation", {}).get("max_gps_only_time_sec", 15) * 1000
    
    def calculate_confidence(self) -> float:
        """
        Расчёт показателя достоверности (confidence) на основе текущего состояния.
        
        Returns:
            float: Достоверность от 0.0 до 1.0
        """
        with self._lock:
            current_time = int(time.time() * 1000)
            time_since_last_gps = 0
            
            if self._last_gps_time:
                time_since_last_gps = (current_time - self._last_gps_time) / 1000.0  # в секундах
            
            # Получение информации об активных GPS датчиках
            active_gps_sensors = self._get_active_gps_sensors()
            num_active_gps = len(active_gps_sensors)
            num_configured_gps = len(self._gps_configured_sensors)
            
            # Расчёт confidence в зависимости от состояния
            if self._current_state == "BOOTING":
                return 0.0
            elif self._current_state == "GPS_IMU_FUSION":
                # 95% - (время с последнего GPS * 1%)
                confidence = 0.95 - (time_since_last_gps * 0.01)
                # Корректировка на основе количества активных GPS датчиков
                if num_configured_gps >= 2 and num_active_gps < 2:
                    confidence *= 0.9  # Штраф за отсутствие одного датчика
                return max(0.0, min(1.0, confidence))
            elif self._current_state == "IMU_DEAD_RECKONING":
                # 70% - (время с последнего GPS * 5%)
                confidence = 0.70 - (time_since_last_gps * 0.05)
                return max(0.0, min(1.0, confidence))
            elif self._current_state == "GPS_ONLY":
                base_confidence = 0.80
                # Корректировка на основе количества активных GPS датчиков
                if num_configured_gps >= 2:
                    if num_active_gps == 2:
                        confidence = base_confidence  # Полная конфиденциальность
                    elif num_active_gps == 1:
                        confidence = base_confidence * 0.7  # Сниженная конфиденциальность
                    else:
                        confidence = 0.0  # Нет активных датчиков
                else:
                    confidence = base_confidence if num_active_gps > 0 else 0.0
                return max(0.0, min(1.0, confidence))
            elif self._current_state == "STANDBY":
                # 85% - (время с последнего GPS * 0.5%)
                confidence = 0.85 - (time_since_last_gps * 0.005)
                # Корректировка на основе количества активных GPS датчиков
                if num_configured_gps >= 2 and num_active_gps < 2:
                    confidence *= 0.8  # Штраф за отсутствие одного датчика
                return max(0.0, min(1.0, confidence))
            elif self._current_state == "CALIBRATING_MAG":
                return 0.0
            elif self._current_state == "POWER_OFF":
                return 0.0
            elif self._current_state == "DEGRADED":
                return 0.0
            else:
                return 0.0
    
    def get_state_info(self) -> Dict[str, Any]:
        """
        Получение полной информации о состоянии модуля.
        
        Returns:
            dict: Информация о состоянии
        """
        with self._lock:
            active_gps = self._get_active_gps_sensors()
            return {
                "current_state": self._current_state,
                "previous_state": self._previous_state,
                "state_timestamp": self._state_timestamp,
                "vehicle_status": self._vehicle_status,
                "confidence": self.calculate_confidence(),
                "last_gps_time": self._last_gps_time,
                "last_imu_time": self._last_imu_time,
                "last_valid_position_time": self._last_valid_position_time,
                "has_imu_data": self._has_imu_data,
                "active_gps_sensors": active_gps,
                "configured_gps_sensors": self._gps_configured_sensors,
                "gps_sensor_status": {
                    sensor_id: {
                        "last_timestamp": self._gps_last_timestamps.get(sensor_id),
                        "is_active": sensor_id in active_gps
                    } for sensor_id in self._gps_configured_sensors
                }
            }
    
    def set_calibration_state(self, is_calibrating: bool):
        """
        Установка состояния калибровки.
        
        Args:
            is_calibrating (bool): True если идет калибровка
        """
        with self._lock:
            if is_calibrating:
                # Проверяем наличие IMU данных перед началом калибровки
                if not self._has_imu_data:
                    self.logger.warning("Невозможно начать калибровку: нет данных от IMU")
                    return False
                self._current_state = "CALIBRATING_MAG"
            else:
                # Возвращаем предыдущее состояние или определяем новое
                self._update_state()
            return True
    
    def reset(self):
        """Сброс состояния модуля"""
        with self._lock:
            self._current_state = "BOOTING"
            self._previous_state = None
            self._state_timestamp = int(time.time() * 1000)
            self._last_gps_time = None
            self._last_imu_time = None
            self._last_valid_position_time = None
            self._vehicle_status = "off"
            self._gps_last_timestamps = {}
            # Не сбрасываем _has_imu_data и _gps_configured_sensors, так как это свойства системы
            
            self.logger.info("StateProcessor сброшен")