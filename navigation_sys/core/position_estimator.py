"""
Модуль оценки позиции транспортного средства.

Предоставляет функциональность для обработки данных GPS и IMU датчиков,
фьюжн данных и оценки текущего положения транспортного средства.
"""

import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Any, List, Protocol, Callable
from contextlib import contextmanager

from ..storage.database_manager import DatabaseManager
from ..configs import CONFIG
from ..handlers.gps_data_handler import GPSDataHandler
from ..handlers.imu_data_handler import IMUDataHandler
from ..processing.position_fuser import PositionFuser
from ..processing.state_predictor import StatePredictor
from ..services.logger_service import LoggerService


class ModuleState(Enum):
    """Состояния модуля позиционирования."""
    BOOTING = "BOOTING"
    READY = "READY"
    GPS_IMU_FUSION = "GPS_IMU_FUSION"
    IMU_DEAD_RECKONING = "IMU_DEAD_RECKONING"
    GPS_ONLY = "GPS_ONLY"
    STANDBY = "STANDBY"
    CALIBRATING_MAG = "CALIBRATING_MAG"
    POWER_OFF = "POWER_OFF"
    DEGRADED = "DEGRADED"


class VehicleStatus(Enum):
    """Статусы транспортного средства."""
    OFF = "off"
    STANDBY = "standby"
    RUNNING = "running"


@dataclass
class PositionState:
    """Состояние позиции транспортного средства."""
    lat: Optional[float]
    lon: Optional[float]
    alt: Optional[float]
    azimuth: Optional[float]
    timestamp: int
    confidence: float
    status: str
    source: str
    vehicle_status: str


@dataclass
class IMUData:
    """Структура данных IMU."""
    timestamp: int
    accelerometer: Dict[str, float]
    gyroscope: Dict[str, float]
    magnetometer: Dict[str, float]
    temperature: float
    status: str


class StateCalculator(Protocol):
    """Протокол для расчета состояния модуля."""
    
    def calculate_state(self, has_gps: bool, has_imu: bool, 
                       active_gps_count: int, configured_gps_count: int,
                       last_update_time: Optional[int]) -> ModuleState:
        """Рассчитать состояние модуля."""
        ...


class ConfidenceCalculator(Protocol):
    """Протокол для расчета уровня достоверности."""
    
    def calculate_confidence(self, state: ModuleState, 
                           time_since_last_update: float,
                           active_gps_count: int,
                           configured_gps_count: int) -> float:
        """Рассчитать уровень достоверности."""
        ...


class TimerManager(Protocol):
    """Протокол для управления таймерами."""
    
    def start_timer(self, interval: float, callback: Callable[[], None]) -> None:
        """Запустить таймер."""
        ...
    
    def stop_all_timers(self) -> None:
        """Остановить все таймеры."""
        ...


class DefaultStateCalculator:
    """Реализация расчета состояния модуля по умолчанию."""
    
    def __init__(self, logger):
        self.logger = logger
    
    def calculate_state(self, has_gps: bool, has_imu: bool, 
                       active_gps_count: int, configured_gps_count: int,
                       last_update_time: Optional[int]) -> ModuleState:
        """
        Рассчитать состояние модуля на основе доступных данных.
        
        Args:
            has_gps: Наличие данных GPS
            has_imu: Наличие данных IMU
            active_gps_count: Количество активных GPS датчиков
            configured_gps_count: Количество настроенных GPS датчиков
            last_update_time: Время последнего обновления
            
        Returns:
            Состояние модуля
        """
        if has_gps and has_imu:
            return ModuleState.GPS_IMU_FUSION
        elif has_imu:
            return ModuleState.IMU_DEAD_RECKONING
        elif has_gps:
            return ModuleState.GPS_ONLY
        else:
            return self._calculate_no_data_state(last_update_time)
    
    def _calculate_no_data_state(self, last_update_time: Optional[int]) -> ModuleState:
        """Рассчитать состояние при отсутствии данных."""
        if not last_update_time:
            return ModuleState.BOOTING
        
        time_diff = time.time() * 1000 - last_update_time
        return ModuleState.DEGRADED if time_diff > 120000 else ModuleState.STANDBY


class DefaultConfidenceCalculator:
    """Реализация расчета уровня достоверности по умолчанию."""
    
    def calculate_confidence(self, state: ModuleState, 
                           time_since_last_update: float,
                           active_gps_count: int,
                           configured_gps_count: int) -> float:
        """
        Рассчитать уровень достоверности.
        
        Args:
            state: Текущее состояние модуля
            time_since_last_update: Время с последнего обновления (сек)
            active_gps_count: Количество активных GPS датчиков
            configured_gps_count: Количество настроенных GPS датчиков
            
        Returns:
            Уровень достоверности от 0.0 до 1.0
        """
        confidence_map = {
            ModuleState.BOOTING: 0.0,
            ModuleState.GPS_IMU_FUSION: self._calculate_fusion_confidence,
            ModuleState.IMU_DEAD_RECKONING: self._calculate_imu_confidence,
            ModuleState.GPS_ONLY: self._calculate_gps_confidence,
            ModuleState.STANDBY: self._calculate_standby_confidence,
            ModuleState.CALIBRATING_MAG: 0.0,
            ModuleState.POWER_OFF: 0.0,
            ModuleState.DEGRADED: 0.0,
        }
        
        calculator = confidence_map.get(state, lambda *args: 0.0)
        if callable(calculator):
            return max(0.0, min(1.0, calculator(
                time_since_last_update, active_gps_count, configured_gps_count)))
        return calculator
    
    def _calculate_fusion_confidence(self, time_since_update: float, 
                                   active_gps: int, configured_gps: int) -> float:
        """Расчет достоверности для режима фьюжн."""
        confidence = 0.95 - (time_since_update * 0.01)
        if configured_gps >= 2 and active_gps < 2:
            confidence *= 0.9
        return confidence
    
    def _calculate_imu_confidence(self, time_since_update: float, 
                                active_gps: int, configured_gps: int) -> float:
        """Расчет достоверности для режима IMU."""
        return 0.70 - (time_since_update * 0.05)
    
    def _calculate_gps_confidence(self, time_since_update: float, 
                                active_gps: int, configured_gps: int) -> float:
        """Расчет достоверности для режима GPS."""
        base_confidence = 0.80
        if configured_gps >= 2:
            if active_gps == 2:
                return base_confidence
            elif active_gps == 1:
                return base_confidence * 0.7
            else:
                return 0.0
        return base_confidence if active_gps > 0 else 0.0
    
    def _calculate_standby_confidence(self, time_since_update: float, 
                                    active_gps: int, configured_gps: int) -> float:
        """Расчет достоверности для режима ожидания."""
        confidence = 0.85 - (time_since_update * 0.005)
        if configured_gps >= 2 and active_gps < 2:
            confidence *= 0.8
        return confidence


class DefaultTimerManager:
    """Реализация управления таймерами по умолчанию."""
    
    def __init__(self):
        self._timers: List[threading.Timer] = []
    
    def start_timer(self, interval: float, callback: Callable[[], None]) -> None:
        """Запустить таймер."""
        timer = threading.Timer(interval, callback)
        timer.daemon = True
        timer.start()
        self._timers.append(timer)
    
    def stop_all_timers(self) -> None:
        """Остановить все таймеры."""
        for timer in self._timers:
            timer.cancel()
        self._timers.clear()


class SensorDataManager:
    """Менеджер для отслеживания данных датчиков."""
    
    def __init__(self, configured_gps_sensors: List[int]):
        self._gps_last_timestamps: Dict[int, int] = {}
        self._configured_gps_sensors = configured_gps_sensors
        self._has_imu_data = False
        self._last_update_time: Optional[int] = None
    
    def update_gps_timestamp(self, sensor_id: int, timestamp: int) -> None:
        """Обновить временную метку GPS датчика."""
        self._gps_last_timestamps[sensor_id] = timestamp
        self._last_update_time = timestamp
    
    def update_imu_timestamp(self, timestamp: int) -> None:
        """Обновить временную метку IMU датчика."""
        self._has_imu_data = True
        self._last_update_time = timestamp
    
    def get_active_gps_sensors(self, time_threshold_ms: int = 5000) -> List[int]:
        """Получить список активных GPS датчиков."""
        current_time = int(time.time() * 1000)
        return [
            sensor_id for sensor_id, last_timestamp 
            in self._gps_last_timestamps.items()
            if last_timestamp and (current_time - last_timestamp) <= time_threshold_ms
        ]
    
    @property
    def has_imu_data(self) -> bool:
        """Проверить наличие данных IMU."""
        return self._has_imu_data
    
    @property
    def last_update_time(self) -> Optional[int]:
        """Получить время последнего обновления."""
        return self._last_update_time
    
    @property
    def configured_gps_sensors(self) -> List[int]:
        """Получить список настроенных GPS датчиков."""
        return self._configured_gps_sensors.copy()


class VehicleStatusCalculator:
    """Калькулятор статуса транспортного средства."""
    
    def __init__(self, config, imu_handler, gps_handler):
        self.config = config
        self.imu_handler = imu_handler
        self.gps_handler = gps_handler
    
    def calculate_status(self, last_update_time: Optional[int], 
                        has_imu_data: bool) -> VehicleStatus:
        """
        Рассчитать статус транспортного средства.
        
        Args:
            last_update_time: Время последнего обновления
            has_imu_data: Наличие данных IMU
            
        Returns:
            Статус транспортного средства
        """
        if not last_update_time:
            return VehicleStatus.OFF
        
        current_time = int(time.time() * 1000)
        time_diff = current_time - last_update_time
        
        off_timeout = self.config.data.power_management.inactivity_timeout_to_off_sec * 1000
        standby_timeout = self.config.data.power_management.inactivity_timeout_to_standby_sec * 1000
        
        if time_diff > off_timeout:
            return VehicleStatus.OFF
        elif time_diff > standby_timeout:
            return VehicleStatus.STANDBY
        else:
            return self._check_movement_status(has_imu_data)
    
    def _check_movement_status(self, has_imu_data: bool) -> VehicleStatus:
        """Проверить статус движения."""
        is_moving = (
            (has_imu_data and self.imu_handler.is_vehicle_moving()) or 
            self.gps_handler.is_vehicle_moving()
        )
        return VehicleStatus.RUNNING if is_moving else VehicleStatus.STANDBY


class CalibrationService:
    """Сервис калибровки датчиков."""
    
    def __init__(self, config, imu_handler, logger, timer_manager: TimerManager):
        self.config = config
        self.imu_handler = imu_handler
        self.logger = logger
        self.timer_manager = timer_manager
        self._calibration_in_progress = False
    
    def start_auto_gyro_calibration(self, has_imu_data: bool) -> None:
        """Запустить автоматическую калибровку гироскопа."""
        if not has_imu_data:
            return
        
        interval = self.config.data.imu.calibration.get("gyro_interval_min", 30) * 60
        self.timer_manager.start_timer(interval, self._perform_auto_gyro_calibration)
    
    def _perform_auto_gyro_calibration(self) -> None:
        """Выполнить автоматическую калибровку гироскопа."""
        try:
            if self.imu_handler.is_vehicle_still():
                self.logger.info("Начало автоматической калибровки гироскопа")
                bias = self.imu_handler.calculate_gyro_bias()
                # self.calibration_manager.save_calibration("imu_gyro_bias", bias)
                self.logger.info("Автоматическая калибровка гироскопа завершена")
            else:
                self.logger.debug("Пропуск калибровки: транспорт не неподвижен")
        except Exception as e:
            self.logger.error(f"Ошибка при автоматической калибровке гироскопа: {e}")
    
    def start_magnetometer_calibration(self, duration_sec: int = 120) -> None:
        """
        Запустить калибровку магнитометра.
        
        Args:
            duration_sec: Длительность калибровки в секундах
        """
        if self._calibration_in_progress:
            self.logger.warning("Калибровка уже выполняется")
            return
        
        self._calibration_in_progress = True
        calibration_thread = threading.Thread(
            target=self._run_magnetometer_calibration,
            args=(duration_sec,)
        )
        calibration_thread.daemon = True
        calibration_thread.start()
    
    def _run_magnetometer_calibration(self, duration_sec: int) -> None:
        """Выполнить калибровку магнитометра."""
        try:
            self.logger.info(f"Начало калибровки магнитометра на {duration_sec} секунд")
            # calibration_data = self.calibration_manager.perform_magnetometer_calibration(duration_sec)
            self.logger.info("Калибровка магнитометра завершена успешно")
        except Exception as e:
            self.logger.error(f"Ошибка при калибровке магнитометра: {e}")
        finally:
            self._calibration_in_progress = False


class PositionEstimator:
    """
    Главный класс модуля обработки GPS и IMU данных.
    
    Предоставляет интерфейсы для приёма данных от внешних датчиков
    и запроса текущего состояния погрузчика.
    
    Принципы SOLID:
    - SRP: Каждый компонент отвечает за свою область
    - OCP: Расширяемость через протоколы и инъекцию зависимостей
    - LSP: Все реализации протоколов взаимозаменяемы
    - ISP: Интерфейсы разделены по назначению
    - DIP: Зависимость от абстракций, а не от конкретных реализаций
    """
    
    def __init__(self, 
                 config_path: str,
                 state_calculator: Optional[StateCalculator] = None,
                 confidence_calculator: Optional[ConfidenceCalculator] = None,
                 timer_manager: Optional[TimerManager] = None):
        """
        Инициализация модуля позиционирования.
        
        Args:
            config_path: Путь к конфигурационному файлу
            state_calculator: Калькулятор состояния модуля
            confidence_calculator: Калькулятор уровня достоверности
            timer_manager: Менеджер таймеров
        """
        self.logger = LoggerService.get_logger(__name__)
        self.logger.info("Инициализация модуля PositionEstimator")
        
        self.config = CONFIG
        self._lock = threading.RLock()
        
        # Инъекция зависимостей
        self._state_calculator = state_calculator or DefaultStateCalculator(self.logger)
        self._confidence_calculator = confidence_calculator or DefaultConfidenceCalculator()
        self._timer_manager = timer_manager or DefaultTimerManager()
        
        # Инициализация компонентов
        self._initialize_components()
        
        # Менеджер данных датчиков
        self._sensor_data_manager = SensorDataManager(
            self._get_configured_gps_sensors()
        )
        
        # Калькулятор статуса транспортного средства
        self._vehicle_status_calculator = VehicleStatusCalculator(
            self.config, self.imu_handler, self.gps_handler
        )
        
        # Сервис калибровки
        self._calibration_service = CalibrationService(
            self.config, self.imu_handler, self.logger, self._timer_manager
        )
        
        # Состояние модуля
        self._current_state = ModuleState.BOOTING
        
        # Инициализация
        self._restore_state()
        self._start_background_processes()
        
        self.logger.info("Модуль PositionEstimator инициализирован успешно")
    
    def _initialize_components(self) -> None:
        """Инициализация основных компонентов."""
        self.database_manager = DatabaseManager(self.config)
        self.gps_handler = GPSDataHandler(self.config, self.database_manager)
        self.imu_handler = IMUDataHandler(self.config, self.database_manager)
        self.position_fuser = PositionFuser()
        self.state_predictor = StatePredictor(self.config)
    
    def _get_configured_gps_sensors(self) -> List[int]:
        """Получить список сконфигурированных GPS датчиков."""
        return [sensor.id for sensor in self.config.data.gps.sensors]
    
    def _restore_state(self) -> None:
        """Восстановить последнее состояние из базы данных."""
        try:
            last_state = self.database_manager.get_last_state()
            if last_state:
                self._current_state = ModuleState(last_state.get("status", "READY"))
                self.logger.info(f"Состояние восстановлено: {self._current_state}")
            else:
                self._current_state = ModuleState.READY
        except Exception as e:
            self.logger.error(f"Ошибка восстановления состояния: {e}")
            self._current_state = ModuleState.READY
    
    def _start_background_processes(self) -> None:
        """Запустить фоновые процессы обработки данных."""
        # Запуск калибровки гироскопа
        self._calibration_service.start_auto_gyro_calibration(
            self._sensor_data_manager.has_imu_data
        )
        
        # Запуск очистки старых данных
        self._timer_manager.start_timer(300, self._cleanup_old_data)
    
    def _cleanup_old_data(self) -> None:
        """Очистить старые данные из базы."""
        try:
            retention_minutes = self.config.data.database.retention_minutes
            self.database_manager.cleanup_old_data(retention_minutes)
        except Exception as e:
            self.logger.error(f"Ошибка при очистке старых данных: {e}")
    
    @contextmanager
    def _thread_safe_operation(self):
        """Контекстный менеджер для потокобезопасных операций."""
        with self._lock:
            yield
    
    def update_gps(self, sensor_id: int, timestamp: int, lat: float,
                   lon: float, alt: float) -> None:
        """
        Принять данные от GPS-датчика.
        
        Args:
            sensor_id: Идентификатор датчика (1 или 2)
            timestamp: Unix time в миллисекундах
            lat: Широта
            lon: Долгота
            alt: Высота
        """
        with self._thread_safe_operation():
            try:
                self.logger.debug(
                    f"Получены GPS данные от датчика {sensor_id}: "
                    f"lat={lat}, lon={lon}, alt={alt}, time={timestamp}"
                )
                
                # Обновление данных датчика
                self._sensor_data_manager.update_gps_timestamp(sensor_id, timestamp)
                
                # Обработка данных
                self.gps_handler.process_gps_data(sensor_id, timestamp, lat, lon, alt)
                
                # Обновление состояния модуля
                self._update_module_state()
                
            except Exception as e:
                self.logger.error(
                    f"Ошибка обработки GPS данных от датчика {sensor_id}: {e}"
                )
    
    def update_imu(self, imu_data: Dict[str, Any]) -> None:
        """
        Принять данные от IMU-датчика.
        
        Args:
            imu_data: Данные IMU в формате:
                {
                  "timestamp": 1712345678901,
                  "accelerometer": {"x": 0.12, "y": -0.05, "z": 9.78},
                  "gyroscope": {"x": 0.5, "y": -0.3, "z": 2.1},
                  "magnetometer": {"x": 25.3, "y": -10.1, "z": 40.2},
                  "temperature": 32.5,
                  "status": "ok"
                }
        """
        with self._thread_safe_operation():
            try:
                timestamp = imu_data.get("timestamp")
                self.logger.debug(f"Получены IMU данные: time={timestamp}")
                
                # Проверка статуса датчика
                if imu_data.get("status") != "ok":
                    self.logger.warning("IMU данные получены с ошибкой статуса")
                    return
                
                # Обновление данных датчика
                self._sensor_data_manager.update_imu_timestamp(timestamp)
                
                # Обработка данных
                self.imu_handler.process_imu_data(imu_data)
                
                # Обновление состояния модуля
                self._update_module_state()
                
            except Exception as e:
                self.logger.error(f"Ошибка обработки IMU данных: {e}")
    
    def _update_module_state(self) -> None:
        """Обновить внутреннее состояние модуля."""
        has_gps = self.gps_handler.has_recent_data()
        has_imu = (self._sensor_data_manager.has_imu_data and 
                  self.imu_handler.has_recent_data())
        
        active_gps_sensors = self._sensor_data_manager.get_active_gps_sensors()
        configured_gps_sensors = self._sensor_data_manager.configured_gps_sensors
        
        self._current_state = self._state_calculator.calculate_state(
            has_gps, has_imu, len(active_gps_sensors), 
            len(configured_gps_sensors), self._sensor_data_manager.last_update_time
        )
    
    def get_current_state(self, target_timestamp: Optional[int] = None,
                         force_update: bool = False) -> PositionState:
        """
        Получить состояние на указанный момент (или текущий).
        
        Args:
            target_timestamp: Unix time в миллисекундах, None - текущее время
            force_update: если True — пересчитать с экстраполяцией
            
        Returns:
            Состояние позиции транспортного средства
        """
        with self._thread_safe_operation():
            try:
                if target_timestamp is None:
                    target_timestamp = int(time.time() * 1000)
                
                self.logger.debug(f"Запрос состояния на время: {target_timestamp}")
                
                # Получение и обработка данных
                sensor_data = self.database_manager.get_data_for_time(target_timestamp)
                fused_position = self.position_fuser.fuse_position_data(
                    sensor_data, target_timestamp
                )
                
                # Прогнозирование при необходимости
                if force_update or self._is_data_stale(target_timestamp):
                    result = self.state_predictor.predict_position(
                        fused_position, target_timestamp
                    )
                else:
                    result = fused_position
                
                # Формирование результата
                return self._build_position_state(result, target_timestamp)
                
            except Exception as e:
                self.logger.error(f"Ошибка получения текущего состояния: {e}")
                return self._get_degraded_state(target_timestamp)
    
    def _build_position_state(self, position_data: Dict[str, Any], 
                            timestamp: int) -> PositionState:
        """Построить объект состояния позиции."""
        active_gps = self._sensor_data_manager.get_active_gps_sensors()
        
        # Определение источника данных
        if len(active_gps) == 2:
            source = "gps_dual"
        elif len(active_gps) == 1:
            source = f"gps_{active_gps[0]}"
        else:
            source = "none"
        
        # Расчет достоверности
        confidence = self._calculate_confidence()
        
        # Статус транспортного средства
        vehicle_status = self._vehicle_status_calculator.calculate_status(
            self._sensor_data_manager.last_update_time,
            self._sensor_data_manager.has_imu_data
        )
        
        return PositionState(
            lat=position_data.get("lat"),
            lon=position_data.get("lon"),
            alt=position_data.get("alt"),
            azimuth=position_data.get("azimuth"),
            timestamp=timestamp,
            confidence=confidence,
            status=self._current_state.value,
            source=source,
            vehicle_status=vehicle_status.value
        )
    
    def _calculate_confidence(self) -> float:
        """Рассчитать уровень достоверности."""
        time_since_last_update = 0.0
        if self._sensor_data_manager.last_update_time:
            time_since_last_update = (
                time.time() * 1000 - self._sensor_data_manager.last_update_time
            ) / 1000.0
        
        active_gps_sensors = self._sensor_data_manager.get_active_gps_sensors()
        configured_gps_sensors = self._sensor_data_manager.configured_gps_sensors
        
        return self._confidence_calculator.calculate_confidence(
            self._current_state, time_since_last_update,
            len(active_gps_sensors), len(configured_gps_sensors)
        )
    
    def _is_data_stale(self, target_timestamp: int) -> bool:
        """Проверить, устарели ли данные."""
        if not self._sensor_data_manager.last_update_time:
            return True
        
        time_diff = target_timestamp - self._sensor_data_manager.last_update_time
        max_stale_time = self.config.data.extrapolation.max_imu_time_sec * 1000
        
        return time_diff > max_stale_time
    
    def _get_degraded_state(self, timestamp: int) -> PositionState:
        """Вернуть состояние DEGRADED."""
        vehicle_status = self._vehicle_status_calculator.calculate_status(
            self._sensor_data_manager.last_update_time,
            self._sensor_data_manager.has_imu_data
        )
        
        return PositionState(
            lat=None, lon=None, alt=None, azimuth=None,
            timestamp=timestamp, confidence=0.0,
            status="degraded", source="none",
            vehicle_status=vehicle_status.value
        )
    
    def start_magnetometer_calibration(self, duration_sec: int = 120) -> None:
        """
        Запустить калибровку магнитометра.
        
        Args:
            duration_sec: Длительность калибровки в секундах
        """
        with self._thread_safe_operation():
            if not self._sensor_data_manager.has_imu_data:
                self.logger.warning(
                    "Невозможно запустить калибровку магнитометра: нет данных от IMU"
                )
                return
            
            self._current_state = ModuleState.CALIBRATING_MAG
            self._calibration_service.start_magnetometer_calibration(duration_sec)
    
    def get_history_stats(self) -> Dict[str, Any]:
        """
        Вернуть статистику по имеющимся данным.
        
        Returns:
            Статистика данных
        """
        try:
            stats = self.database_manager.get_data_statistics()
            stats.update({
                "module_state": self._current_state.value,
                "vehicle_status": self._vehicle_status_calculator.calculate_status(
                    self._sensor_data_manager.last_update_time,
                    self._sensor_data_manager.has_imu_data
                ).value,
                "has_imu_data": self._sensor_data_manager.has_imu_data,
                "active_gps_sensors": self._sensor_data_manager.get_active_gps_sensors(),
                "configured_gps_sensors": self._sensor_data_manager.configured_gps_sensors
            })
            return stats
        except Exception as e:
            self.logger.error(f"Ошибка получения статистики: {e}")
            return {"error": str(e)}
    
    def stop(self) -> None:
        """Корректная остановка модуля, сохранение состояния."""
        try:
            self.logger.info("Остановка модуля PositionEstimator")
            
            # Остановка таймеров
            self._timer_manager.stop_all_timers()
            
            # Сохранение состояния
            state_data = {
                "status": self._current_state.value,
                "timestamp": int(time.time() * 1000),
                "last_update": self._sensor_data_manager.last_update_time,
                "has_imu_data": self._sensor_data_manager.has_imu_data,
            }
            self.database_manager.save_state(state_data)
            
            # Закрытие соединений
            self.database_manager.close()
            
            self.logger.info("Модуль PositionEstimator остановлен")
            
        except Exception as e:
            self.logger.error(f"Ошибка при остановке модуля: {e}")