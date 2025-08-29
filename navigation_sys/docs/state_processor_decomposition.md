

```python
# state_models.py
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from enum import Enum

class ModuleState(Enum):
    """Возможные состояния модуля"""
    BOOTING = "BOOTING"
    GPS_IMU_FUSION = "GPS_IMU_FUSION"
    IMU_DEAD_RECKONING = "IMU_DEAD_RECKONING"
    GPS_ONLY = "GPS_ONLY"
    STANDBY = "STANDBY"
    CALIBRATING_MAG = "CALIBRATING_MAG"
    POWER_OFF = "POWER_OFF"
    DEGRADED = "DEGRADED"

class VehicleStatus(Enum):
    """Статусы транспортного средства"""
    OFF = "off"
    STANDBY = "standby"
    RUNNING = "running"

@dataclass
class StateInfo:
    """Информация о состоянии модуля"""
    current_state: str
    previous_state: Optional[str]
    state_timestamp: int
    vehicle_status: str
    confidence: float
    last_gps_time: Optional[int]
    last_imu_time: Optional[int]
    last_valid_position_time: Optional[int]
    has_imu_data: bool
    active_gps_sensors: List[int]
    configured_gps_sensors: List[int]
    gps_sensor_status: Dict[int, Dict[str, Any]]
```

```python
# gps_sensor_manager.py
import time
from typing import Dict, List, Optional

class GPSSensorManager:
    """Управление GPS датчиками"""
    
    def __init__(self, config: Dict):
        self.config = config
        self._last_timestamps: Dict[int, int] = {}
        self._configured_sensors = self._get_configured_sensors()
    
    def _get_configured_sensors(self) -> List[int]:
        """Получение списка сконфигурированных GPS датчиков"""
        gps_config = self.config.get("gps", {}).get("sensors", [])
        return [sensor.get("id") for sensor in gps_config]
    
    def update_sensor_timestamp(self, sensor_id: int, timestamp: int):
        """Обновление временной метки датчика"""
        self._last_timestamps[sensor_id] = timestamp
    
    def get_active_sensors(self, time_threshold_ms: int = 2000) -> List[int]:
        """Получение списка активных датчиков"""
        current_time = int(time.time() * 1000)
        active_sensors = []
        
        for sensor_id, last_timestamp in self._last_timestamps.items():
            if last_timestamp and (current_time - last_timestamp) <= time_threshold_ms:
                active_sensors.append(sensor_id)
        
        return active_sensors
    
    @property
    def configured_sensors(self) -> List[int]:
        """Список сконфигурированных датчиков"""
        return self._configured_sensors.copy()
    
    @property
    def last_timestamps(self) -> Dict[int, int]:
        """Временные метки датчиков"""
        return self._last_timestamps.copy()
    
    def get_latest_timestamp(self) -> Optional[int]:
        """Получение последней временной метки среди всех датчиков"""
        if not self._last_timestamps:
            return None
        return max(self._last_timestamps.values())
    
    def reset(self):
        """Сброс состояния датчиков"""
        self._last_timestamps.clear()
```

```python
# confidence_calculator.py
import time
from typing import Dict, List

class ConfidenceCalculator:
    """Расчет показателя достоверности"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def calculate(self, 
                 current_state: str,
                 last_gps_time: int = None,
                 active_gps_count: int = 0,
                 configured_gps_count: int = 0) -> float:
        """
        Расчет показателя достоверности
        
        Args:
            current_state: Текущее состояние модуля
            last_gps_time: Время последних GPS данных
            active_gps_count: Количество активных GPS датчиков
            configured_gps_count: Количество сконфигурированных GPS датчиков
            
        Returns:
            Достоверность от 0.0 до 1.0
        """
        current_time = int(time.time() * 1000)
        time_since_last_gps = 0
        
        if last_gps_time:
            time_since_last_gps = (current_time - last_gps_time) / 1000.0
        
        confidence = self._calculate_base_confidence(current_state, time_since_last_gps)
        confidence = self._apply_gps_correction(confidence, active_gps_count, configured_gps_count)
        
        return max(0.0, min(1.0, confidence))
    
    def _calculate_base_confidence(self, state: str, time_since_gps: float) -> float:
        """Базовый расчет достоверности по состоянию"""
        confidence_map = {
            "BOOTING": 0.0,
            "GPS_IMU_FUSION": 0.95 - (time_since_gps * 0.01),
            "IMU_DEAD_RECKONING": 0.70 - (time_since_gps * 0.05),
            "GPS_ONLY": 0.80,
            "STANDBY": 0.85 - (time_since_gps * 0.005),
            "CALIBRATING_MAG": 0.0,
            "POWER_OFF": 0.0,
            "DEGRADED": 0.0
        }
        
        return confidence_map.get(state, 0.0)
    
    def _apply_gps_correction(self, confidence: float, active_count: int, configured_count: int) -> float:
        """Применение корректировок на основе GPS датчиков"""
        if configured_count < 2:
            return confidence if active_count > 0 else 0.0
        
        if active_count == 2:
            return confidence
        elif active_count == 1:
            return confidence * 0.7
        else:
            return 0.0
```

```python
# vehicle_status_manager.py
import time
from typing import Optional, Dict

class VehicleStatusManager:
    """Управление статусом транспортного средства"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def determine_status(self, 
                        last_gps_time: Optional[int],
                        last_imu_time: Optional[int]) -> str:
        """
        Определение статуса транспортного средства
        
        Args:
            last_gps_time: Время последних GPS данных
            last_imu_time: Время последних IMU данных
            
        Returns:
            Статус транспортного средства
        """
        current_time = int(time.time() * 1000)
        
        last_activity_time = max(
            filter(None, [last_gps_time, last_imu_time]),
            default=0
        )
        
        if last_activity_time == 0:
            return "off"
        
        inactivity_time = current_time - last_activity_time
        
        off_timeout = self._get_off_timeout_ms()
        standby_timeout = self._get_standby_timeout_ms()
        
        if inactivity_time > off_timeout:
            return "off"
        elif inactivity_time > standby_timeout:
            return "standby"
        else:
            return "running"
    
    def _get_off_timeout_ms(self) -> int:
        """Получение таймаута выключения в миллисекундах"""
        return self.config.get("power_management", {}).get("inactivity_timeout_to_off_sec", 300) * 1000
    
    def _get_standby_timeout_ms(self) -> int:
        """Получение таймаута перехода в режим ожидания в миллисекундах"""
        return self.config.get("power_management", {}).get("inactivity_timeout_to_standby_sec", 30) * 1000
```

```python
# state_manager.py
import time
import threading
from typing import Dict, Any, Optional

from .state_models import ModuleState, StateInfo
from .gps_sensor_manager import GPSSensorManager
from .confidence_calculator import ConfidenceCalculator
from .vehicle_status_manager import VehicleStatusManager

class StateManager:
    """Управление состоянием модуля"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._lock = threading.RLock()
        
        # Компоненты
        self.gps_manager = GPSSensorManager(config)
        self.confidence_calc = ConfidenceCalculator(config)
        self.vehicle_status_manager = VehicleStatusManager(config)
        
        # Состояние
        self._current_state = ModuleState.BOOTING.value
        self._previous_state = None
        self._state_timestamp = int(time.time() * 1000)
        
        # Временные метки
        self._last_imu_time = None
        self._last_valid_position_time = None
        
        # IMU статус
        self._has_imu_data = False
    
    @property
    def current_state(self) -> str:
        with self._lock:
            return self._current_state
    
    def update_state(self):
        """Обновление состояния модуля"""
        with self._lock:
            previous_state = self._current_state
            self._previous_state = previous_state
            
            current_time = int(time.time() * 1000)
            
            # Определение наличия свежих данных
            has_recent_gps = self._has_recent_data(self.gps_manager.get_latest_timestamp(), current_time, 2000)
            has_recent_imu = self._has_imu_data and self._has_recent_data(self._last_imu_time, current_time, 100)
            
            # Определение статуса ТС
            vehicle_status = self.vehicle_status_manager.determine_status(
                self.gps_manager.get_latest_timestamp(),
                self._last_imu_time
            )
            
            # Определение состояния
            self._current_state = self._determine_new_state(
                has_recent_gps, has_recent_imu, vehicle_status, current_time
            )
            
            self._state_timestamp = current_time
            
            if previous_state != self._current_state:
                self._log_state_change(previous_state)
    
    def _determine_new_state(self, has_gps: bool, has_imu: bool, vehicle_status: str, current_time: int) -> str:
        """Определение нового состояния на основе данных"""
        if self._current_state == ModuleState.CALIBRATING_MAG.value:
            return self._current_state
        elif vehicle_status == "off":
            return ModuleState.POWER_OFF.value
        elif self._is_data_stale(current_time):
            return ModuleState.DEGRADED.value
        elif has_gps and has_imu:
            return ModuleState.GPS_IMU_FUSION.value
        elif has_imu:
            return ModuleState.IMU_DEAD_RECKONING.value
        elif has_gps:
            return ModuleState.GPS_ONLY.value
        elif self.gps_manager.get_latest_timestamp() is None and self._last_imu_time is None:
            return ModuleState.BOOTING.value
        else:
            return ModuleState.STANDBY.value
    
    def _has_recent_data(self, last_time: Optional[int], current_time: int, threshold_ms: int) -> bool:
        """Проверка свежести данных"""
        if last_time is None:
            return False
        return (current_time - last_time) <= threshold_ms
    
    def _is_data_stale(self, current_time: int) -> bool:
        """Проверка устаревания данных"""
        if self._last_valid_position_time is None:
            return True
        
        stale_threshold = self._get_stale_threshold()
        time_since_last_valid = current_time - self._last_valid_position_time
        
        return time_since_last_valid > stale_threshold
    
    def _get_stale_threshold(self) -> int:
        """Получение порога устаревания данных"""
        has_recent_imu = self._has_imu_data and self._has_recent_data(
            self._last_imu_time, int(time.time() * 1000), 100
        )
        
        if has_recent_imu:
            return self.config.get("extrapolation", {}).get("max_imu_time_sec", 120) * 1000
        else:
            return self.config.get("extrapolation", {}).get("max_gps_only_time_sec", 15) * 1000
    
    def _log_state_change(self, previous_state: str):
        """Логирование изменения состояния"""
        # Здесь будет логирование
        pass
```

```python
# state_processor.py
import time
import threading
from typing import Dict, Any

from .state_manager import StateManager
from .state_models import StateInfo
from ..services.logger_service import LoggerService

class StateProcessor:
    """Упрощенный процессор состояний - основной интерфейс"""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = LoggerService.get_logger(__name__)
        self.config = config
        self._lock = threading.RLock()
        
        self.state_manager = StateManager(config)
        
        self.logger.info("StateProcessor инициализирован")
    
    @property
    def current_state(self) -> str:
        return self.state_manager.current_state
    
    @property
    def vehicle_status(self) -> str:
        return self.state_manager.vehicle_status_manager.determine_status(
            self.state_manager.gps_manager.get_latest_timestamp(),
            self.state_manager._last_imu_time
        )
    
    @property
    def has_imu_data(self) -> bool:
        with self._lock:
            return self.state_manager._has_imu_data
    
    def set_imu_data_available(self, available: bool):
        with self._lock:
            self.state_manager._has_imu_data = available
            self.state_manager.update_state()
    
    def update_gps_timestamp(self, sensor_id: int, timestamp: int):
        with self._lock:
            self.state_manager.gps_manager.update_sensor_timestamp(sensor_id, timestamp)
            self.state_manager.update_state()
    
    def update_imu_timestamp(self, timestamp: int):
        with self._lock:
            self.state_manager._last_imu_time = timestamp
            if not self.state_manager._has_imu_data:
                self.state_manager._has_imu_data = True
            self.state_manager.update_state()
    
    def update_valid_position_timestamp(self, timestamp: int):
        with self._lock:
            self.state_manager._last_valid_position_time = timestamp
            self.state_manager.update_state()
    
    def calculate_confidence(self) -> float:
        active_gps = self.state_manager.gps_manager.get_active_sensors()
        configured_gps = self.state_manager.gps_manager.configured_sensors
        
        return self.state_manager.confidence_calc.calculate(
            self.state_manager._current_state,
            self.state_manager.gps_manager.get_latest_timestamp(),
            len(active_gps),
            len(configured_gps)
        )
    
    def get_state_info(self) -> Dict[str, Any]:
        with self._lock:
            active_gps = self.state_manager.gps_manager.get_active_sensors()
            configured_gps = self.state_manager.gps_manager.configured_sensors
            
            return {
                "current_state": self.state_manager._current_state,
                "previous_state": self.state_manager._previous_state,
                "state_timestamp": self.state_manager._state_timestamp,
                "vehicle_status": self.vehicle_status,
                "confidence": self.calculate_confidence(),
                "last_gps_time": self.state_manager.gps_manager.get_latest_timestamp(),
                "last_imu_time": self.state_manager._last_imu_time,
                "last_valid_position_time": self.state_manager._last_valid_position_time,
                "has_imu_data": self.state_manager._has_imu_data,
                "active_gps_sensors": active_gps,
                "configured_gps_sensors": configured_gps,
                "gps_sensor_status": {
                    sensor_id: {
                        "last_timestamp": self.state_manager.gps_manager.last_timestamps.get(sensor_id),
                        "is_active": sensor_id in active_gps
                    } for sensor_id in configured_gps
                }
            }
    
    def set_calibration_state(self, is_calibrating: bool):
        with self._lock:
            if is_calibrating:
                if not self.state_manager._has_imu_data:
                    self.logger.warning("Невозможно начать калибровку: нет данных от IMU")
                    return False
                self.state_manager._current_state = "CALIBRATING_MAG"
            else:
                self.state_manager.update_state()
            return True
    
    def reset(self):
        with self._lock:
            self.state_manager = StateManager(self.config)
            self.logger.info("StateProcessor сброшен")
```


5. **Переиспользование**: Компоненты можно использовать в других частях системы.

6. **Снижение связности**: Компоненты слабо связаны между собой.

Основной класс `StateProcessor` остается простым интерфейсом, делегирующим работу специализированным компонентам.
