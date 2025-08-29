

## 1. Создание базового координатора

```python
# core/position_estimator.py
class PositionEstimator:
    """Главный координатор модуля позиционирования"""
    
    def __init__(self, config_path: str):
        self.logger = LoggerService.get_logger(__name__)
        
        # Инициализация менеджеров
        self.config_manager = ConfigManager(config_path)
        self.database_manager = DatabaseManager(self.config_manager.get_config())
        
        # Создание специализированных менеджеров
        self.sensor_manager = SensorManager(self.config_manager, self.database_manager)
        self.state_manager = StateManager(self.config_manager)
        self.calibration_service = CalibrationService(self.config_manager, self.database_manager)
        self.background_service = BackgroundService(self.config_manager, self.database_manager)
        
        # Инициализация
        self.state_manager.restore_state(self.database_manager)
        self.background_service.start()
    
    def update_gps(self, sensor_id: int, timestamp: int, lat: float, lon: float, alt: float):
        """Обновление GPS данных"""
        self.sensor_manager.update_gps(sensor_id, timestamp, lat, lon, alt)
        self.state_manager.update_from_sensors(self.sensor_manager.get_sensor_status())
    
    def update_imu(self, imu_data: Dict[str, Any]):
        """Обновление IMU данных"""
        self.sensor_manager.update_imu(imu_data)
        self.state_manager.update_from_sensors(self.sensor_manager.get_sensor_status())
    
    def get_current_state(self, target_timestamp: Optional[int] = None, force_update: bool = False) -> Dict[str, Any]:
        """Получение текущего состояния"""
        return self.state_manager.get_current_state(
            target_timestamp, 
            force_update, 
            self.sensor_manager,
            self.database_manager
        )
    
    def start_magnetometer_calibration(self, duration_sec: int = 120):
        """Запуск калибровки магнитометра"""
        self.calibration_service.start_magnetometer_calibration(
            duration_sec, 
            self.sensor_manager.has_imu_data()
        )
    
    def get_history_stats(self) -> Dict[str, Any]:
        """Получение статистики"""
        return self.state_manager.get_history_stats(
            self.database_manager, 
            self.sensor_manager
        )
    
    def stop(self):
        """Остановка модуля"""
        self.state_manager.save_state(self.database_manager)
        self.background_service.stop()
        self.database_manager.close()
```

## 2. Менеджер датчиков

```python
# managers/sensor_manager.py
class SensorManager:
    """Управление всеми датчиками и их данными"""
    
    def __init__(self, config_manager: ConfigManager, database_manager: DatabaseManager):
        self.config = config_manager.get_config()
        self.logger = LoggerService.get_logger(__name__)
        
        # Инициализация обработчиков данных
        self.gps_handler = GPSDataHandler(self.config, database_manager)
        self.imu_handler = IMUDataHandler(self.config, database_manager)
        self.position_fuser = PositionFuser(self.config)
        self.state_predictor = StatePredictor(self.config)
        
        # Состояние датчиков
        self.gps_tracker = GPSTracker(config_manager)
        self.imu_tracker = IMUTracker()
        
        self._lock = threading.RLock()
    
    def update_gps(self, sensor_id: int, timestamp: int, lat: float, lon: float, alt: float):
        """Обработка GPS данных"""
        with self._lock:
            try:
                self.logger.debug(f"GPS данные от датчика {sensor_id}: lat={lat}, lon={lon}")
                
                self.gps_tracker.update_sensor(sensor_id, timestamp)
                self.gps_handler.process_gps_data(sensor_id, timestamp, lat, lon, alt)
                
            except Exception as e:
                self.logger.error(f"Ошибка обработки GPS данных: {e}")
    
    def update_imu(self, imu_data: Dict[str, Any]):
        """Обработка IMU данных"""
        with self._lock:
            try:
                timestamp = imu_data.get("timestamp")
                self.logger.debug(f"IMU данные: time={timestamp}")
                
                if imu_data.get("status") != "ok":
                    self.logger.warning("IMU данные с ошибкой статуса")
                    return
                
                self.imu_tracker.update_data(timestamp)
                self.imu_handler.process_imu_data(imu_data)
                
            except Exception as e:
                self.logger.error(f"Ошибка обработки IMU данных: {e}")
    
    def get_sensor_status(self) -> Dict[str, Any]:
        """Получение статуса всех датчиков"""
        return {
            "gps": self.gps_tracker.get_status(),
            "imu": self.imu_tracker.get_status(),
            "has_recent_gps": self.gps_handler.has_recent_data(),
            "has_recent_imu": self.imu_handler.has_recent_data(),
            "is_vehicle_moving": self._is_vehicle_moving()
        }
    
    def has_imu_data(self) -> bool:
        """Проверка наличия IMU данных"""
        return self.imu_tracker.has_data()
    
    def _is_vehicle_moving(self) -> bool:
        """Проверка движения транспорта"""
        if self.imu_tracker.has_data() and self.imu_handler.is_vehicle_moving():
            return True
        return self.gps_handler.is_vehicle_moving()
```

## 3. Менеджер состояния

```python
# managers/state_manager.py
class StateManager:
    """Управление состоянием модуля"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager.get_config()
        self.logger = LoggerService.get_logger(__name__)
        
        self._state = "BOOTING"
        self._last_update_time = None
        self._lock = threading.RLock()
        
        self.confidence_calculator = ConfidenceCalculator(self.config)
        self.state_analyzer = StateAnalyzer(self.config)
    
    def restore_state(self, database_manager: DatabaseManager):
        """Восстановление состояния из БД"""
        try:
            last_state = database_manager.get_last_state()
            if last_state:
                self._state = last_state.get("status", "READY")
                self._last_update_time = last_state.get("timestamp")
                self.logger.info(f"Состояние восстановлено: {self._state}")
            else:
                self._state = "READY"
        except Exception as e:
            self.logger.error(f"Ошибка восстановления состояния: {e}")
            self._state = "READY"
    
    def update_from_sensors(self, sensor_status: Dict[str, Any]):
        """Обновление состояния на основе данных датчиков"""
        with self._lock:
            self._state = self.state_analyzer.analyze_state(sensor_status)
            self._last_update_time = int(time.time() * 1000)
    
    def get_current_state(self, target_timestamp: Optional[int], force_update: bool, 
                         sensor_manager: 'SensorManager', database_manager: DatabaseManager) -> Dict[str, Any]:
        """Получение текущего состояния"""
        with self._lock:
            try:
                if target_timestamp is None:
                    target_timestamp = int(time.time() * 1000)
                
                # Получение и обработка данных
                sensor_data = database_manager.get_data_for_time(target_timestamp)
                fused_position = sensor_manager.position_fuser.fuse_position_data(sensor_data, target_timestamp)
                
                if force_update or self._is_data_stale(target_timestamp):
                    result = sensor_manager.state_predictor.predict_position(fused_position, target_timestamp)
                else:
                    result = fused_position
                
                # Добавление метаинформации
                self._enrich_result(result, sensor_manager.get_sensor_status())
                
                return result
                
            except Exception as e:
                self.logger.error(f"Ошибка получения состояния: {e}")
                return self._get_degraded_state(target_timestamp)
    
    def _enrich_result(self, result: Dict[str, Any], sensor_status: Dict[str, Any]):
        """Обогащение результата метаинформацией"""
        result["status"] = self._state
        result["vehicle_status"] = self._get_vehicle_status()
        result["source"] = self._get_data_source(sensor_status["gps"])
        result["confidence"] = self.confidence_calculator.calculate(
            self._state, self._last_update_time, sensor_status
        )
    
    def save_state(self, database_manager: DatabaseManager):
        """Сохранение состояния в БД"""
        try:
            state_data = {
                "status": self._state,
                "timestamp": int(time.time() * 1000),
                "last_update": self._last_update_time
            }
            database_manager.save_state(state_data)
        except Exception as e:
            self.logger.error(f"Ошибка сохранения состояния: {e}")
```

## 4. Трекеры датчиков

```python
# trackers/gps_tracker.py
class GPSTracker:
    """Отслеживание состояния GPS датчиков"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager.get_config()
        self._last_timestamps = {}
        self._configured_sensors = self._get_configured_sensors()
    
    def update_sensor(self, sensor_id: int, timestamp: int):
        """Обновление временной метки датчика"""
        self._last_timestamps[sensor_id] = timestamp
    
    def get_active_sensors(self, time_threshold_ms: int = 2000) -> List[int]:
        """Получение активных датчиков"""
        current_time = int(time.time() * 1000)
        active_sensors = []
        
        for sensor_id, last_timestamp in self._last_timestamps.items():
            if last_timestamp and (current_time - last_timestamp) <= time_threshold_ms:
                active_sensors.append(sensor_id)
        
        return active_sensors
    
    def get_status(self) -> Dict[str, Any]:
        """Получение статуса GPS системы"""
        active_sensors = self.get_active_sensors()
        return {
            "active_sensors": active_sensors,
            "configured_sensors": self._configured_sensors,
            "num_active": len(active_sensors),
            "num_configured": len(self._configured_sensors),
            "last_timestamps": self._last_timestamps.copy()
        }

# trackers/imu_tracker.py
class IMUTracker:
    """Отслеживание состояния IMU датчика"""
    
    def __init__(self):
        self._has_data = False
        self._last_timestamp = None
    
    def update_data(self, timestamp: int):
        """Обновление данных IMU"""
        self._has_data = True
        self._last_timestamp = timestamp
    
    def has_data(self) -> bool:
        """Проверка наличия данных"""
        return self._has_data
    
    def get_status(self) -> Dict[str, Any]:
        """Получение статуса IMU"""
        return {
            "has_data": self._has_data,
            "last_timestamp": self._last_timestamp
        }
```

## 5. Сервис фоновых процессов

```python
# services/background_service.py
class BackgroundService:
    """Управление фоновыми процессами"""
    
    def __init__(self, config_manager: ConfigManager, database_manager: DatabaseManager):
        self.config = config_manager.get_config()
        self.database_manager = database_manager
        self.logger = LoggerService.get_logger(__name__)
        
        self._timers = []
    
    def start(self):
        """Запуск всех фоновых процессов"""
        self._start_gyro_calibration_timer()
        self._start_data_cleanup_timer()
    
    def stop(self):
        """Остановка всех фоновых процессов"""
        for timer in self._timers:
            timer.cancel()
        self._timers.clear()
    
    def _start_gyro_calibration_timer(self):
        """Запуск таймера калибровки гироскопа"""
        interval = self.config.get("imu", {}).get("calibration", {}).get("gyro_interval_min", 30) * 60
        timer = threading.Timer(interval, self._perform_auto_gyro_calibration)
        timer.daemon = True
        self._timers.append(timer)
        timer.start()
    
    def _start_data_cleanup_timer(self):
        """Запуск таймера очистки данных"""
        timer = threading.Timer(300, self._cleanup_old_data)
        timer.daemon = True
        self._timers.append(timer)
        timer.start()
```

## 6. Вспомогательные классы

```python
# analyzers/state_analyzer.py
class StateAnalyzer:
    """Анализ и определение состояния системы"""
    
    def analyze_state(self, sensor_status: Dict[str, Any]) -> str:
        """Определение состояния на основе данных датчиков"""
        has_gps = sensor_status["has_recent_gps"]
        has_imu = sensor_status["has_recent_imu"]
        gps_info = sensor_status["gps"]
        
        if has_gps and has_imu:
            return "GPS_IMU_FUSION"
        elif has_imu:
            return "IMU_DEAD_RECKONING"
        elif has_gps:
            return "GPS_ONLY"
        else:
            return "STANDBY"

# calculators/confidence_calculator.py  
class ConfidenceCalculator:
    """Расчет уровня достоверности"""
    
    def calculate(self, state: str, last_update_time: Optional[int], 
                 sensor_status: Dict[str, Any]) -> float:
        """Расчет confidence на основе состояния и данных датчиков"""
        if state == "BOOTING":
            return 0.0
        elif state == "GPS_IMU_FUSION":
            return self._calculate_fusion_confidence(last_update_time, sensor_status)
        elif state == "GPS_ONLY":
            return self._calculate_gps_confidence(sensor_status)
        # ... другие состояния
```


