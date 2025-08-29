import time
import threading
from typing import Dict, Optional, Any, List
from ..storage.database_manager import DatabaseManager
from ..storage.config_manager import ConfigManager
from ..handlers.gps_data_handler import GPSDataHandler
from ..handlers.imu_data_handler import IMUDataHandler
from ..processing.position_fuser import PositionFuser
from ..processing.state_predictor import StatePredictor
from ..calibration.calibration_manager import CalibrationManager
from ..services.logger_service import LoggerService


class PositionEstimator:
    """
    Главный класс модуля обработки GPS и IMU данных.
    Предоставляет интерфейсы для приёма данных от внешних датчиков
    и запроса текущего состояния погрузчика.
    """

    def __init__(self, config_path: str):
        """
        Инициализация модуля позиционирования.

        Args:
            config_path (str): Путь к конфигурационному файлу
        """
        self.logger = LoggerService.get_logger(__name__)
        self.logger.info("Инициализация модуля PositionEstimator")

        # Загрузка конфигурации
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()

        # Инициализация компонентов
        self.database_manager = DatabaseManager(self.config)
        self.gps_handler = GPSDataHandler(self.config, self.database_manager)
        self.imu_handler = IMUDataHandler(self.config, self.database_manager)
        self.position_fuser = PositionFuser(self.config)
        self.state_predictor = StatePredictor(self.config)
        self.calibration_manager = CalibrationManager(
            self.config, self.database_manager)

        # Состояние модуля
        self._state = "BOOTING"
        self._last_update_time = None
        self._lock = threading.RLock()

        # Отслеживание состояния GPS датчиков
        self._gps_last_timestamps = {}  # sensor_id -> timestamp
        self._gps_configured_sensors = self._get_configured_gps_sensors()

        # Флаг наличия IMU данных
        self._has_imu_data = False
        self._imu_initialization_checked = False

        # Восстановление состояния из БД
        self._restore_state()

        # Запуск фоновых процессов
        self._start_background_processes()

        self.logger.info("Модуль PositionEstimator инициализирован успешно")

    def _get_configured_gps_sensors(self) -> List[int]:
        """Получение списка сконфигурированных GPS датчиков"""
        gps_config = self.config.get("gps", {}).get("sensors", [])
        return [sensor.get("id") for sensor in gps_config]

    def _restore_state(self):
        """Восстановление последнего состояния из базы данных"""
        try:
            # Здесь должна быть логика восстановления последнего состояния
            # из таблицы sensor_data или отдельной таблицы состояний
            last_state = self.database_manager.get_last_state()
            if last_state:
                self._state = last_state.get("status", "READY")
                self._last_update_time = last_state.get("timestamp")
                # Восстановление временных меток GPS
                if "gps_timestamps" in last_state:
                    self._gps_last_timestamps = last_state["gps_timestamps"]
                self.logger.info(f"Состояние восстановлено: {self._state}")
            else:
                self._state = "READY"
        except Exception as e:
            self.logger.error(f"Ошибка восстановления состояния: {e}")
            self._state = "READY"

    def _start_background_processes(self):
        """Запуск фоновых процессов обработки данных"""
        # Запуск периодической калибровки гироскопа (с проверкой наличия IMU)
        self._start_gyro_calibration_timer()

        # Запуск очистки старых данных
        self._start_data_cleanup_timer()

    def _start_gyro_calibration_timer(self):
        """Запуск таймера для периодической калибровки гироскопа"""
        interval = self.config.get("imu", {}).get(
            "calibration", {}).get("gyro_interval_min", 30) * 60
        timer = threading.Timer(interval, self._perform_auto_gyro_calibration)
        timer.daemon = True
        timer.start()

    def _perform_auto_gyro_calibration(self):
        """Выполнение автоматической калибровки гироскопа"""
        try:
            # Проверяем, есть ли вообще IMU данные
            if not self._has_imu_data:
                self.logger.debug(
                    "Пропуск калибровки гироскопа: нет данных от IMU")
                return

            # Проверяем, неподвижен ли транспорт
            if self.imu_handler.is_vehicle_still():
                self.logger.info("Начало автоматической калибровки гироскопа")
                bias = self.imu_handler.calculate_gyro_bias()
                self.calibration_manager.save_calibration(
                    "imu_gyro_bias", bias)
                self.logger.info(
                    "Автоматическая калибровка гироскопа завершена")
            else:
                self.logger.debug(
                    "Пропуск калибровки: транспорт не неподвижен")
        except Exception as e:
            self.logger.error(
                f"Ошибка при автоматической калибровке гироскопа: {e}")
        finally:
            # Перезапуск таймера
            self._start_gyro_calibration_timer()

    def _start_data_cleanup_timer(self):
        """Запуск таймера для очистки старых данных"""
        timer = threading.Timer(300, self._cleanup_old_data)  # Каждые 5 минут
        timer.daemon = True
        timer.start()

    def _cleanup_old_data(self):
        """Очистка старых данных из базы"""
        try:
            retention_minutes = self.config.get(
                "database", {}).get("retention_minutes", 5)
            self.database_manager.cleanup_old_data(retention_minutes)
        except Exception as e:
            self.logger.error(f"Ошибка при очистке старых данных: {e}")
        finally:
            # Перезапуск таймера
            self._start_data_cleanup_timer()

    # === Методы для приёма данных от внешней системы ===

    def update_gps(self, sensor_id: int, timestamp: int, lat: float,
                   lon: float, alt: float):
        """
        Приём данных от GPS-датчика.

        Args:
            sensor_id (int): Идентификатор датчика (1 или 2)
            timestamp (int): Unix time в миллисекундах
            lat (float): Широта
            lon (float): Долгота
            alt (float): Высота
        """
        with self._lock:
            try:
                self.logger.debug(
                            f"Получены GPS данные от датчика {sensor_id}: "
                            f"lat={lat}, lon={lon}, "
                            f"alt={alt}, time={timestamp}"
                        )

                # Обновление временной метки для конкретного датчика
                self._gps_last_timestamps[sensor_id] = timestamp

                # Передача данных в обработчик GPS
                self.gps_handler.process_gps_data(
                    sensor_id, timestamp, lat, lon, alt)

                # Обновление времени последнего обновления
                self._last_update_time = timestamp
                self._update_module_state()

            except Exception as e:
                self.logger.error(
                    f"Ошибка обработки GPS данных от датчика {sensor_id}: {e}")

    def update_imu(self, imu_data: Dict[str, Any]):
        """
        Приём данных от IMU-датчика.

        Args:
            imu_data (dict): Данные IMU в формате:
                {
                  "timestamp": 1712345678901,
                  "accelerometer": {"x": 0.12, "y": -0.05, "z": 9.78},
                  "gyroscope": {"x": 0.5, "y": -0.3, "z": 2.1},
                  "magnetometer": {"x": 25.3, "y": -10.1, "z": 40.2},
                  "temperature": 32.5,
                  "status": "ok"
                }
        """
        with self._lock:
            try:
                timestamp = imu_data.get("timestamp")
                self.logger.debug(f"Получены IMU данные: time={timestamp}")

                # Отмечаем, что IMU данные получены
                self._has_imu_data = True
                self._imu_initialization_checked = True

                # Проверка статуса датчика
                if imu_data.get("status") != "ok":
                    self.logger.warning(
                        "IMU данные получены с ошибкой статуса")
                    return

                # Передача данных в обработчик IMU
                self.imu_handler.process_imu_data(imu_data)

                # Обновление времени последнего обновления
                self._last_update_time = timestamp
                self._update_module_state()

            except Exception as e:
                self.logger.error(f"Ошибка обработки IMU данных: {e}")

    def _get_active_gps_sensors(self,
                                time_threshold_ms: int = 2000) -> List[int]:
        """
        Получение списка активных GPS датчиков.

        Args:
            time_threshold_ms (int): Порог времени для определения 
                                    активности (мс)

        Returns:
            List[int]: Список ID активных датчиков
        """
        current_time = int(time.time() * 1000)
        active_sensors = []

        for sensor_id, last_timestamp in self._gps_last_timestamps.items():
            if last_timestamp and (current_time - last_timestamp) <= time_threshold_ms:
                active_sensors.append(sensor_id)

        return active_sensors

    def _update_module_state(self):
        """Обновление внутреннего состояния модуля"""
        # Проверка наличия данных от датчиков
        has_gps = self.gps_handler.has_recent_data()
        has_imu = self._has_imu_data and self.imu_handler.has_recent_data()

        # Получение активных GPS датчиков
        active_gps_sensors = self._get_active_gps_sensors()
        num_active_gps = len(active_gps_sensors)
        num_configured_gps = len(self._gps_configured_sensors)

        self.logger.debug(
            f"GPS датчики: активные {active_gps_sensors}, всего \
                сконфигурировано {num_configured_gps}")

        # Определение состояния модуля
        if has_gps and has_imu:
            self._state = "GPS_IMU_FUSION"
        elif has_imu:
            self._state = "IMU_DEAD_RECKONING"
        elif has_gps:
            # Проверяем конфигурацию GPS
            if num_configured_gps >= 2 and num_active_gps >= 2:
                # Все датчики активны
                self._state = "GPS_ONLY"
            elif num_configured_gps >= 2 and num_active_gps == 1:
                # Только один из двух датчиков активен
                self._state = "GPS_ONLY"
                self.logger.info(
                    f"Работает только один GPS датчик из \
                        {num_configured_gps}:{active_gps_sensors}")
            elif num_configured_gps == 1 and num_active_gps == 1:
                # Один датчик сконфигурирован и активен
                self._state = "GPS_ONLY"
            else:
                # Нет активных датчиков (теоретически не должно происходить)
                self._state = "STANDBY"
        else:
            # Проверка времени последнего обновления
            if self._last_update_time:
                time_diff = time.time() * 1000 - self._last_update_time
                if time_diff > 120000:  # 2 минуты
                    self._state = "DEGRADED"
                else:
                    self._state = "STANDBY"
            else:
                self._state = "BOOTING"

    # === Методы для запроса данных внешней системой ===

    def get_current_state(self, target_timestamp: Optional[int] = None,
                          force_update: bool = False) -> Dict[str, Any]:
        """
        Получение состояния на указанный момент (или текущий).

        Args:
            target_timestamp (int, optional): Unix time в миллисекундах, 
            None - текущее время
            force_update (bool): если True — пересчитать с экстраполяцией

        Returns:
            dict: Словарь с координатами, азимутом, confidence и статусом
        """
        with self._lock:
            try:
                if target_timestamp is None:
                    target_timestamp = int(time.time() * 1000)

                self.logger.debug(
                    f"Запрос состояния на время: {target_timestamp}")

                # Получение данных из БД для указанного времени
                sensor_data = self.database_manager.get_data_for_time(
                    target_timestamp)

                # Фузия данных
                fused_position = self.position_fuser.fuse_position_data(
                    sensor_data, target_timestamp
                )

                # Прогнозирование, если данные устарели
                if force_update or self._is_data_stale(target_timestamp):
                    predicted_position = self.state_predictor.predict_position(
                        fused_position, target_timestamp
                    )
                    result = predicted_position
                else:
                    result = fused_position

                # Добавление информации о состоянии модуля
                result["status"] = self._state
                result["vehicle_status"] = self._get_vehicle_status()

                # Добавление информации об активных GPS датчиках
                active_gps = self._get_active_gps_sensors()
                if len(active_gps) == 2:
                    result["source"] = "gps_dual"
                elif len(active_gps) == 1:
                    result["source"] = f"gps_{active_gps[0]}"
                else:
                    result["source"] = "none"

                # Расчет confidence
                result["confidence"] = self._calculate_confidence()

                self.logger.debug(f"Возвращено состояние: {result}")
                return result

            except Exception as e:
                self.logger.error(f"Ошибка получения текущего состояния: {e}")
                return self._get_degraded_state(target_timestamp)

    def _calculate_confidence(self) -> float:
        """Расчет уровня достоверности на основе текущего состояния"""
        current_time = time.time() * 1000
        time_since_last_gps = 0

        if self._last_update_time:
            time_since_last_gps = (
                current_time - self._last_update_time) / 1000.0  # в секундах

        # Получение информации об активных GPS датчиках
        active_gps_sensors = self._get_active_gps_sensors()
        num_active_gps = len(active_gps_sensors)
        num_configured_gps = len(self._gps_configured_sensors)

        # Расчет confidence в зависимости от состояния
        if self._state == "BOOTING":
            return 0.0
        elif self._state == "GPS_IMU_FUSION":
            # 95% - (время с последнего GPS * 1%)
            confidence = 0.95 - (time_since_last_gps * 0.01)
            # Корректировка на основе количества активных GPS датчиков
            if num_configured_gps >= 2 and num_active_gps < 2:
                confidence *= 0.9  # Штраф за отсутствие одного датчика
            return max(0.0, min(1.0, confidence))
        elif self._state == "IMU_DEAD_RECKONING":
            # 70% - (время с последнего GPS * 5%)
            confidence = 0.70 - (time_since_last_gps * 0.05)
            return max(0.0, min(1.0, confidence))
        elif self._state == "GPS_ONLY":
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
        elif self._state == "STANDBY":
            # 85% - (время с последнего GPS * 0.5%)
            confidence = 0.85 - (time_since_last_gps * 0.005)
            # Корректировка на основе количества активных GPS датчиков
            if num_configured_gps >= 2 and num_active_gps < 2:
                confidence *= 0.8  # Штраф за отсутствие одного датчика
            return max(0.0, min(1.0, confidence))
        elif self._state == "CALIBRATING_MAG":
            return 0.0
        elif self._state == "POWER_OFF":
            return 0.0
        elif self._state == "DEGRADED":
            return 0.0
        else:
            return 0.0

    def _is_data_stale(self, target_timestamp: int) -> bool:
        """Проверка, устарели ли данные"""
        if not self._last_update_time:
            return True

        time_diff = target_timestamp - self._last_update_time
        max_stale_time = self.config.get("extrapolation", {}).get(
            "max_imu_time_sec", 120) * 1000

        return time_diff > max_stale_time

    def _get_vehicle_status(self) -> str:
        """Определение статуса погрузчика"""
        if not self._last_update_time:
            return "off"

        current_time = int(time.time() * 1000)
        time_diff = current_time - self._last_update_time

        off_timeout = self.config.get("power_management", {}).get(
            "inactivity_timeout_to_off_sec", 300) * 1000
        standby_timeout = self.config.get("power_management", {}).get(
            "inactivity_timeout_to_standby_sec", 30) * 1000

        if time_diff > off_timeout:
            return "off"
        elif time_diff > standby_timeout:
            return "standby"
        else:
            # Проверка активности (движения)
            if (self._has_imu_data and self.imu_handler.is_vehicle_moving()) or self.gps_handler.is_vehicle_moving():
                return "running"
            else:
                return "standby"

    def _get_degraded_state(self, timestamp: int) -> Dict[str, Any]:
        """Возвращение состояния DEGRADED"""
        return {
            "lat": None,
            "lon": None,
            "alt": None,
            "azimuth": None,
            "timestamp": timestamp,
            "confidence": 0.0,
            "status": "degraded",
            "source": "none",
            "vehicle_status": self._get_vehicle_status()
        }

    def start_magnetometer_calibration(self, duration_sec: int = 120):
        """
        Запуск калибровки магнитометра.

        Args:
            duration_sec (int): Длительность калибровки в секундах
        """
        with self._lock:
            # Проверяем наличие IMU данных
            if not self._has_imu_data:
                self.logger.warning(
                    "Невозможно запустить калибровку магнитометра: нет данных от IMU")
                return

            try:
                self.logger.info(
                    f"Начало калибровки магнитометра на {duration_sec} секунд")
                self._state = "CALIBRATING_MAG"

                # Запуск калибровки в отдельном потоке
                calibration_thread = threading.Thread(
                    target=self._run_magnetometer_calibration,
                    args=(duration_sec,)
                )
                calibration_thread.daemon = True
                calibration_thread.start()

            except Exception as e:
                self.logger.error(
                    f"Ошибка запуска калибровки магнитометра: {e}")
                self._state = "READY"

    def _run_magnetometer_calibration(self, duration_sec: int):
        """Выполнение запуска калибровки магнитометра 
           perform_magnetometer_calibration - запускает и сохраняет результаты в БД
        """
        try:
            calibration_data = self.calibration_manager.perform_magnetometer_calibration(
                duration_sec)
            self.logger.info("Калибровка магнитометра завершена успешно")
        except Exception as e:
            self.logger.error(f"Ошибка при калибровке магнитометра: {e}")
        finally:
            self._state = "READY"

    def get_history_stats(self) -> Dict[str, Any]:
        """
        Возвращает статистику по имеющимся данным.

        Returns:
            dict: Статистика данных
        """
        try:
            stats = self.database_manager.get_data_statistics()
            stats["module_state"] = self._state
            stats["vehicle_status"] = self._get_vehicle_status()
            stats["has_imu_data"] = self._has_imu_data
            stats["active_gps_sensors"] = self._get_active_gps_sensors()
            stats["configured_gps_sensors"] = self._gps_configured_sensors
            return stats
        except Exception as e:
            self.logger.error(f"Ошибка получения статистики: {e}")
            return {
                "error": str(e),
                "module_state": self._state,
                "vehicle_status": self._get_vehicle_status(),
                "has_imu_data": self._has_imu_data,
                "active_gps_sensors": self._get_active_gps_sensors(),
                "configured_gps_sensors": self._gps_configured_sensors
            }

    def stop(self):
        """Корректная остановка модуля, сохранение состояния"""
        try:
            self.logger.info("Остановка модуля PositionEstimator")

            # Сохранение текущего состояния
            state_data = {
                "status": self._state,
                "timestamp": int(time.time() * 1000),
                "last_update": self._last_update_time,
                "has_imu_data": self._has_imu_data,
                "gps_timestamps": self._gps_last_timestamps
            }
            self.database_manager.save_state(state_data)

            # Остановка компонентов
            self.database_manager.close()

            self.logger.info("Модуль PositionEstimator остановлен")

        except Exception as e:
            self.logger.error(f"Ошибка при остановке модуля: {e}")
