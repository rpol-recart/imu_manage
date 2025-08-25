Вы правы, класс получился довольно объемным. Предлагаю разбить его на несколько компонентов с четким разделением ответственности:

## 1. Базовый менеджер соединений

```python
# /storage/database_connection.py

import sqlite3
import threading
from typing import Optional
from contextlib import contextmanager
from ..services.logger_service import LoggerService

class DatabaseConnection:
    """Управляет соединениями с базой данных."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.logger = LoggerService.get_logger(self.__class__.__name__)
        self._lock = threading.RLock()
        self._init_database()
    
    def _init_database(self):
        """Создает схему базы данных."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                self._create_tables(cursor)
                conn.commit()
                self.logger.info("Схема базы данных инициализирована.")
        except Exception as e:
            self.logger.error(f"Ошибка инициализации базы данных: {e}")
            raise
    
    def _create_tables(self, cursor):
        """Создает необходимые таблицы."""
        # Таблица данных сенсоров
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sensor_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                sensor_type TEXT NOT NULL,
                sensor_id INTEGER,
                data_json TEXT NOT NULL,
                created_at INTEGER DEFAULT (strftime('%s','now') * 1000)
            )
        """)
        
        # Таблица калибровки
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS calibration (
                id INTEGER PRIMARY KEY,
                sensor_name TEXT NOT NULL UNIQUE,
                parameters_json TEXT NOT NULL,
                date_calibrated INTEGER NOT NULL
            )
        """)
        
        # Таблица состояния модуля
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS module_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                state_json TEXT NOT NULL,
                saved_at INTEGER DEFAULT (strftime('%s','now') * 1000)
            )
        """)
        
        # Индексы
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sensor_data_timestamp ON sensor_data(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sensor_data_type ON sensor_data(sensor_type)")
    
    @contextmanager
    def get_connection(self):
        """Контекстный менеджер для получения соединения с БД."""
        conn = None
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                conn.row_factory = sqlite3.Row
                yield conn
        finally:
            if conn:
                conn.close()
```

## 2. Асинхронный писатель

```python
# /storage/database_writer.py

import threading
import time
import json
from queue import Queue, Empty
from typing import List, Tuple, Any, Dict, Optional
from .database_connection import DatabaseConnection
from ..services.logger_service import LoggerService

class DatabaseWriter:
    """Асинхронный писатель данных в базу данных."""
    
    def __init__(self, db_connection: DatabaseConnection, batch_size: int = 10, batch_timeout: float = 0.5):
        self.db_connection = db_connection
        self.logger = LoggerService.get_logger(self.__class__.__name__)
        
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        
        self._write_queue: Queue = Queue()
        self._stop_event = threading.Event()
        self._worker_thread = None
        
        self._start_worker()
    
    def _start_worker(self):
        """Запускает фоновый поток записи."""
        self._worker_thread = threading.Thread(
            target=self._worker_loop, 
            daemon=True, 
            name="DatabaseWriter"
        )
        self._worker_thread.start()
        self.logger.info("Поток асинхронной записи запущен.")
    
    def _worker_loop(self):
        """Основной цикл обработки очереди записи."""
        batch = []
        last_write_time = time.time()
        
        while not self._stop_event.is_set():
            try:
                item = self._write_queue.get(timeout=0.1)
                batch.append(item)
                
                current_time = time.time()
                if (len(batch) >= self.batch_size) or (current_time - last_write_time >= self.batch_timeout):
                    self._write_batch(batch)
                    batch.clear()
                    last_write_time = current_time
                    
            except Empty:
                current_time = time.time()
                if batch and (current_time - last_write_time >= self.batch_timeout):
                    self._write_batch(batch)
                    batch.clear()
                    last_write_time = current_time
            except Exception as e:
                self.logger.error(f"Ошибка в потоке записи: {e}")
        
        # Записываем оставшиеся данные при остановке
        if batch:
            self._write_batch(batch)
        
        self.logger.info("Поток записи остановлен.")
    
    def _write_batch(self, batch: List[Tuple]):
        """Записывает пакет данных в БД."""
        if not batch:
            return
            
        try:
            with self.db_connection.get_connection() as conn:
                cursor = conn.cursor()
                cursor.executemany(
                    "INSERT INTO sensor_data (timestamp, sensor_type, sensor_id, data_json) VALUES (?, ?, ?, ?)",
                    batch
                )
                conn.commit()
            self.logger.debug(f"Записан пакет: {len(batch)} записей")
        except Exception as e:
            self.logger.error(f"Ошибка записи пакета: {e}")
    
    def enqueue_data(self, timestamp: int, sensor_type: str, sensor_id: Optional[int], data: Dict[str, Any]):
        """Добавляет данные в очередь на запись."""
        try:
            data_json = json.dumps(data, ensure_ascii=False)
            self._write_queue.put((timestamp, sensor_type, sensor_id, data_json))
        except Exception as e:
            self.logger.error(f"Ошибка добавления данных в очередь: {e}")
    
    def stop(self):
        """Останавливает поток записи."""
        self.logger.info("Остановка писателя БД.")
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2)
            if self._worker_thread.is_alive():
                self.logger.warning("Поток записи не завершился вовремя.")
```

## 3. Читатель данных

```python
# /storage/database_reader.py

import json
import time
from typing import Dict, Any, Optional, List
from .database_connection import DatabaseConnection
from ..services.logger_service import LoggerService

class DatabaseReader:
    """Читает данные из базы данных."""
    
    def __init__(self, db_connection: DatabaseConnection, config: Dict[str, Any]):
        self.db_connection = db_connection
        self.config = config
        self.logger = LoggerService.get_logger(self.__class__.__name__)
    
    def get_data_for_time(self, target_timestamp: int, window_ms: int = 1000) -> Dict[str, Any]:
        """Получает данные датчиков вокруг указанного времени."""
        half_window = window_ms // 2
        start_time = target_timestamp - half_window
        end_time = target_timestamp + half_window
        result_data = {'gps': {}, 'imu': {}}
        
        try:
            with self.db_connection.get_connection() as conn:
                cursor = conn.cursor()
                
                # GPS данные
                self._get_gps_data(cursor, start_time, end_time, result_data)
                
                # IMU данные
                self._get_imu_data(cursor, start_time, end_time, result_data)
            
            self.logger.debug(f"Получены данные для времени {target_timestamp}")
            return result_data
            
        except Exception as e:
            self.logger.error(f"Ошибка получения данных для времени {target_timestamp}: {e}")
            return {'gps': {}, 'imu': {}}
    
    def _get_gps_data(self, cursor, start_time: int, end_time: int, result_data: Dict[str, Any]):
        """Получает GPS данные для указанного временного окна."""
        gps_sensors = self.config.get("gps", {}).get("sensors", [])
        
        for sensor_cfg in gps_sensors:
            sensor_id = sensor_cfg.get("id")
            if sensor_id is not None:
                cursor.execute("""
                    SELECT * FROM sensor_data
                    WHERE sensor_type = 'gps' AND sensor_id = ? AND timestamp BETWEEN ? AND ?
                    ORDER BY timestamp DESC LIMIT 1
                """, (sensor_id, start_time, end_time))
                
                row = cursor.fetchone()
                if row:
                    try:
                        result_data['gps'][sensor_id] = {
                            'timestamp': row['timestamp'],
                            'data': json.loads(row['data_json']),
                            'sensor_id': row['sensor_id']
                        }
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Ошибка JSON для GPS {sensor_id}: {e}")
    
    def _get_imu_data(self, cursor, start_time: int, end_time: int, result_data: Dict[str, Any]):
        """Получает IMU данные для указанного временного окна."""
        cursor.execute("""
            SELECT * FROM sensor_data
            WHERE sensor_type IN ('imu_raw', 'orientation') AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp DESC LIMIT 1
        """, (start_time, end_time))
        
        row = cursor.fetchone()
        if row:
            try:
                result_data['imu'] = {
                    'timestamp': row['timestamp'],
                    'type': row['sensor_type'],
                    'data': json.loads(row['data_json'])
                }
            except json.JSONDecodeError as e:
                self.logger.error(f"Ошибка JSON для IMU: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику по данным."""
        stats = {}
        try:
            with self.db_connection.get_connection() as conn:
                cursor = conn.cursor()
                
                # Общее количество
                cursor.execute("SELECT COUNT(*) as total FROM sensor_data")
                stats['total_records'] = cursor.fetchone()['total']
                
                # Последнее обновление
                cursor.execute("SELECT MAX(timestamp) as last_update FROM sensor_data")
                row = cursor.fetchone()
                stats['last_update_timestamp'] = row['last_update']
                
                # По типам сенсоров
                cursor.execute("SELECT sensor_type, COUNT(*) as count FROM sensor_data GROUP BY sensor_type")
                stats['records_by_type'] = {row['sensor_type']: row['count'] for row in cursor.fetchall()}
                
                # GPS по ID
                cursor.execute("SELECT sensor_id, COUNT(*) as count FROM sensor_data WHERE sensor_type = 'gps' GROUP BY sensor_id")
                stats['gps_records_by_id'] = {row['sensor_id']: row['count'] for row in cursor.fetchall()}
            
            return stats
        except Exception as e:
            self.logger.error(f"Ошибка получения статистики: {e}")
            return {'error': str(e)}
```

## 4. Менеджер калибровки и состояния

```python
# /storage/database_state_manager.py

import json
import time
from typing import Dict, Any, Optional
from .database_connection import DatabaseConnection
from ..services.logger_service import LoggerService

class DatabaseStateManager:
    """Управляет калибровкой и состоянием модуля."""
    
    def __init__(self, db_connection: DatabaseConnection):
        self.db_connection = db_connection
        self.logger = LoggerService.get_logger(self.__class__.__name__)
    
    def save_calibration(self, sensor_name: str, parameters: Dict[str, Any], date_calibrated: int = None):
        """Сохраняет калибровочные параметры."""
        if date_calibrated is None:
            date_calibrated = int(time.time() * 1000)
        
        try:
            parameters_json = json.dumps(parameters, ensure_ascii=False)
            with self.db_connection.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO calibration (sensor_name, parameters_json, date_calibrated)
                    VALUES (?, ?, ?)
                """, (sensor_name, parameters_json, date_calibrated))
                conn.commit()
            
            self.logger.info(f"Калибровка для '{sensor_name}' сохранена")
        except Exception as e:
            self.logger.error(f"Ошибка сохранения калибровки '{sensor_name}': {e}")
    
    def get_calibration(self, sensor_name: str) -> Optional[Dict[str, Any]]:
        """Получает калибровочные параметры."""
        try:
            with self.db_connection.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT parameters_json FROM calibration WHERE sensor_name = ?", (sensor_name,))
                row = cursor.fetchone()
                
                if row:
                    return json.loads(row['parameters_json'])
                return None
        except Exception as e:
            self.logger.error(f"Ошибка получения калибровки '{sensor_name}': {e}")
            return None
    
    def save_state(self, state_data: Dict[str, Any]):
        """Сохраняет состояние модуля."""
        try:
            state_json = json.dumps(state_data, ensure_ascii=False)
            with self.db_connection.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO module_state (id, state_json)
                    VALUES (1, ?)
                """, (state_json,))
                conn.commit()
            
            self.logger.info("Состояние модуля сохранено")
        except Exception as e:
            self.logger.error(f"Ошибка сохранения состояния: {e}")
    
    def get_last_state(self) -> Optional[Dict[str, Any]]:
        """Восстанавливает последнее состояние модуля."""
        try:
            with self.db_connection.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT state_json FROM module_state WHERE id = 1")
                row = cursor.fetchone()
                
                if row:
                    return json.loads(row['state_json'])
                return None
        except Exception as e:
            self.logger.error(f"Ошибка получения состояния: {e}")
            return None
```

## 5. Обновленный основной менеджер

```python
# /storage/database_manager.py

import time
from typing import Dict, Any, List, Optional
from .database_connection import DatabaseConnection
from .database_writer import DatabaseWriter
from .database_reader import DatabaseReader
from .database_state_manager import DatabaseStateManager
from ..services.logger_service import LoggerService

class DatabaseManager:
    """Главный менеджер базы данных с делегированием функций."""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = LoggerService.get_logger(self.__class__.__name__)
        self.config = config
        
        # Инициализация компонентов
        db_path = config.get("database", {}).get("path", "sensor_data.db")
        self.connection = DatabaseConnection(db_path)
        
        batch_size = config.get("database", {}).get("batch_size", 10)
        batch_timeout = config.get("database", {}).get("batch_timeout", 0.5)
        self.writer = DatabaseWriter(self.connection, batch_size, batch_timeout)
        
        self.reader = DatabaseReader(self.connection, config)
        self.state_manager = DatabaseStateManager(self.connection)
        
        self.logger.info(f"DatabaseManager инициализирован. БД: {db_path}")
    
    # Методы записи данных
    def save_sensor_data(self, timestamp: int, sensor_type: str, sensor_id: Optional[int], data: Dict[str, Any]):
        """Сохраняет данные датчика."""
        self.writer.enqueue_data(timestamp, sensor_type, sensor_id, data)
    
    # Методы чтения данных
    def get_data_for_time(self, target_timestamp: int, window_ms: int = 1000) -> Dict[str, Any]:
        """Получает данные для фузии."""
        return self.reader.get_data_for_time(target_timestamp, window_ms)
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику данных."""
        return self.reader.get_statistics()
    
    # Методы калибровки и состояния
    def save_calibration(self, sensor_name: str, parameters: Dict[str, Any], date_calibrated: int = None):
        """Сохраняет калибровочные параметры."""
        self.state_manager.save_calibration(sensor_name, parameters, date_calibrated)
    
    def get_calibration(self, sensor_name: str) -> Optional[Dict[str, Any]]:
        """Получает калибровочные параметры."""
        return self.state_manager.get_calibration(sensor_name)
    
    def save_state(self, state_data: Dict[str, Any]):
        """Сохраняет состояние модуля."""
        self.state_manager.save_state(state_data)
    
    def get_last_state(self) -> Optional[Dict[str, Any]]:
        """Восстанавливает состояние модуля."""
        return self.state_manager.get_last_state()
    
    # Утилиты
    def cleanup_old_data(self, retention_minutes: int):
        """Удаляет старые данные."""
        try:
            cutoff_timestamp = int(time.time() * 1000) - (retention_minutes * 60 * 1000)
            with self.connection.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM sensor_data WHERE timestamp < ?", (cutoff_timestamp,))
                deleted_count = cursor.rowcount
                conn.commit()
            
            if deleted_count > 0:
                self.logger.info(f"Очищено {deleted_count} старых записей")
        except Exception as e:
            self.logger.error(f"Ошибка очистки данных: {e}")
    
    def close(self):
        """Корректно закрывает менеджер."""
        self.logger.info("Закрытие DatabaseManager")
        self.writer.stop()
        self.logger.info("DatabaseManager закрыт")
```

## Преимущества новой архитектуры:

1. **Разделение ответственности**: каждый класс имеет четкую роль
2. **Тестируемость**: компоненты легко тестировать изолированно
3. **Расширяемость**: легко добавить новые типы операций
4. **Читаемость**: код стал более структурированным
5. **Повторное использование**: компоненты можно использовать независимо

Теперь каждый компонент фокусируется на своей задаче, а основной `DatabaseManager` выступает фасадом, предоставляя единый интерфейс.