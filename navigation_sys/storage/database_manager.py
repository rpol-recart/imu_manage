# /storage/database_manager.py

import sqlite3
import threading
import time
import json
import os
from typing import Dict, Any, List, Optional, Tuple
import logging
from queue import Queue, Empty
from ..configs import ConfigProvider
# Предполагается, что LoggerService уже реализован
from ..services.logger_service import LoggerService


class DatabaseManager:
    """
    Управляет асинхронной записью данных датчиков в SQLite базу данных
    и предоставляет методы для их извлечения.
    """

    def __init__(self, config: ConfigProvider):
        """
        Инициализирует менеджер базы данных.

        Args:
            config (dict): Конфигурация модуля, включая настройки БД.
        """
        self.logger = LoggerService.get_logger(self.__class__.__name__)
        self._config = config
        self._db_path = self._config.data.database.path

        # Настройки записи
        # Записывать каждые N сообщений
        self._batch_size = self._config.data.database.batch_size
        self._batch_timeout_sec = self._config.data.database.batch_timeout  # Или каждые N секунд

        # Очередь для асинхронной записи
        self._write_queue: Queue = Queue()
        self._stop_event = threading.Event()

        # Блокировка для потокобезопасности при работе с соединением
        # (хотя основная запись в отдельном потоке, соединение может использоваться и для чтения)
        self._db_lock = threading.RLock()

        # Инициализация БД и запуск фонового потока
        self._init_database()
        self._writer_thread = threading.Thread(
            target=self._writer_worker, daemon=True, name="DBWriterThread")
        self._writer_thread.start()
        self.logger.info(
            f"DatabaseManager инициализирован. БД: {self._db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Получает новое соединение с БД. Используется в основном потоке записи."""
        conn = sqlite3.connect(
            # check_same_thread=False для использования в worker
            self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row  # Для удобного доступа к столбцам по имени
        return conn

    def _init_database(self):
        """Создает таблицы в базе данных, если они не существуют."""
        try:
            with self._db_lock:
                # Используем временное соединение для инициализации
                conn = self._get_connection()
                cursor = conn.cursor()

                # Создание таблиц
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
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS calibration (
                        id INTEGER PRIMARY KEY,
                        sensor_name TEXT NOT NULL UNIQUE,
                        parameters_json TEXT NOT NULL,
                        date_calibrated INTEGER NOT NULL
                    )
                """)
                # Создание индексов
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_sensor_data_timestamp ON sensor_data(timestamp)")
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_sensor_data_type ON sensor_data(sensor_type)")
                # Создание таблицы для хранения состояния модуля (если нужно)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS module_state (
                        id INTEGER PRIMARY KEY CHECK (id = 1), -- Ограничиваем одну запись
                        state_json TEXT NOT NULL,
                        saved_at INTEGER DEFAULT (strftime('%s','now') * 1000)
                    )
                """)

                conn.commit()
                conn.close()
                self.logger.info("Схема базы данных инициализирована.")
        except Exception as e:
            self.logger.error(f"Ошибка инициализации базы данных: {e}")
            raise

    def _writer_worker(self):
        """Фоновый поток для асинхронной записи данных в БД."""
        batch = []
        last_write_time = time.time()

        while not self._stop_event.is_set():
            try:
                # Ждем данные из очереди с таймаутом для проверки batch_timeout
                item = self._write_queue.get(timeout=0.1)
                batch.append(item)

                current_time = time.time()
                # Записываем, если накопился batch или истек таймаут
                if (len(batch) >= self._batch_size) or (current_time - last_write_time >= self._batch_timeout_sec):
                    self._write_batch(batch)
                    batch.clear()
                    last_write_time = current_time

            except Empty:
                # Таймаут очереди истек, проверяем таймаут для неполного батча
                current_time = time.time()
                if batch and (current_time - last_write_time >= self._batch_timeout_sec):
                    self._write_batch(batch)
                    batch.clear()
                    last_write_time = current_time
            except Exception as e:
                self.logger.error(f"Ошибка в потоке записи БД: {e}")

        # Записываем оставшиеся данные при остановке
        if batch:
            self._write_batch(batch)
        self.logger.info("Поток записи в БД остановлен.")

    def _write_batch(self, batch: List[Tuple]):
        """Записывает пакет данных в БД."""
        if not batch:
            return
        try:
            # Используем временное соединение для записи
            conn = self._get_connection()
            cursor = conn.cursor()
            # Используем executemany для эффективности
            cursor.executemany(
                "INSERT INTO sensor_data (timestamp, sensor_type, sensor_id, data_json) VALUES (?, ?, ?, ?)",
                batch
            )
            conn.commit()
            conn.close()
            self.logger.debug(
                f"Записан пакет данных в БД: {len(batch)} записей.")
        except Exception as e:
            self.logger.error(f"Ошибка записи пакета данных в БД: {e}")
            # В реальной системе здесь можно добавить логику повтора или сохранения в файл

    def save_sensor_data(self, timestamp: int, sensor_type: str, sensor_id: Optional[int], data_json_str: str):
        """
        Асинхронно сохраняет данные датчика в очередь для записи в БД.

        Args:
            timestamp (int): Временная метка Unix (мс).
            sensor_type (str): Тип датчика ('gps', 'imu_raw', 'orientation').
            sensor_id (int, optional): ID датчика (1, 2 для GPS).
            data (str): Данные датчика в формате словаря сериализованные в JSON
        """
        try:
            #data_json_str = json.dumps(data, ensure_ascii=False)
            # Кладем данные в очередь для обработки фоновым потоком
            self._write_queue.put(
                (timestamp, sensor_type, sensor_id, data_json_str))
        except Exception as e:
            self.logger.error(f"Ошибка постановки данных в очередь БД: {e}")

    def save_calibration(self, sensor_name: str, parameters: Dict[str, Any], date_calibrated: int = None):
        """
        Сохраняет или обновляет калибровочные параметры.

        Args:
            sensor_name (str): Имя сенсора/параметра калибровки.
            parameters (dict): Калибровочные параметры.
            date_calibrated (int, optional): Время калибровки (Unix мс). По умолчанию текущее время.
        """
        if date_calibrated is None:
            date_calibrated = int(time.time() * 1000)

        try:
            parameters_json_str = json.dumps(parameters, ensure_ascii=False)
            with self._db_lock:
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO calibration (sensor_name, parameters_json, date_calibrated)
                    VALUES (?, ?, ?)
                """, (sensor_name, parameters_json_str, date_calibrated))
                conn.commit()
                conn.close()
            self.logger.info(
                f"Калибровочные данные для '{sensor_name}' сохранены.")
        except Exception as e:
            self.logger.error(
                f"Ошибка сохранения калибровочных данных для '{sensor_name}': {e}")

    def get_calibration(self, sensor_name: str) -> Optional[Dict[str, Any]]:
        """
        Получает калибровочные параметры по имени.

        Args:
            sensor_name (str): Имя сенсора/параметра калибровки.

        Returns:
            dict or None: Словарь с параметрами или None, если не найдено.
        """
        try:
            with self._db_lock:
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT parameters_json FROM calibration WHERE sensor_name = ?
                """, (sensor_name,))
                row = cursor.fetchone()
                conn.close()

                if row:
                    return json.loads(row['parameters_json'])
                else:
                    self.logger.debug(
                        f"Калибровочные данные для '{sensor_name}' не найдены.")
                    return None
        except Exception as e:
            self.logger.error(
                f"Ошибка получения калибровочных данных для '{sensor_name}': {e}")
            return None

    def get_data_for_time(self, target_timestamp: int, window_ms: int = 1000) -> Dict[str, Any]:
        """
        Получает данные датчиков вокруг указанного времени для фузии.

        Args:
            target_timestamp (int): Целевое время (Unix мс).
            window_ms (int): Окно поиска данных (по умолчанию ±500 мс).

        Returns:
            dict: Словарь с данными датчиков, структурированный для фузии.
                  Пример: {'gps': {1: {...}, 2: {...}}, 'imu': {...}}
        """
        half_window = window_ms // 2
        start_time = target_timestamp - self._config.data.gps.history_window_ms
        end_time = target_timestamp + half_window
        result_data = {'gps': [], 'imu': []}

        try:
            with self._db_lock:
                conn = self._get_connection()
                cursor = conn.cursor()

                # Получаем последние GPS данные до или в окне для каждого сенсора
                gps_sensors_config = self._config.data.gps.sensors
                for sensor_cfg in gps_sensors_config:
                    sensor_id = sensor_cfg.id
                    if sensor_id is not None:
                        cursor.execute("""
                            SELECT * FROM sensor_data
                            WHERE sensor_type = 'gps' AND sensor_id = ? AND timestamp BETWEEN ? AND ?
                            ORDER BY timestamp DESC LIMIT 15
                        """, (sensor_id, start_time, end_time))
                        rows = cursor.fetchall()

                        for row in rows:
                            if row:
                                try:
                                    result_data['gps'].append({
                                        'timestamp': row['timestamp'],
                                        'data': json.loads(row['data_json']),
                                        'sensor_id': row['sensor_id']
                                    })
                                except json.JSONDecodeError:
                                    self.logger.error(
                                        f"Ошибка декодирования JSON для GPS {sensor_id} в момент {row['timestamp']}")

                # Получаем последние IMU данные (предполагаем, что тип 'imu_raw' или 'orientation')
                # Можно уточнить, какой тип использовать или брать оба
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
                            # Уточняем тип IMU данных
                            'type': row['sensor_type'],
                            'data': json.loads(row['data_json'])
                        }
                    except json.JSONDecodeError:
                        self.logger.error(
                            f"Ошибка декодирования JSON для IMU в момент {row['timestamp']}")

                conn.close()
            self.logger.debug(
                f"Получены данные для фузии на время {target_timestamp}: GPS{len(result_data['gps'])} записей, IMU: {'type' in result_data['imu']}")
            return result_data

        except Exception as e:
            self.logger.error(
                f"Ошибка получения данных для фузии на время {target_timestamp}: {e}")
            return {'gps': {}, 'imu': {}}

    def get_last_state(self) -> Optional[Dict[str, Any]]:
        """
        Восстанавливает последнее сохраненное состояние модуля из БД.

        Returns:
            dict or None: Словарь с состоянием или None, если не найдено.
        """
        try:
            with self._db_lock:
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT state_json FROM module_state WHERE id = 1")
                row = cursor.fetchone()
                conn.close()

                if row:
                    return json.loads(row['state_json'])
                else:
                    self.logger.debug(
                        "Сохраненное состояние модуля не найдено.")
                    return None
        except Exception as e:
            self.logger.error(f"Ошибка восстановления состояния модуля: {e}")
            return None

    def save_state(self, state_data: Dict[str, Any]):
        """
        Сохраняет текущее состояние модуля в БД.

        Args:
            state_data (dict): Данные состояния для сохранения.
        """
        try:
            state_json_str = json.dumps(state_data, ensure_ascii=False)
            with self._db_lock:
                conn = self._get_connection()
                cursor = conn.cursor()
                # Используем INSERT OR REPLACE для обновления единственной записи
                cursor.execute("""
                    INSERT OR REPLACE INTO module_state (id, state_json)
                    VALUES (1, ?)
                """, (state_json_str,))
                conn.commit()
                conn.close()
            self.logger.info("Состояние модуля сохранено в БД.")
        except Exception as e:
            self.logger.error(f"Ошибка сохранения состояния модуля: {e}")

    def cleanup_old_data(self, retention_minutes: int):
        """
        Удаляет старые данные из таблицы sensor_data.

        Args:
            retention_minutes (int): Время хранения данных в минутах.
        """
        try:
            cutoff_timestamp = int(time.time() * 1000) - \
                (retention_minutes * 60 * 1000)
            with self._db_lock:
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM sensor_data WHERE timestamp < ?
                """, (cutoff_timestamp,))
                deleted_count = cursor.rowcount
                conn.commit()
                conn.close()
            if deleted_count > 0:
                self.logger.info(
                    f"Очищено {deleted_count} старых записей из sensor_data.")
        except Exception as e:
            self.logger.error(f"Ошибка очистки старых данных: {e}")

    def get_data_statistics(self) -> Dict[str, Any]:
        """
        Возвращает статистику по имеющимся данным.

        Returns:
            dict: Словарь со статистикой.
        """
        stats = {}
        try:
            with self._db_lock:
                conn = self._get_connection()
                cursor = conn.cursor()

                # Общее количество записей
                cursor.execute("SELECT COUNT(*) as total FROM sensor_data")
                stats['total_records'] = cursor.fetchone()['total']

                # Последнее обновление (максимальная временная метка)
                cursor.execute(
                    "SELECT MAX(timestamp) as last_update FROM sensor_data")
                row = cursor.fetchone()
                stats['last_update_timestamp'] = row['last_update'] if row['last_update'] else None

                # Количество записей по типам сенсоров
                cursor.execute("""
                    SELECT sensor_type, COUNT(*) as count FROM sensor_data GROUP BY sensor_type
                """)
                stats['records_by_type'] = {
                    row['sensor_type']: row['count'] for row in cursor.fetchall()}

                # Количество записей по ID GPS сенсоров
                cursor.execute("""
                    SELECT sensor_id, COUNT(*) as count FROM sensor_data WHERE sensor_type = 'gps' GROUP BY sensor_id
                """)
                stats['gps_records_by_id'] = {
                    row['sensor_id']: row['count'] for row in cursor.fetchall()}

                # Последняя калибровка
                cursor.execute(
                    "SELECT sensor_name, date_calibrated FROM calibration ORDER BY date_calibrated DESC LIMIT 1")
                row = cursor.fetchone()
                if row:
                    stats['last_calibration'] = {
                        'sensor_name': row['sensor_name'],
                        'date_calibrated': row['date_calibrated']
                    }

                conn.close()
            self.logger.debug("Статистика данных получена.")
            return stats
        except Exception as e:
            self.logger.error(f"Ошибка получения статистики данных: {e}")
            return {'error': str(e)}

    def close(self):
        """Корректно закрывает менеджер БД."""
        self.logger.info("Запрос на остановку DatabaseManager.")
        self._stop_event.set()
        if self._writer_thread.is_alive():
            self._writer_thread.join(timeout=2)  # Ждем завершения потока
            if self._writer_thread.is_alive():
                self.logger.warning("Поток записи БД не завершился вовремя.")
        self.logger.info("DatabaseManager закрыт.")
