from api import PositionEstimator

# Инициализация модуля
estimator = PositionEstimator(config_path="/path/to/config.json")

# Прием данных от внешних датчиков (вызывается внешней системой)
estimator.update_gps(sensor_id=1, timestamp=1712345678901, lat=55.7558, lon=37.6173, alt=150.0)
estimator.update_imu({
    "timestamp": 1712345678910,
    "accelerometer": {"x": 0.1, "y": -0.05, "z": 9.81},
    "gyroscope": {"x": 0.0, "y": 0.0, "z": 0.5},
    "magnetometer": {"x": 25.0, "y": -10.0, "z": 40.0},
    "temperature": 30.0,
    "status": "ok"
})

# Запрос текущего состояния (вызывается внешней системой)
current_state = estimator.get_current_state()

print(current_state)
# Вывод будет примерно таким (в зависимости от данных и состояния):
# {
#   "lat": 55.75581,
#   "lon": 37.61732,
#   "alt": 150.1,
#   "azimuth": 45.2,
#   "timestamp": 1712345678950,
#   "confidence": 0.92,
#   "status": "gps_imu_fusion",
#   "source": "fused",
#   "vehicle_status": "running"
# }

# Остановка модуля при завершении работы
estimator.stop()