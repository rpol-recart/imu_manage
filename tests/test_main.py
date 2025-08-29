from navigation_sys.api import PositionEstimator

def test_esimator():
    estimator = PositionEstimator(config_path="/path/to/config.json")
    estimator.update_gps(sensor_id=1, timestamp=1712345678901, lat=55.7558, lon=37.6173, alt=150.0)