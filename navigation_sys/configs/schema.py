# storage/config/schema.py
from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator, root_validator
import logging



class Offset(BaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


class GPSSensor(BaseModel):
    id: int
    offset: Offset


class GPSValidation(BaseModel):
    max_hdop: float = Field(2.5, ge=0)
    max_speed_ms: float = Field(20.0, ge=0)
    max_position_jump_speed: float = Field(100.0, ge=0)
    max_stale_time_ms: int = Field(5000, ge=0)
    min_time_between_points_ms: int = Field(100, ge=0)
    outlier_check_enabled: bool = True


class GPS(BaseModel):
    sensors: List[GPSSensor] = []
    validation: GPSValidation = GPSValidation()
    #defaults for Krasnoyarsk
    max_lon: float = Field(92, ge=-180, le=180)
    min_lon: float = Field(91, ge=-180, le=180)
    max_lat: float = Field(57, ge=-180, le=180)
    min_lat: float = Field(55, ge=-180, le=180)
    history_window_ms: int = Field(18000000, ge=0) # допустимое окно 5 часов для поиска истории перемещений
    
    
class Database(BaseModel):
    path: str
    retention_minutes: int = 300
    batch_size: int = Field(10, ge=0)
    batch_timeout: float = Field(0, ge=0., le=5.)


class IMU(BaseModel):
    offset: Offset = Offset()
    gyro_units: str = "rad_s"
    calibration: Dict[str, Any] = {}


class Fusion(BaseModel):
    kf_process_variance: float = 0.1
    kf_gps_measurement_variance: float = 0.5
    complementary_filter_alpha: float = 0.98


class Interpolation(BaseModel):
    time_window_ms: int = 1000
    max_points: int = 10


class Extrapolation(BaseModel):
    max_imu_time_sec: int = 120
    max_gps_only_time_sec: int = 15


class PowerManagement(BaseModel):
    inactivity_timeout_to_standby_sec: int = 30
    inactivity_timeout_to_off_sec: int = 300


class System(BaseModel):
    request_timeout_ms: int = 1000
    default_confidence_threshold: float = 0.1


class Logging(BaseModel):
    level: str = Field("INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    directory: str = "logs_dir"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    max_file_size_mb: int = Field(10, ge=1, le=100)
    backup_count: int = Field(5, ge=1, le=20)
    console_output: bool = True
    
    @validator('level')
    def validate_level(cls, v):
        """Проверяем, что уровень логирования корректный"""
        valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        if v.upper() not in valid_levels:
            raise ValueError(f'Invalid log level: {v}. Must be one of {valid_levels}')
        return v.upper()
    
    def get_log_level(self) -> int:
        """Возвращает числовой уровень логирования для logging модуля"""
        return getattr(logging, self.level)


class RootConfig(BaseModel):
    logging: Logging = Logging() 
    database: Database
    gps: GPS
    imu: IMU
    fusion: Fusion
    interpolation: Interpolation
    extrapolation: Extrapolation
    power_management: PowerManagement
    system: System

    # Позволяем «добавлять» произвольные секции, если они нужны в будущем
    class Config:
        extra = "allow"
