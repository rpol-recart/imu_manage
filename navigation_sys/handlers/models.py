from pydantic import BaseModel, Field, validator, root_validator, conint, field_validator
from typing import Optional,Union,Literal
import math
from enum import Enum
from ..configs import CONFIG


class ValidationResult(Enum):
    """Результаты валидации GPS данных."""
    VALID = "valid"
    INVALID_COORDINATES = "invalid_coordinates"
    POOR_HDOP = "poor_hdop"
    EXCESSIVE_SPEED = "excessive_speed"
    POSITION_OUTLIER = "position_outlier"
    INVALID_NMEA_STATUS = "invalid_nmea_status"
    STALE_TIMESTAMP = "stale_timestamp"
  
    
class GPSCoordinates(BaseModel):
    lat: float = Field(..., description="Широта в градусах")
    lon: float = Field(..., description="Долгота в градусах")
    alt: float = Field(..., description="Высота над уровнем моря")

    @field_validator('lat')
    @classmethod
    def validate_lat(cls, v):
        
        min_lat = CONFIG.data.gps.min_lat
        max_lat = CONFIG.data.gps.max_lat
        if not (min_lat <= v <= max_lat):
            raise ValueError(
                f"Широта должна находиться в диапазоне "
                f"[{min_lat}, {max_lat}]. Получено: {v}"
            )
        return v

    @field_validator('lon')
    @classmethod
    def validate_lon(cls, v):
    
        min_lon = CONFIG.data.gps.min_lon
        max_lon = CONFIG.data.gps.max_lon
        if not (min_lon <= v <= max_lon):
            raise ValueError(
                f"Долгота должна находиться в диапазоне "
                f"[{min_lon}, {max_lon}]. Получено: {v}"
            )
        return v

    def distance_to(self, other: "GPSCoordinates") -> float:
        """Haversine‑distance в метрах ."""
        R = 6371000  # м
        lat1, lon1, lat2, lon2 = map(
            math.radians, [self.lat, self.lon, other.lat, other.lon]
        )
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c
