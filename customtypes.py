import enum
from pydantic import BaseModel


class DataItemType(BaseModel):
    dered_i: float
    dered_z: float
    dered_u: float
    dered_g: float
    dered_r: float
    extinction_r: float
    run: float
    camCol: float
    field: float
    obj: float
    photoz: float
    ra: float
    dec: float


class PredItemType(enum.Enum):
    zero = 0
    one = 1
    two = 2
