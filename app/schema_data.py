from pydantic import BaseModel
from typing import List, Any


class HealthCheckResponse(BaseModel):
    status: str
    checks: List[Any]

class BaseParameter(BaseModel):
    parameter1: str