from pydantic import BaseModel


class PredictionInput(BaseModel):
    features: list[float]

    class config:
        schema_extra = {
            "example": {
                "features": [0.5, 0.2, 0.1, 0.8]
            }
        }
