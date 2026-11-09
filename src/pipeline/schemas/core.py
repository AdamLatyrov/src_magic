from pydantic import BaseModel


class CoreSchema(BaseModel):
    class Config:
        use_enum_values = True
