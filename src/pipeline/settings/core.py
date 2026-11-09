from pydantic_settings import BaseSettings


class CoreSettings(BaseSettings):
    class Config:
        env_file = ".env"
        use_enum_values = True
        populate_by_name = True
        extra = "ignore"
