from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    datagolf_api_key: str = ""
    datagolf_base_url: str = "https://feeds.datagolf.com"
    http_timeout_seconds: float = 20.0
    default_simulations: int = 10_000
    learning_database_path: str = ".pga_sim_learning.sqlite3"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
