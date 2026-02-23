from functools import lru_cache
from datetime import datetime, timezone

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    datagolf_api_key: str = ""
    datagolf_base_url: str = "https://feeds.datagolf.com"
    http_timeout_seconds: float = 20.0
    default_simulations: int = 1_000_000
    simulation_max_batch_size: int = 2_000
    learning_database_path: str = ".pga_sim_learning.sqlite3"
    app_auth_mode: str = "none"
    app_auth_exempt_paths: str = "/health"
    app_auth_shared_token: str = ""
    app_auth_shared_token_role: str = "admin"
    app_auth_basic_username: str = ""
    app_auth_basic_password: str = ""
    app_auth_basic_role: str = "admin"
    auth_allowed_emails: str = ""
    auth_allowed_email_domains: str = ""
    auth_admin_emails: str = ""
    auth_admin_email_domains: str = ""
    auth_admin_subjects: str = ""
    cloudflare_access_team_domain: str = ""
    cloudflare_access_audience: str = ""
    lifecycle_automation_enabled: bool = True
    lifecycle_automation_interval_seconds: int = 600
    lifecycle_tour: str = "pga"
    lifecycle_pre_event_simulations: int = 1_000_000
    lifecycle_pre_event_seed: int = 20260223
    lifecycle_sync_max_events: int = 40
    lifecycle_backfill_enabled: bool = True
    lifecycle_backfill_batch_size: int = 25
    lifecycle_target_year: int = datetime.now(timezone.utc).year

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
