from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
import pytest

from pga_sim.auth import AuthConfigurationError, RequestAuthenticator
from pga_sim.config import Settings


def _build_app(settings: Settings) -> FastAPI:
    authenticator = RequestAuthenticator(settings)
    authenticator.validate_configuration()

    app = FastAPI()

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        try:
            authenticator.authenticate_request(request)
        except HTTPException as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail},
                headers=exc.headers or None,
            )
        return await call_next(request)

    def require_admin(request: Request) -> None:
        authenticator.require_role(request, "admin")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/protected")
    async def protected(request: Request) -> dict[str, object]:
        user = authenticator.current_user(request)
        return {
            "mode": authenticator.mode,
            "authenticated": user is not None,
            "roles": sorted(user.roles) if user is not None else [],
        }

    @app.post("/learning/sync-train")
    async def learning_sync_train(_: None = Depends(require_admin)) -> dict[str, bool]:
        return {"ok": True}

    return app


def test_auth_none_mode_allows_requests_without_token() -> None:
    app = _build_app(Settings(_env_file=None, app_auth_mode="none"))
    client = TestClient(app)
    response = client.get("/protected")
    assert response.status_code == 200
    assert response.json()["authenticated"] is True


def test_basic_auth_requires_credentials() -> None:
    app = _build_app(
        Settings(
            _env_file=None,
            app_auth_mode="basic_auth",
            app_auth_basic_username="paul",
            app_auth_basic_password="secret-password",
        )
    )
    client = TestClient(app)
    response = client.get("/protected")
    assert response.status_code == 401


def test_basic_auth_accepts_valid_credentials() -> None:
    app = _build_app(
        Settings(
            _env_file=None,
            app_auth_mode="basic_auth",
            app_auth_basic_username="paul",
            app_auth_basic_password="secret-password",
        )
    )
    client = TestClient(app)
    response = client.get("/protected", auth=("paul", "secret-password"))
    assert response.status_code == 200
    assert response.json()["authenticated"] is True


def test_basic_auth_admin_endpoint_requires_admin_role() -> None:
    app = _build_app(
        Settings(
            _env_file=None,
            app_auth_mode="basic_auth",
            app_auth_basic_username="paul",
            app_auth_basic_password="secret-password",
            app_auth_basic_role="user",
        )
    )
    client = TestClient(app)
    response = client.post("/learning/sync-train", auth=("paul", "secret-password"))
    assert response.status_code == 403


def test_basic_auth_mode_requires_configured_credentials() -> None:
    authenticator = RequestAuthenticator(
        Settings(
            _env_file=None,
            app_auth_mode="basic_auth",
            app_auth_basic_username="",
            app_auth_basic_password="",
        )
    )
    with pytest.raises(AuthConfigurationError):
        authenticator.validate_configuration()


def test_shared_token_requires_bearer_token() -> None:
    app = _build_app(
        Settings(
            _env_file=None,
            app_auth_mode="shared_token",
            app_auth_shared_token="secret-token",
        )
    )
    client = TestClient(app)
    response = client.get("/protected")
    assert response.status_code == 401


def test_shared_token_accepts_valid_bearer_token() -> None:
    app = _build_app(
        Settings(
            _env_file=None,
            app_auth_mode="shared_token",
            app_auth_shared_token="secret-token",
        )
    )
    client = TestClient(app)
    response = client.get(
        "/protected",
        headers={"Authorization": "Bearer secret-token"},
    )
    assert response.status_code == 200
    assert response.json()["authenticated"] is True


def test_shared_token_admin_endpoint_requires_admin_role() -> None:
    app = _build_app(
        Settings(
            _env_file=None,
            app_auth_mode="shared_token",
            app_auth_shared_token="secret-token",
            app_auth_shared_token_role="user",
        )
    )
    client = TestClient(app)
    response = client.post(
        "/learning/sync-train",
        headers={"Authorization": "Bearer secret-token"},
    )
    assert response.status_code == 403


def test_shared_token_admin_endpoint_allows_admin_role() -> None:
    app = _build_app(
        Settings(
            _env_file=None,
            app_auth_mode="shared_token",
            app_auth_shared_token="secret-token",
            app_auth_shared_token_role="admin",
        )
    )
    client = TestClient(app)
    response = client.post(
        "/learning/sync-train",
        headers={"Authorization": "Bearer secret-token"},
    )
    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_auth_exempt_paths_bypass_authentication() -> None:
    app = _build_app(
        Settings(
            _env_file=None,
            app_auth_mode="shared_token",
            app_auth_shared_token="secret-token",
            app_auth_exempt_paths="/health",
        )
    )
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_shared_token_mode_requires_configured_token() -> None:
    authenticator = RequestAuthenticator(
        Settings(
            _env_file=None,
            app_auth_mode="shared_token",
            app_auth_shared_token="",
        )
    )
    with pytest.raises(AuthConfigurationError):
        authenticator.validate_configuration()
