from __future__ import annotations

import base64
from dataclasses import dataclass
import secrets
from typing import Any

from fastapi import HTTPException, Request

from .config import Settings


class AuthConfigurationError(RuntimeError):
    pass


@dataclass
class AuthenticatedUser:
    subject: str
    email: str | None
    auth_mode: str
    roles: set[str]


class RequestAuthenticator:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._mode = settings.app_auth_mode.strip().lower()
        self._exempt_paths = _parse_path_prefixes(settings.app_auth_exempt_paths)
        self._shared_token = settings.app_auth_shared_token.strip()
        self._shared_token_role = settings.app_auth_shared_token_role.strip().lower()
        self._basic_username = settings.app_auth_basic_username.strip()
        self._basic_password = settings.app_auth_basic_password
        self._basic_role = settings.app_auth_basic_role.strip().lower()

        self._allowed_emails = _parse_email_list(settings.auth_allowed_emails)
        self._allowed_domains = _parse_domain_list(settings.auth_allowed_email_domains)
        self._admin_emails = _parse_email_list(settings.auth_admin_emails)
        self._admin_domains = _parse_domain_list(settings.auth_admin_email_domains)
        self._admin_subjects = _parse_text_set(settings.auth_admin_subjects)

        self._cf_team_domain = settings.cloudflare_access_team_domain.strip().lower()
        self._cf_audience = settings.cloudflare_access_audience.strip()
        self._cf_jwks_client: Any = None
        self._jwt_module: Any = None

    @property
    def mode(self) -> str:
        return self._mode

    def validate_configuration(self) -> None:
        if self._mode not in {"none", "shared_token", "cloudflare_access", "basic_auth"}:
            raise AuthConfigurationError(
                "APP_AUTH_MODE must be one of: none, shared_token, cloudflare_access, basic_auth."
            )
        if self._mode == "shared_token":
            if not self._shared_token:
                raise AuthConfigurationError(
                    "APP_AUTH_SHARED_TOKEN must be set when APP_AUTH_MODE=shared_token."
                )
            if self._shared_token_role not in {"user", "admin"}:
                raise AuthConfigurationError(
                    "APP_AUTH_SHARED_TOKEN_ROLE must be one of: user, admin."
                )
            return
        if self._mode == "basic_auth":
            if not self._basic_username or not self._basic_password:
                raise AuthConfigurationError(
                    "APP_AUTH_BASIC_USERNAME and APP_AUTH_BASIC_PASSWORD must be set when APP_AUTH_MODE=basic_auth."
                )
            if self._basic_role not in {"user", "admin"}:
                raise AuthConfigurationError(
                    "APP_AUTH_BASIC_ROLE must be one of: user, admin."
                )
            return
        if self._mode == "cloudflare_access":
            if not self._cf_team_domain:
                raise AuthConfigurationError(
                    "CLOUDFLARE_ACCESS_TEAM_DOMAIN is required when APP_AUTH_MODE=cloudflare_access."
                )
            if not self._cf_audience:
                raise AuthConfigurationError(
                    "CLOUDFLARE_ACCESS_AUDIENCE is required when APP_AUTH_MODE=cloudflare_access."
                )
            # Fail fast if JWT dependency is unavailable.
            self._load_jwt_support()

    def authenticate_request(self, request: Request) -> AuthenticatedUser | None:
        if self._is_exempt(request.url.path):
            return None

        if self._mode == "none":
            user = AuthenticatedUser(
                subject="local-dev",
                email=None,
                auth_mode="none",
                roles={"user", "admin"},
            )
        elif self._mode == "basic_auth":
            user = self._authenticate_basic_auth(request)
        elif self._mode == "shared_token":
            user = self._authenticate_shared_token(request)
        elif self._mode == "cloudflare_access":
            user = self._authenticate_cloudflare(request)
        else:
            raise AuthConfigurationError(
                "APP_AUTH_MODE must be one of: none, shared_token, cloudflare_access, basic_auth."
            )

        request.state.auth_user = user
        return user

    def current_user(self, request: Request) -> AuthenticatedUser | None:
        return getattr(request.state, "auth_user", None)

    def require_role(self, request: Request, role: str) -> None:
        if self._mode == "none":
            return
        required = role.strip().lower()
        if not required:
            return
        user = self.current_user(request)
        if user is None:
            raise HTTPException(status_code=401, detail="Authentication required.")
        if required not in user.roles:
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions: '{required}' role required.",
            )

    def _authenticate_shared_token(self, request: Request) -> AuthenticatedUser:
        token = _bearer_token(request)
        if not token:
            raise HTTPException(
                status_code=401,
                detail="Missing bearer token.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if not secrets.compare_digest(token, self._shared_token):
            raise HTTPException(
                status_code=401,
                detail="Invalid bearer token.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        roles = {"user"}
        if self._shared_token_role == "admin":
            roles.add("admin")
        return AuthenticatedUser(
            subject="shared-token-user",
            email=None,
            auth_mode="shared_token",
            roles=roles,
        )

    def _authenticate_basic_auth(self, request: Request) -> AuthenticatedUser:
        username, password = _basic_credentials(request)
        if not username:
            raise HTTPException(
                status_code=401,
                detail="Missing basic auth credentials.",
                headers={"WWW-Authenticate": 'Basic realm="PGA Simulator"'},
            )

        if not secrets.compare_digest(username, self._basic_username) or not secrets.compare_digest(
            password, self._basic_password
        ):
            raise HTTPException(
                status_code=401,
                detail="Invalid basic auth credentials.",
                headers={"WWW-Authenticate": 'Basic realm="PGA Simulator"'},
            )

        roles = {"user"}
        if self._basic_role == "admin":
            roles.add("admin")
        return AuthenticatedUser(
            subject=f"basic:{username}",
            email=None,
            auth_mode="basic_auth",
            roles=roles,
        )

    def _authenticate_cloudflare(self, request: Request) -> AuthenticatedUser:
        token = request.headers.get("cf-access-jwt-assertion", "").strip()
        if not token:
            raise HTTPException(
                status_code=401,
                detail=(
                    "Missing Cloudflare Access assertion token. "
                    "Ensure requests flow through Cloudflare Access."
                ),
            )

        jwt_module, jwk_client = self._load_jwt_support()
        try:
            signing_key = jwk_client.get_signing_key_from_jwt(token)
            payload = jwt_module.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                audience=self._cf_audience,
                issuer=f"https://{self._cf_team_domain}",
                options={"require": ["sub", "iss", "aud", "exp"]},
            )
        except Exception as exc:  # pragma: no cover - depends on external token/JWKS responses
            raise HTTPException(
                status_code=401,
                detail="Invalid Cloudflare Access token.",
            ) from exc

        subject = str(payload.get("sub") or "").strip()
        if not subject:
            raise HTTPException(status_code=401, detail="Cloudflare Access token missing subject.")

        email_header = request.headers.get("cf-access-authenticated-user-email")
        email_claim = payload.get("email")
        email = _normalize_email(email_header or email_claim)

        if self._allowed_emails:
            if not email or email not in self._allowed_emails:
                raise HTTPException(status_code=403, detail="Email is not authorized for this app.")
        if self._allowed_domains:
            domain = _email_domain(email)
            if not domain or domain not in self._allowed_domains:
                raise HTTPException(status_code=403, detail="Email domain is not authorized.")

        roles = {"user"}
        if self._is_admin(subject=subject, email=email):
            roles.add("admin")

        return AuthenticatedUser(
            subject=subject,
            email=email,
            auth_mode="cloudflare_access",
            roles=roles,
        )

    def _is_exempt(self, path: str) -> bool:
        normalized = str(path or "/").strip()
        if not normalized.startswith("/"):
            normalized = f"/{normalized}"
        for prefix in self._exempt_paths:
            if normalized == prefix or normalized.startswith(f"{prefix}/"):
                return True
        return False

    def _is_admin(self, *, subject: str, email: str | None) -> bool:
        if subject and subject.lower() in self._admin_subjects:
            return True
        if email:
            lowered = email.lower()
            if lowered in self._admin_emails:
                return True
            domain = _email_domain(lowered)
            if domain and domain in self._admin_domains:
                return True
        return False

    def _load_jwt_support(self) -> tuple[Any, Any]:
        if self._jwt_module is not None and self._cf_jwks_client is not None:
            return self._jwt_module, self._cf_jwks_client

        try:
            import jwt
            from jwt import PyJWKClient
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise AuthConfigurationError(
                "PyJWT is required for APP_AUTH_MODE=cloudflare_access. "
                "Install with: pip install 'PyJWT[crypto]>=2.9.0'"
            ) from exc

        self._jwt_module = jwt
        self._cf_jwks_client = PyJWKClient(
            f"https://{self._cf_team_domain}/cdn-cgi/access/certs"
        )
        return self._jwt_module, self._cf_jwks_client


def _bearer_token(request: Request) -> str:
    value = request.headers.get("authorization", "").strip()
    if not value:
        return ""
    scheme, _, token = value.partition(" ")
    if scheme.strip().lower() != "bearer":
        return ""
    return token.strip()


def _basic_credentials(request: Request) -> tuple[str, str]:
    value = request.headers.get("authorization", "").strip()
    if not value:
        return "", ""
    scheme, _, token = value.partition(" ")
    if scheme.strip().lower() != "basic" or not token.strip():
        return "", ""

    encoded = token.strip()
    try:
        decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
    except Exception:
        return "", ""

    username, sep, password = decoded.partition(":")
    if not sep:
        return "", ""
    return username, password


def _normalize_email(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text or "@" not in text:
        return None
    return text


def _email_domain(email: str | None) -> str | None:
    if not email or "@" not in email:
        return None
    _, _, domain = email.rpartition("@")
    return domain.strip().lower() or None


def _parse_path_prefixes(value: str) -> tuple[str, ...]:
    prefixes: list[str] = []
    for item in _split_csv(value):
        normalized = item.strip()
        if not normalized:
            continue
        if not normalized.startswith("/"):
            normalized = f"/{normalized}"
        prefixes.append(normalized.rstrip("/") or "/")
    if not prefixes:
        return ("/health",)
    return tuple(prefixes)


def _parse_email_list(value: str) -> set[str]:
    out: set[str] = set()
    for item in _split_csv(value):
        normalized = _normalize_email(item)
        if normalized:
            out.add(normalized)
    return out


def _parse_domain_list(value: str) -> set[str]:
    out: set[str] = set()
    for item in _split_csv(value):
        normalized = str(item).strip().lower().lstrip("@")
        if normalized:
            out.add(normalized)
    return out


def _parse_text_set(value: str) -> set[str]:
    out: set[str] = set()
    for item in _split_csv(value):
        normalized = str(item).strip().lower()
        if normalized:
            out.add(normalized)
    return out


def _split_csv(value: str) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]
