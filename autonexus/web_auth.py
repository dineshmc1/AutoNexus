"""Authentication boundary for the Auto Nexus Studio."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

from .exceptions import CapabilityError, ConfigurationError


class AuthenticationError(RuntimeError):
    """Raised when a Studio request has no valid identity."""


@dataclass(frozen=True)
class Principal:
    uid: str
    email: str | None = None
    name: str | None = None


class StudioAuthenticator:
    """Single-user loopback authentication used for local development."""

    mode = "local"

    def authenticate(self, authorization: str | None) -> Principal:
        del authorization
        return Principal(uid="local-user", name="Local operator")

    def public_config(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "required": False,
            "firebase": None,
            "warning": (
                "Loopback single-user mode. Configure Firebase before "
                "exposing the Studio to other machines."
            ),
        }


class FirebaseAuthenticator(StudioAuthenticator):
    """Verify Firebase ID tokens and expose only non-secret client config."""

    mode = "firebase"

    def __init__(
        self,
        *,
        api_key: str,
        project_id: str,
        auth_domain: str | None = None,
        app_id: str | None = None,
        token_verifier: Callable[[str], dict[str, Any]] | None = None,
    ) -> None:
        if not api_key or not project_id:
            raise ConfigurationError(
                "Firebase auth requires AUTONEXUS_FIREBASE_API_KEY and "
                "AUTONEXUS_FIREBASE_PROJECT_ID."
            )
        self.api_key = api_key
        self.project_id = project_id
        self.auth_domain = auth_domain or f"{project_id}.firebaseapp.com"
        self.app_id = app_id
        self._verify = token_verifier or self._firebase_verifier(project_id)

    @staticmethod
    def _firebase_verifier(
        project_id: str,
    ) -> Callable[[str], dict[str, Any]]:
        try:
            import firebase_admin
            from firebase_admin import auth
        except ImportError as exc:
            raise CapabilityError(
                'Firebase login requires: pip install "AutoNexus[auth]"'
            ) from exc
        try:
            app = firebase_admin.get_app()
        except ValueError:
            app = firebase_admin.initialize_app(
                options={"projectId": project_id}
            )

        def verify(token: str) -> dict[str, Any]:
            return auth.verify_id_token(
                token,
                app=app,
                check_revoked=True,
            )

        return verify

    def authenticate(self, authorization: str | None) -> Principal:
        scheme, _, token = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise AuthenticationError("Firebase sign-in is required.")
        try:
            decoded = self._verify(token)
        except Exception as exc:
            raise AuthenticationError(
                "The Firebase session is invalid or expired."
            ) from exc
        uid = str(decoded.get("uid") or decoded.get("sub") or "").strip()
        if not uid:
            raise AuthenticationError("Firebase token has no user identity.")
        return Principal(
            uid=uid,
            email=decoded.get("email"),
            name=decoded.get("name"),
        )

    def public_config(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "required": True,
            "firebase": {
                "apiKey": self.api_key,
                "authDomain": self.auth_domain,
                "projectId": self.project_id,
                "appId": self.app_id,
            },
            "warning": None,
        }


def authenticator_from_env() -> StudioAuthenticator:
    mode = os.getenv("AUTONEXUS_AUTH_MODE", "local").strip().lower()
    if mode == "local":
        return StudioAuthenticator()
    if mode != "firebase":
        raise ConfigurationError(
            "AUTONEXUS_AUTH_MODE must be 'local' or 'firebase'."
        )
    return FirebaseAuthenticator(
        api_key=os.getenv("AUTONEXUS_FIREBASE_API_KEY", "").strip(),
        project_id=os.getenv("AUTONEXUS_FIREBASE_PROJECT_ID", "").strip(),
        auth_domain=os.getenv("AUTONEXUS_FIREBASE_AUTH_DOMAIN"),
        app_id=os.getenv("AUTONEXUS_FIREBASE_APP_ID"),
    )
