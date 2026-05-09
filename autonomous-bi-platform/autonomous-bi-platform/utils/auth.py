"""
utils/auth.py

Authentication layer using streamlit-authenticator.

Two default roles:
  - executive : sees company-wide KPIs, strategic indicators, full anomaly feed
  - manager   : sees team-level metrics, individual performance, coaching flags

Credentials are stored in auth_config.yaml (not committed to git).
A setup script generates bcrypt-hashed passwords on first run.
"""

import os
import logging
from pathlib import Path
from typing import Optional

import yaml
import bcrypt
import streamlit as st

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "auth_config.yaml"


DEFAULT_CONFIG = {
    "credentials": {
        "usernames": {
            "executive": {
                "name": "Executive User",
                "password": "",          # filled by setup_auth_config()
                "email": "exec@company.ae",
                "role": "executive",
            },
            "manager": {
                "name": "Manager User",
                "password": "",
                "email": "manager@company.ae",
                "role": "manager",
            },
        }
    },
    "cookie": {
        "expiry_days": 1,
        "key": os.getenv("APP_SECRET_KEY", "change_me_in_production"),
        "name": "autonomous_bi_auth",
    },
    "preauthorized": {"emails": []},
}

DEFAULT_PASSWORDS = {
    "executive": "exec2024!",
    "manager": "manager2024!",
}


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def setup_auth_config():
    """
    Create auth_config.yaml with hashed default passwords.
    Run once. Safe to re-run — won't overwrite if file exists.
    """
    if CONFIG_PATH.exists():
        return

    config = DEFAULT_CONFIG.copy()
    for username, plain_pw in DEFAULT_PASSWORDS.items():
        config["credentials"]["usernames"][username]["password"] = _hash_password(
            plain_pw
        )

    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info(f"Auth config created at {CONFIG_PATH}")


def load_auth_config() -> dict:
    if not CONFIG_PATH.exists():
        setup_auth_config()

    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def get_user_role(username: str) -> str:
    """Return 'executive' or 'manager' for a given username."""
    try:
        config = load_auth_config()
        user = config["credentials"]["usernames"].get(username, {})
        return user.get("role", "manager")
    except Exception:
        return "manager"


def render_login_page():
    """
    Renders the login UI inline. Returns (authenticator, name, auth_status, username).
    Call this at the top of app.py before any other rendering.
    """
    try:
        import streamlit_authenticator as stauth

        config = load_auth_config()
        authenticator = stauth.Authenticate(
            config["credentials"],
            config["cookie"]["name"],
            config["cookie"]["key"],
            config["cookie"]["expiry_days"],
            config.get("preauthorized", {}).get("emails", []),
        )

        name, auth_status, username = authenticator.login("Login", "main")
        return authenticator, name, auth_status, username

    except Exception as e:
        logger.error(f"Auth setup failed: {e}")
        # Graceful fallback — no authentication in demo mode
        st.warning("⚠️  Running in demo mode (authentication disabled). Set up auth_config.yaml for production.")
        return None, "Demo User", True, "executive"
