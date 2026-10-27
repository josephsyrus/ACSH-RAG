"""Account settings. Everything comes from environment variables (or .env) so the
same code runs unchanged on a laptop, a VPS or a container platform."""

import os
import secrets
import sys

from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else default


def _database_url() -> str:
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        if IS_HTTPS:
            # A local file on a server is lost on redeploy, taking every account with it.
            sys.exit("DATABASE_URL must be set when APP_BASE_URL is https (use your PostgreSQL connection string).")
        print("NOTE: DATABASE_URL not set; using a local SQLite file (./data/app.db). Use PostgreSQL for anything shared.")
        return "sqlite:///./data/app.db"
    # Hosts such as Neon hand out "postgresql://" (older ones "postgres://"). Both would
    # make SQLAlchemy look for the older psycopg2 driver, so name psycopg 3 explicitly.
    for scheme in ("postgres://", "postgresql://"):
        if url.startswith(scheme):
            return "postgresql+psycopg://" + url[len(scheme):]
    return url


# Public address of the site, with no trailing slash. The Google redirect URI is
# built from this rather than from request headers, so sign-in keeps working
# behind a reverse proxy or CDN.
BASE_URL = (os.getenv("APP_BASE_URL", "").strip() or "http://localhost:8000").rstrip("/")
IS_HTTPS = BASE_URL.startswith("https://")

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "").strip()
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "").strip()
GOOGLE_CONFIGURED = bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET)

DATABASE_URL = _database_url()

DEMO_DAILY_LIMIT = _int("DEMO_DAILY_LIMIT", 3)
PRO_DAILY_LIMIT = _int("PRO_DAILY_LIMIT", 200)

SESSION_MAX_AGE = _int("SESSION_MAX_DAYS", 14) * 24 * 3600


def _session_secret() -> str:
    secret = os.getenv("SESSION_SECRET", "").strip()
    if secret:
        return secret
    if IS_HTTPS:
        sys.exit(
            "SESSION_SECRET must be set when APP_BASE_URL is https. Generate one with:\n"
            '  python -c "import secrets; print(secrets.token_urlsafe(48))"'
        )
    print("WARNING: SESSION_SECRET not set; using a random one. Logins reset on every restart.")
    return secrets.token_urlsafe(48)


SESSION_SECRET = _session_secret()

if not GOOGLE_CONFIGURED:
    print("WARNING: GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET not set; Google sign-in is disabled.")
