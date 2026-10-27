"""Test environment for the accounts tests. Must run before `accounts` is imported.

Tests use a throwaway SQLite file by default. To run them against PostgreSQL, set
TEST_DATABASE_URL (in the shell or in .env) to a database you can afford to fill with
test users, such as a separate Neon branch. Never your production database.
"""

import os
import sys
import tempfile

from dotenv import dotenv_values

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

_env_file = dotenv_values(os.path.join(ROOT, ".env"))
_test_url = os.getenv("TEST_DATABASE_URL") or _env_file.get("TEST_DATABASE_URL")
_real_url = os.getenv("DATABASE_URL") or _env_file.get("DATABASE_URL")

if _test_url and _test_url == _real_url:
    sys.exit("TEST_DATABASE_URL is the same as DATABASE_URL. Use a separate database for tests.")

os.environ["DATABASE_URL"] = _test_url or (
    "sqlite:///" + os.path.join(tempfile.mkdtemp(), "test.db").replace("\\", "/")
)
os.environ["SESSION_SECRET"] = "test-secret"
os.environ["APP_BASE_URL"] = "http://localhost:8000"
os.environ["GOOGLE_CLIENT_ID"] = "test-client-id"
os.environ["GOOGLE_CLIENT_SECRET"] = "test-client-secret"
os.environ.setdefault("GEMINI_API_KEY", "test-key-never-used")   # pipeline modules build a client on import
os.environ["DEMO_DAILY_LIMIT"] = "3"
os.environ["PRO_DAILY_LIMIT"] = "5"
