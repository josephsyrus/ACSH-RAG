"""User storage and the daily message allowance.

The allowance is enforced with single conditional UPDATE statements, so two
simultaneous requests can never both take the last message. That holds for
SQLite and PostgreSQL alike."""

import os
from datetime import datetime, timedelta, timezone
from typing import Iterator, Optional

from sqlalchemy import Boolean, DateTime, Integer, String, Text, create_engine, update
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from . import config

PLANS = ("demo", "pro")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_engine(url: str):
    if url.startswith("sqlite"):
        path = make_url(url).database
        if path and path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        # The app serves requests from worker threads.
        return create_engine(
            url, pool_pre_ping=True,
            connect_args={"check_same_thread": False, "timeout": 15},
        )

    connect_args = {"connect_timeout": 15}   # allows for a serverless database waking up
    if url.startswith("postgresql+psycopg:"):
        # Poolers such as Neon's PgBouncer hand each query to a different server
        # connection, which breaks server-side prepared statements.
        connect_args["prepare_threshold"] = None
    return create_engine(
        url,
        pool_pre_ping=True,   # serverless Postgres drops idle connections; test before use
        pool_recycle=300,
        pool_size=5,
        max_overflow=5,
        connect_args=connect_args,
    )


engine = _make_engine(config.DATABASE_URL)
SessionLocal = sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    google_sub: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(320), index=True)
    # Unbounded on purpose: PostgreSQL rejects a value longer than a VARCHAR(n),
    # and Google's names and picture URLs have no fixed maximum.
    name: Mapped[str] = mapped_column(Text, default="")
    picture: Mapped[str] = mapped_column(Text, default="")
    plan: Mapped[str] = mapped_column(String(16), default="demo")
    is_disabled: Mapped[bool] = mapped_column(Boolean, default=False)
    # usage_day is "" until the first message. It must not be NULL: the reset
    # query compares it with `!=`, which never matches NULL.
    usage_day: Mapped[str] = mapped_column(String(10), default="")
    usage_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    last_login_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


def init_db() -> None:
    try:
        Base.metadata.create_all(engine)
    except OperationalError as e:
        first_line = str(e.orig).strip().splitlines()[0] if e.orig else "unknown error"
        raise SystemExit(
            f"Could not connect to the database: {first_line}\n"
            "Check DATABASE_URL and that the database is reachable."
        ) from None


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def upsert_google_user(db: Session, sub: str, email: str, name: str, picture: str) -> User:
    user = db.query(User).filter(User.google_sub == sub).one_or_none()
    if user is None:
        user = User(google_sub=sub, email=email.lower(), name=name, picture=picture)
        db.add(user)
    else:
        user.email = email.lower()
        user.name = name
        user.picture = picture
        user.last_login_at = _now()
    db.commit()
    return user


def daily_limit(plan: str) -> int:
    return config.PRO_DAILY_LIMIT if plan == "pro" else config.DEMO_DAILY_LIMIT


def _today(now: datetime) -> str:
    return now.astimezone(timezone.utc).strftime("%Y-%m-%d")


def next_reset(now: datetime) -> datetime:
    today = now.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return today + timedelta(days=1)


def consume_message(db: Session, user: User, now: Optional[datetime] = None) -> bool:
    """Take one message from today's allowance. Returns False if none is left."""
    now = now or _now()
    today = _today(now)
    # A new day starts the counter from zero. If two requests race here, only the
    # first UPDATE matches a row, and the second finds the day already current.
    db.execute(
        update(User)
        .where(User.id == user.id, User.usage_day != today)
        .values(usage_day=today, usage_count=0)
        .execution_options(synchronize_session=False)
    )
    result = db.execute(
        update(User)
        .where(User.id == user.id, User.usage_day == today,
               User.usage_count < daily_limit(user.plan))
        .values(usage_count=User.usage_count + 1)
        .execution_options(synchronize_session=False)
    )
    db.commit()
    return result.rowcount == 1


def refund_message(db: Session, user: User, now: Optional[datetime] = None) -> None:
    """Give a message back when the request failed through no fault of the user."""
    today = _today(now or _now())
    db.execute(
        update(User)
        .where(User.id == user.id, User.usage_day == today, User.usage_count > 0)
        .values(usage_count=User.usage_count - 1)
        .execution_options(synchronize_session=False)
    )
    db.commit()


def usage_snapshot(db: Session, user_id: int, now: Optional[datetime] = None) -> dict:
    now = now or _now()
    user = db.get(User, user_id, populate_existing=True)
    limit = daily_limit(user.plan)
    used = user.usage_count if user.usage_day == _today(now) else 0
    return {
        "plan": user.plan,
        "limit": limit,
        "used": used,
        "remaining": max(limit - used, 0),
        "resets_at": next_reset(now).isoformat().replace("+00:00", "Z"),
    }
