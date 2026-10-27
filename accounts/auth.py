"""Google sign-in, cookie sessions, and the dependencies that protect endpoints."""

from typing import Optional

from authlib.integrations.starlette_client import OAuth, OAuthError
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from starlette.middleware.sessions import SessionMiddleware

from . import config
from .db import User, get_db, init_db, upsert_google_user, usage_snapshot

oauth = OAuth()
if config.GOOGLE_CONFIGURED:
    oauth.register(
        name="google",
        client_id=config.GOOGLE_CLIENT_ID,
        client_secret=config.GOOGLE_CLIENT_SECRET,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )

router = APIRouter()


def current_user(request: Request, db: Session = Depends(get_db)) -> Optional[User]:
    uid = request.session.get("uid")
    return db.get(User, uid) if uid else None


def require_user(user: Optional[User] = Depends(current_user)) -> User:
    if user is None:
        raise HTTPException(status_code=401, detail="Please sign in to continue.")
    if user.is_disabled:
        raise HTTPException(status_code=403, detail="This account has been disabled.")
    return user


def _back(error: str) -> RedirectResponse:
    return RedirectResponse(f"/?auth_error={error}", status_code=303)


@router.get("/auth/login")
async def login(request: Request):
    if not config.GOOGLE_CONFIGURED:
        return _back("not_configured")
    return await oauth.google.authorize_redirect(request, f"{config.BASE_URL}/auth/callback")


@router.get("/auth/callback")
async def callback(request: Request, db: Session = Depends(get_db)):
    if not config.GOOGLE_CONFIGURED:
        return _back("not_configured")
    try:
        token = await oauth.google.authorize_access_token(request)
    except OAuthError:
        # Covers the user pressing "Cancel" on Google's screen as well as a lost
        # or mismatched state cookie.
        return _back("sign_in_failed")

    info = token.get("userinfo") or {}
    if not (info.get("sub") and info.get("email") and info.get("email_verified")):
        return _back("unverified_email")

    user = upsert_google_user(
        db, info["sub"], info["email"], info.get("name") or "", info.get("picture") or ""
    )
    if user.is_disabled:
        return _back("disabled")

    request.session.clear()
    request.session["uid"] = user.id
    return RedirectResponse("/", status_code=303)


@router.post("/auth/logout")
def logout(request: Request):
    request.session.clear()
    return {"ok": True}


@router.get("/api/me")
def me(user: Optional[User] = Depends(current_user), db: Session = Depends(get_db)):
    signed_out = {
        "authenticated": False,
        "login_configured": config.GOOGLE_CONFIGURED,
        "demo_limit": config.DEMO_DAILY_LIMIT,
        "pro_limit": config.PRO_DAILY_LIMIT,
    }
    if user is None:
        return signed_out
    if user.is_disabled:
        return {**signed_out, "disabled": True}
    return {
        "authenticated": True,
        "name": user.name,
        "email": user.email,
        "picture": user.picture,
        "usage": usage_snapshot(db, user.id),
        "demo_limit": config.DEMO_DAILY_LIMIT,
        "pro_limit": config.PRO_DAILY_LIMIT,
    }


def setup(app) -> None:
    """Create tables, enable cookie sessions and mount the auth routes."""
    init_db()
    app.add_middleware(
        SessionMiddleware,
        secret_key=config.SESSION_SECRET,
        session_cookie="acsh_session",
        max_age=config.SESSION_MAX_AGE,
        same_site="lax",
        https_only=config.IS_HTTPS,
    )
    app.include_router(router)
