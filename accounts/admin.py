"""Manage accounts from the command line, until payments can do it automatically.

    python -m accounts.admin list
    python -m accounts.admin set-plan someone@example.com pro
    python -m accounts.admin disable someone@example.com
    python -m accounts.admin enable someone@example.com
"""

import argparse
import sys

from .db import PLANS, SessionLocal, User, init_db


def _find(db, email: str) -> User:
    user = db.query(User).filter(User.email == email.lower()).one_or_none()
    if user is None:
        sys.exit(f"No user with email {email}. They need to sign in once first.")
    return user


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m accounts.admin")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    p = sub.add_parser("set-plan")
    p.add_argument("email")
    p.add_argument("plan", choices=PLANS)
    for name in ("disable", "enable"):
        sub.add_parser(name).add_argument("email")
    args = parser.parse_args()

    init_db()
    with SessionLocal() as db:
        if args.cmd == "list":
            for u in db.query(User).order_by(User.id):
                flag = "  DISABLED" if u.is_disabled else ""
                print(f"{u.id:>4}  {u.plan:<5}  {u.email}{flag}")
            return
        user = _find(db, args.email)
        if args.cmd == "set-plan":
            user.plan = args.plan
        else:
            user.is_disabled = args.cmd == "disable"
        db.commit()
        print(f"{user.email}: plan={user.plan}, disabled={user.is_disabled}")


if __name__ == "__main__":
    main()
