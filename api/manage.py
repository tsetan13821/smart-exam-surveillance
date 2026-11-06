# api/manage.py
"""Proctor account administration.

  python -m api.manage create-user alice
  python -m api.manage set-password alice
  echo "secret-password" | python -m api.manage create-user alice --password-stdin
"""
import argparse
import getpass
import sys

import app.utils.database as db
from api import auth_store


def _read_password(from_stdin):
    if from_stdin:
        return sys.stdin.readline().rstrip("\r\n")
    first = getpass.getpass("Password: ")
    if first != getpass.getpass("Repeat password: "):
        sys.exit("Passwords do not match.")
    return first


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m api.manage", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("create-user", "set-password"):
        p = sub.add_parser(name)
        p.add_argument("username")
        p.add_argument("--password-stdin", action="store_true", help="Read the password from standard input.")
    args = parser.parse_args(argv)

    db.init_database()
    auth_store.create_schema()
    password = _read_password(args.password_stdin)
    try:
        if args.command == "create-user":
            auth_store.create_user(args.username, password)
            print(f"Created proctor account '{args.username}'.")
        elif not auth_store.set_password(args.username, password):
            sys.exit(f"No user '{args.username}'.")
        else:
            print(f"Password changed for '{args.username}'; existing logins were signed out.")
    except ValueError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
