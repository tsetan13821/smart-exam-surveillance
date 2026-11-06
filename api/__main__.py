# api/__main__.py
"""python -m api [--host 127.0.0.1] [--port 8000]"""
import argparse
import os

import uvicorn


def main():
    parser = argparse.ArgumentParser(prog="python -m api")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Interface to listen on (default: this computer only).")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    os.environ["SMARTSQUAD_PORT"] = str(args.port)      # pages served from this port may call the API
    # One worker: the camera and the running session belong to a single process.
    uvicorn.run("api.main:create_app", factory=True, host=args.host, port=args.port, workers=1,
                timeout_graceful_shutdown=3, log_level="info")


if __name__ == "__main__":
    main()
