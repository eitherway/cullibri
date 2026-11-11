import argparse

import uvicorn

from . import server

parser = argparse.ArgumentParser(prog="cull", description="Pick the best of similar photos, with local models only.")
parser.add_argument("--folder", help="use a plain folder of photos instead of Immich (for testing)")
parser.add_argument("--port", type=int, default=8765)
args = parser.parse_args()

server.configure(args.folder)
print(f"Source: {server.source.label}\nOpen http://127.0.0.1:{args.port}")
uvicorn.run(server.app, host="127.0.0.1", port=args.port, log_level="warning")
