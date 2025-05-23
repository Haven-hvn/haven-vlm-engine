import os
from lib.server.server_manager import app, port
import lib.server.routes
import uvicorn
import signal
import asyncio
from uvicorn import Config, Server
from typing import Optional
from types import FrameType

# Get the directory of the current script
script_dir: str = os.path.dirname(os.path.abspath(__file__))
# Change the current working directory to the script's directory
os.chdir(script_dir)

if __name__ == "__main__":
    config: Config = Config(app, host="0.0.0.0", port=port)
    server: Server = Server(config)

    async def stop_servers():
        await server.shutdown()

    def signal_handler(s: int, f: Optional[FrameType]):
        asyncio.create_task(stop_servers())

    signal.signal(signal.SIGINT, signal_handler)

    server.run()