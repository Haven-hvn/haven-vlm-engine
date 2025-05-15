import asyncio
from asyncio import Task
from contextlib import asynccontextmanager
import gc
import logging
import os
import signal
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import torch
from lib.config.config_utils import load_config
from lib.configurator.configure_active_ai import choose_active_models
from lib.logging.logger import setup_logger
from lib.pipeline.pipeline_manager import PipelineManager
from lib.server.exceptions import NoActiveModelsException, ServerStopException
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response
from starlette.types import ASGIApp
import requests
from typing import Optional, Dict, Any, List

class ServerManager:
    def __init__(self):
        config_path: str = "./config/config.yaml"
        if os.path.exists(config_path):
            config: Dict[str, Any] = load_config(config_path, default_config={})
        else:
            ServerStopException(f"Main config file does not exist: {config_path}")
            config: Dict[str, Any] = {}
        loglevel: str = config.get("loglevel", "INFO")
        setup_logger("logger", loglevel)
        self.logger: logging.Logger = logging.getLogger("logger")
        self.port: int = config.get("port", 8000)

        version_path: str = "./config/version.yaml"
        versionconfig: Dict[str, Any]
        if os.path.exists(version_path):
            versionconfig = load_config(version_path, default_config={})
        else:
            versionconfig = {}
        version: str = versionconfig.get("VERSION", "1.3.4")
        self.config: Dict[str, Any] = config
        self.pipeline_manager: PipelineManager = PipelineManager()
        self.default_image_pipeline: Optional[str] = config.get("default_image_pipeline", None)
        if self.default_image_pipeline is None:
            self.logger.error("No default image pipeline found in the configuration file.")
            raise ServerStopException("No default image pipeline found in the configuration file.")
        
        self.default_video_pipeline: Optional[str] = config.get("default_video_pipeline", None)
        if self.default_video_pipeline is None:
            self.logger.error("No default video pipeline found in the configuration file.")
            raise ServerStopException("No default video pipeline found in the configuration file.")

    async def startup(self):
        pipelines: List[Any] = self.config.get("active_pipelines", [])
        if not pipelines:
            self.logger.error("No pipelines found in the configuration file.")
            raise ServerStopException("No pipelines found in the configuration file.")
        try:
            await self.pipeline_manager.load_pipelines(pipelines)
        except NoActiveModelsException as e:
            self.logger.error(f"Error: No active AI models found in active_ai.yaml")
            try:
                choose_active_models()
            except Exception as e_inner:
                self.logger.debug(f"Error: {e_inner}")
            raise ServerStopException("No active AI models. Choose models in select_ai_models.ps1/sh and start the server again.")
        self.logger.info("Pipelines loaded successfully")
        self.background_task: Task = asyncio.create_task(check_inactivity())

    async def get_request_future(self, data: Any, pipeline_name: str) -> Any:
        return await self.pipeline_manager.get_request_future(data, pipeline_name)
    

@asynccontextmanager
async def lifespan(app: FastAPI):
    await server_manager.startup()
    yield
    pass

class OutstandingRequestsMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.outstanding_requests: int = 0
        self.last_request_timestamp: float = asyncio.get_event_loop().time()

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        self.outstanding_requests += 1
        self.last_request_timestamp = asyncio.get_event_loop().time()
        response: Response
        try:
            response = await call_next(request)
        except Exception as e:
            response = JSONResponse({"detail": str(e)}, status_code=500)
        self.outstanding_requests -= 1
        return response
    
server_manager: ServerManager = ServerManager()
port: int = server_manager.port
app: FastAPI = FastAPI(lifespan=lifespan)

outstanding_requests_middleware: OutstandingRequestsMiddleware = OutstandingRequestsMiddleware(app)
app.add_middleware(BaseHTTPMiddleware, dispatch=outstanding_requests_middleware.dispatch)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next: RequestResponseEndpoint) -> Response:
    response: Response
    try:
        response = await call_next(request)
    except Exception as e:
        response = JSONResponse({"detail": str(e)}, status_code=500)
    if response and hasattr(response, 'headers'):
        response.headers["X-Outstanding-Requests"] = str(outstanding_requests_middleware.outstanding_requests)
    return response

last_request_timestamp: float = 0.0

async def check_inactivity():
    global last_request_timestamp
    while True:
        await asyncio.sleep(300)
        middleware: OutstandingRequestsMiddleware = outstanding_requests_middleware
        if middleware.outstanding_requests == 0 and asyncio.get_event_loop().time() - middleware.last_request_timestamp > 300 and middleware.last_request_timestamp > last_request_timestamp:
            last_request_timestamp = middleware.last_request_timestamp
            print("No requests in the last 5 minutes, Clearing cached memory")
            torch.cuda.empty_cache()
            gc.collect()

origins: List[str] = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(ServerStopException)
async def server_stop_exception_handler(request: Request, exc: ServerStopException) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"message": f"Server is stopping due to error: {exc.message}"},
    )
