from lib.model.model import Model
from importlib import import_module
from typing import Dict, Any, Optional, Callable, List
from lib.async_lib.async_processing import QueueItem
import types

class PythonModel(Model):
    def __init__(self, configValues: Dict[str, Any]):
        Model.__init__(self, configValues)
        self.function_name: Optional[str] = configValues.get("function_name")
        if self.function_name is None:
            raise ValueError("function_name is required for models of type python")
        module_name: str = "lib.model.python_functions"
        try:
            module: types.ModuleType = import_module(module_name)
            self.function: Callable[[List[QueueItem]], None] = getattr(module, self.function_name)
        except ImportError:
            raise ImportError(f"Module '{module_name}' not found.")
        except AttributeError:
            raise AttributeError(f"Function '{self.function_name}' not found in module '{module_name}'.")

    async def worker_function(self, data: List[QueueItem]) -> None:
        await self.function(data)