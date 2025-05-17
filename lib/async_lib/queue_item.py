from typing import List, Union, Optional, Dict, Any, Callable, Awaitable, Generator
import asyncio
import logging

# Use the globally configured logger if available, otherwise default for the module
logger: logging.Logger = logging.getLogger("logger")

class ItemFuture:
    def __init__(self, parent: Optional['ItemFuture'], event_handler: Callable[['ItemFuture', str], Awaitable[None]]):
        self.parent: Optional['ItemFuture'] = parent
        self.handler: Callable[['ItemFuture', str], Awaitable[None]] = event_handler
        self.future: asyncio.Future[Any] = asyncio.Future()
        self.data: Optional[Dict[str, Any]] = {}

    async def set_data(self, key: str, value: Any) -> None:
        if self.data is not None:
            self.data[key] = value
        await self.handler(self, key)

    async def __setitem__(self, key: str, value: Any) -> None:
        await self.set_data(key, value)

    def close_future(self, value: Any) -> None:
        self.data = None # Clear data when future is closed
        if not self.future.done():
            self.future.set_result(value)

    def set_exception(self, exception: Exception) -> None:
        self.data = None # Clear data on exception
        if not self.future.done():
            self.future.set_exception(exception)

    def __contains__(self, key: str) -> bool:
        if self.data is None:
            return False
        is_present = key in self.data
        return is_present

    def __getitem__(self, key: str) -> Any:
        if self.data is None:
            return None
        
        # Log before and after the actual .get() call
        value = self.data.get(key)
        return value

    def __await__(self) -> Generator[Any, None, Any]:
        yield from self.future.__await__()
        return self.future.result()

    @classmethod
    async def create(cls, parent: Optional['ItemFuture'], data: Dict[str, Any], event_handler: Callable[['ItemFuture', str], Awaitable[None]]) -> 'ItemFuture':
        self_ref: 'ItemFuture' = cls(parent, event_handler)
        if self_ref.data is not None:
            key: str
            for key in data:
                self_ref.data[key] = data[key]
                await self_ref.handler(self_ref, key)
        return self_ref

class QueueItem:
    def __init__(self, itemFuture: ItemFuture, input_names: List[str], output_names: Union[str, List[str]]):
        self.item_future: ItemFuture = itemFuture
        self.input_names: List[str] = input_names
        self.output_names: Union[str, List[str]] = output_names 