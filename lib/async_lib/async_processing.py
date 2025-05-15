import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable, Generator, TYPE_CHECKING

from lib.model.ai_model import AIModel
from lib.model.skip_input import Skip

if TYPE_CHECKING:
    from lib.model.model import Model # For type hinting ModelProcessor.model
    from lib.model.ai_model import AIModel # For type hinting ModelProcessor.is_ai_model check

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
    
    def __getitem__(self, key: str) -> Any:
        if self.data is None:
            # Or raise an error, or return a default, depending on desired behavior
            # when accessing data after future is closed/excepted.
            return None 
        return self.data.get(key)

    def __await__(self) -> Generator[Any, None, Any]:
        # This makes ItemFuture awaitable, yielding the result of its internal asyncio.Future
        yield from self.future.__await__()
        return self.future.result()

    @classmethod
    async def create(cls, parent: Optional['ItemFuture'], data: Dict[str, Any], event_handler: Callable[['ItemFuture', str], Awaitable[None]]) -> 'ItemFuture':
        self_ref: 'ItemFuture' = cls(parent, event_handler) # Renamed 'self' to 'self_ref' for clarity
        if self_ref.data is not None: # Should always be true here as it's just initialized
            key: str
            for key in data:
                # Directly assign to data dict before calling handler for initial setup
                self_ref.data[key] = data[key] 
                await self_ref.handler(self_ref, key)
        return self_ref

class QueueItem:
    def __init__(self, itemFuture: ItemFuture, input_names: List[str], output_names: Union[str, List[str]]):
        self.item_future: ItemFuture = itemFuture
        self.input_names: List[str] = input_names
        self.output_names: Union[str, List[str]] = output_names

class ModelProcessor():
    def __init__(self, model: 'Model'): # Use forward reference for Model
        self.model: 'Model' = model
        self.instance_count: int = model.instance_count
        if model.max_queue_size is None:
            self.queue: asyncio.Queue[QueueItem] = asyncio.Queue()
        else:
            self.queue: asyncio.Queue[QueueItem] = asyncio.Queue(maxsize=model.max_queue_size)
        self.max_batch_size: int = self.model.max_batch_size
        self.max_batch_waits: int = self.model.max_batch_waits # Can be -1 for indefinite
        self.workers_started: bool = False
        self.failed_loading: bool = False
        # isinstance check doesn't strictly need AIModel type hint here if AIModel is imported
        # but it's good for clarity. We need to import AIModel from lib.model.ai_model
        from lib.model.ai_model import AIModel # Local import for isinstance check and attribute access
        self.is_ai_model: bool = isinstance(self.model, AIModel)

    def update_values_from_child_model(self) -> None:
        self.instance_count = self.model.instance_count
        if self.model.max_queue_size is None:
            self.queue = asyncio.Queue()
        else:
            self.queue = asyncio.Queue(maxsize=self.model.max_queue_size)
        self.max_batch_size = self.model.max_batch_size
        self.max_batch_waits = self.model.max_batch_waits
        
    async def add_to_queue(self, data: QueueItem) -> None:
        await self.queue.put(data)

    async def add_items_to_queue(self, data: List[QueueItem]) -> None:
        item: QueueItem
        for item in data:
            await self.queue.put(item)

    async def complete_item(self, item: QueueItem) -> None:
        output_target: str
        if isinstance(item.output_names, list):
            for output_target in item.output_names:
                await item.item_future.set_data(output_target, Skip())
        else: # it's a string
            await item.item_future.set_data(item.output_names, Skip())


    async def batch_data_append_with_skips(self, batch_data: List[QueueItem], item: QueueItem) -> bool:
        if self.is_ai_model:
            # Assuming item.input_names[3] exists and holds skipped_categories
            # Need to ensure AIModel is correctly typed or use hasattr for model_category
            from lib.model.ai_model import AIModel # Ensure AIModel is in scope
            if isinstance(self.model, AIModel): # Check if self.model is indeed an AIModel
                skipped_categories: Optional[List[str]] = item.item_future[item.input_names[3]] if len(item.input_names) > 3 else None
                if skipped_categories is not None:
                    this_ai_categories: Optional[Union[str, List[str]]] = self.model.model_category
                    if this_ai_categories: # Check if not None
                        if isinstance(this_ai_categories, str):
                            if this_ai_categories in skipped_categories:
                                await self.complete_item(item)
                                return True
                        elif isinstance(this_ai_categories, list): # it's a list
                            this_category: str
                            if all(this_category in skipped_categories for this_category in this_ai_categories):
                                await self.complete_item(item)
                                return True
        batch_data.append(item)
        return False

    async def worker_process(self) -> None:
        while True:
            firstItem: QueueItem = await self.queue.get()
            batch_data: List[QueueItem] = []
            if (await self.batch_data_append_with_skips(batch_data, firstItem)):
                self.queue.task_done() # task_done for the skipped firstItem
                continue

            waitsSoFar: int = 0
            # max_batch_waits can be -1 for indefinite wait, or 0 for no wait beyond first item
            
            # Condition for the loop: continue if batch is not full AND
            # (we haven't exhausted waits OR we wait indefinitely)
            while len(batch_data) < self.max_batch_size and \
                  (self.max_batch_waits == -1 or waitsSoFar < self.max_batch_waits):
                if not self.queue.empty():
                    next_item: QueueItem = await self.queue.get()
                    if await self.batch_data_append_with_skips(batch_data, next_item):
                        self.queue.task_done() # task_done for the skipped next_item
                    # If not skipped, it's added to batch_data, task_done will be called later
                elif self.max_batch_waits != 0: # Only sleep if we are allowed to wait
                    waitsSoFar += 1
                    await asyncio.sleep(1) # Sleep for 1 second
                else: # No items in queue and no more waits allowed (max_batch_waits is 0)
                    break
            
            if batch_data: # Ensure batch_data is not empty
                try:
                    # worker_function_wrapper is part of the Model base class, should exist
                    await self.model.worker_function_wrapper(batch_data)
                finally:
                    # Mark all items in the processed batch (or attempted batch) as done
                    for _ in batch_data: # Iterate as many times as items were in batch_data
                        self.queue.task_done()
            else: # If firstItem was the only one and got skipped, it's already task_done.
                  # If firstItem was not skipped but batch remains empty (should not happen with current logic)
                  # ensure its task_done if it wasn't processed.
                  # However, if batch_data is empty here, it means firstItem was skipped
                  # and its task_done was called. If firstItem was not skipped and is the only one,
                  # it should be in batch_data.
                  # The case of firstItem being put in queue but not processed and batch_data being empty
                  # seems unlikely unless batch_data_append_with_skips had an issue or firstItem was skipped.
                  # If firstItem was NOT skipped and is the only one, it's in batch_data and handled above.
                  # If firstItem IS skipped, its task_done is handled.
                  # This 'else' path where batch_data is empty means firstItem was likely skipped.
                  # If it was never added to queue (e.g. error before firstItem = await self.queue.get()),
                  # that's a different issue.
                  # Let's ensure the first item fetched is always marked done if not processed in a batch.
                  # This is now handled because if firstItem is skipped, task_done is called.
                  # If firstItem is not skipped, it goes into batch_data and is handled in the finally block.
                  pass


    async def start_workers(self) -> None:
        if self.workers_started:
            if self.failed_loading:
                # Re-raise or handle specific error type
                raise Exception("Error: Model failed to load previously!") 
            return
        # else: # Not really needed, logic flows if not returned
        try:
            self.workers_started = True # Set early to prevent re-entry attempts
            await self.model.load() # load is part of the Model base class
            _: int
            for _ in range(self.instance_count):
                asyncio.create_task(self.worker_process())
            # self.workers_started = True # Already set above
        except Exception as e:
            self.failed_loading = True
            logger.error(f"Failed to start workers for model: {e}", exc_info=True)
            raise # Re-throw the exception to be caught by the caller