from typing import List, Any, Union

class ModelWrapper:
    def __init__(self, model: Any, inputs: List[str], outputs: Union[str, List[str]]):
        self.model: Any = model
        self.inputs: List[str] = inputs
        self.outputs: Union[str, List[str]] = outputs 