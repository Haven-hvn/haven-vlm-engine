class ServerStopException(Exception):
    message: str
    def __init__(self, message: str):
        self.message = message

class NoActiveModelsException(Exception):
    message: str
    def __init__(self, message: str):
        self.message = message