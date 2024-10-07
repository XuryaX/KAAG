import pickle
from typing import Any

class StateManager:
    @staticmethod
    def save_state(state: Any, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(state, f)

    @staticmethod
    def load_state(filename: str) -> Any:
        with open(filename, 'rb') as f:
            return pickle.load(f)