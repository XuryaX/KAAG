from typing import Dict, Any
from abc import ABC, abstractmethod

class Aptitude(ABC):
    """
    Abstract base class for aptitude functions.
    """

    @abstractmethod
    def f(self, previous_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the previous interaction state.

        Args:
            previous_state (Dict[str, Any]): The previous interaction state.

        Returns:
            Dict[str, Any]: Processed state.
        """
        pass

    @abstractmethod
    def g(self, current_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the current interaction data.

        Args:
            current_data (Dict[str, Any]): The current interaction data.

        Returns:
            Dict[str, Any]: Processed data.
        """
        pass

    @abstractmethod
    def utility(self, next_node: str, current_node: str, interaction_state: Dict[str, Any], response: str) -> float:
        """
        Calculate the utility of transitioning to the next node.

        Args:
            next_node (str): The potential next node.
            current_node (str): The current node.
            interaction_state (Dict[str, Any]): The current interaction state.
            response (str): The generated response.

        Returns:
            float: The calculated utility.
        """
        pass

class BasicAptitude(Aptitude):
    """
    A basic implementation of the Aptitude class.
    """

    def f(self, previous_state: Dict[str, Any]) -> Dict[str, Any]:
        # Implement basic processing of previous state
        return previous_state

    def g(self, current_data: Dict[str, Any]) -> Dict[str, Any]:
        # Implement basic processing of current data
        return current_data

    def utility(self, next_node: str, current_node: str, interaction_state: Dict[str, Any], response: str) -> float:
        # Implement basic utility calculation
        return 0.0