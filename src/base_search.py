from abc import ABC, abstractmethod

class BaseSearch(ABC):
    """ Abstract base class for search. """

    def __init__(self):
        pass

    @abstractmethod
    def move(self, valuator, board, colour, time_limit):
        """Get the engine move."""
        pass