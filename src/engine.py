from Evaluate import BaseValuator
from base_search import BaseSearch

class Engine():

    def __init__(self, valuator: BaseValuator , search: BaseSearch):

        self.valuator = valuator
        self.search = search
        self.number_moves = 0

        if hasattr(search, "TranspositionTable"):
            self.tt = search.TranspositionTable
        else:
            self.tt = None

        if hasattr(search, "RepetitionTable"):
            self.rt = search.RepetitionTable
        else:
            self.rt = None

    def move(self, board, colour, time_limit):

        best_move = self.search.move(self.valuator, board, colour, time_limit)
        self.number_moves += 1

        return best_move

    def time_for_move(self, time_remaining, increment):

        estimated_moves_left = max(100 - self.number_moves, 10)
        return increment + time_remaining/estimated_moves_left

    def clear_table(self):

        if self.tt is not None:
            self.tt.clear()

        if self.rt is not None:
            self.rt.clear()