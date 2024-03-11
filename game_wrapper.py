from game import Move, Player, Game
from copy import deepcopy

class GameWrapper(Game):
    def __init__(self, board=None, current_player=None) -> None:
        super().__init__()
        if board is not None:
            self._board = board
        if current_player is not None:
            self.current_player_idx = current_player
    
    def move(self, from_pos: tuple[int, int], slide: Move, player_id: int) -> bool:
        return super()._Game__move(from_pos, slide, player_id)
    
    
    def clone(self):
        board = deepcopy(self._board)
        return GameWrapper(board, self.current_player_idx)