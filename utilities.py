import numpy as np
from game import Move


# Convert a game board to a binary representation
def convert_board_to_binary(board, current_player):
    # Create a channel representing the current player's pieces
    player_channel = np.where(board == current_player, 1, np.where(board == -1, 0, -1))
    # Create a channel representing the opponent's pieces
    opponent_channel = np.where(
        board == current_player, -1, np.where(board == -1, 0, 1)
    )

    # Stack the two channels to form the binary board representation
    binary_board = np.stack([player_channel, opponent_channel], axis=0)
    return binary_board


# Generate value labels for game states based on the game winner
def generate_val_labels(game_states, winner):
    val_labels = []
    for _, current_player in game_states:
        # Assign 0 for a draw, 1 if the current player won, and -1 otherwise
        if winner == -1:
            val_labels.append(0)
        else:
            val_labels.append(1 if current_player == winner else -1)
    return val_labels


# Generate policy labels from game moves
def generate_pol_labels(game_moves, num_possible_moves=44):
    pol_labels = []
    for move in game_moves:
        # Initialize a label vector with zeros
        label = [0] * num_possible_moves
        # Set the index corresponding to the move to 1
        move_index = move_to_index(move)
        label[move_index] = 1
        pol_labels.append(label)
    return pol_labels


# Get all possible moves for the board
def get_all_move(board_size=5):
    moves = []
    # Generate moves for corners and edges of the board
    for x in [0, board_size - 1]:
        for y in [0, board_size - 1]:
            if x == 0 and y == 0:
                moves.append(((x, y), Move.RIGHT))
                moves.append(((x, y), Move.BOTTOM))
            elif x == 0 and y == board_size - 1:
                moves.append(((x, y), Move.RIGHT))
                moves.append(((x, y), Move.TOP))
            elif x == board_size - 1 and y == 0:
                moves.append(((x, y), Move.BOTTOM))
                moves.append(((x, y), Move.LEFT))
            elif x == board_size - 1 and y == board_size - 1:
                moves.append(((x, y), Move.TOP))
                moves.append(((x, y), Move.LEFT))
    for x in [0, board_size - 1]:
        for y in range(1, board_size - 1):
            moves.append(((x, y), Move.TOP))
            moves.append(((x, y), Move.BOTTOM))
            if x == 0:
                moves.append(((x, y), Move.RIGHT))
            else:
                moves.append(((x, y), Move.LEFT))
    for x in range(1, board_size - 1):
        for y in [0, board_size - 1]:
            moves.append(((x, y), Move.RIGHT))
            moves.append(((x, y), Move.LEFT))
            if y == 0:
                moves.append(((x, y), Move.BOTTOM))
            else:
                moves.append(((x, y), Move.TOP))
    return moves


# Convert a move to its corresponding index in the moves list
def move_to_index(move, moves=None):
    if moves is None:
        moves = get_all_move()
    moves_dict = {i: move for i, move in enumerate(moves)}
    return list(moves_dict.keys())[list(moves_dict.values()).index(move)]


# Convert an index back to its corresponding move
def index_to_move(index, moves=None):
    if moves is None:
        moves = get_all_move()
    moves_dict = {i: move for i, move in enumerate(moves)}
    return moves_dict[index]


# Compute the total number of possible moves for a given board size
def compute_number_moves(size):
    # Calculate the number of moves based on corners and sides
    corners = 4
    sides = (size - 2) * 4
    moves = (corners * 2) + (sides * 3)
    return moves


def is_legal_move_sa(board, player, x, y, direction):
    # Determine if a move is legal
    # Get possible moves for the piece at the specified position
    possible_moves = possible_moves_for_piece(x, y)
    # Check if the move is legal (either an empty spot or the player's piece)
    if board[y][x] == player or board[y][x] == -1:
        # Check if the specified move is among the possible moves
        if ((x, y), direction) in possible_moves:
            return True
    return False


# Generate possible moves for a given piece on the board
def possible_moves_for_piece(x, y, board_size=5):
    moves = []
    # Generate moves based on the position of the piece (corner, edge, etc.)
    if x in [0, board_size - 1] and y in [0, board_size - 1]:
        if x == 0 and y == 0:
            moves.append(((x, y), Move.RIGHT))
            moves.append(((x, y), Move.BOTTOM))
        elif x == 0 and y == board_size - 1:
            moves.append(((x, y), Move.RIGHT))
            moves.append(((x, y), Move.TOP))
        elif x == board_size - 1 and y == 0:
            moves.append(((x, y), Move.BOTTOM))
            moves.append(((x, y), Move.LEFT))
        elif x == board_size - 1 and y == board_size - 1:
            moves.append(((x, y), Move.TOP))
            moves.append(((x, y), Move.LEFT))
    else:
        if x in [0, board_size - 1]:
            moves.append(((x, y), Move.TOP))
            moves.append(((x, y), Move.BOTTOM))
            if x == 0:
                moves.append(((x, y), Move.RIGHT))
            else:
                moves.append(((x, y), Move.LEFT))
        if y in [0, board_size - 1]:
            moves.append(((x, y), Move.RIGHT))
            moves.append(((x, y), Move.LEFT))
            if y == 0:
                moves.append(((x, y), Move.BOTTOM))
            else:
                moves.append(((x, y), Move.TOP))

    return moves


def get_possible_moves(board, player, random = False):
    # Get all possible moves in the game
    all_moves = get_all_move()

    # Filter out the legal moves based on the current state
    legal_moves = [
        move
        for move in all_moves
        if is_legal_move_sa(board, player, move[0][0], move[0][1], move[1])
    ]
    
    if random:
        np.random.shuffle(legal_moves)
        
    return legal_moves


def calculate_depth(value, max_value=44, max_depth=5):
    ratio = (max_value - value) / max_value
    depth = int(ratio * max_depth)
    depth = max(1, depth)
    return depth

def print_custom_board(board):
    n = len(board)
    symbols = {0: "❌", 1: "⭕️", -1: "⬜️"}
    winning_symbols = {0: "❎", 1: "🟢", -1: "⬜️"}

    def check_line(line):
        return all(val == line[0] and val != -1 for val in line)

    winning_rows = [check_line(board[row]) for row in range(n)]
    winning_cols = [check_line([board[row][col] for row in range(n)]) for col in range(n)]
    winning_diag1 = check_line([board[i][i] for i in range(n)])
    winning_diag2 = check_line([board[i][n - 1 - i] for i in range(n)])

    for row in range(n):
        for col in range(n):
            if winning_rows[row] or winning_cols[col] or (winning_diag1 and row == col) or (winning_diag2 and row == n - 1 - col):
                print(winning_symbols[board[row][col]], end=" ")
            else:
                print(symbols[board[row][col]], end=" ")
        print()