import torch
from alphazeroquixo import train, QuixoDataset, get_data_loader
from utilities import (
    convert_board_to_binary,
    generate_pol_labels,
    generate_val_labels,
    get_all_move,
    possible_moves_for_piece,
    move_to_index,
)
from game import Player, Move
from game_wrapper import GameWrapper as Game
import math
import numpy as np
from tqdm.auto import tqdm
from copy import deepcopy


# Determine if CUDA is available and use it for training, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class GameState:
    def __init__(self, board, player):
        self.board = board
        self.player = player


class MCTSNode:
    def __init__(self, state, move=None, move_index=None, move_prob=None, parent=None):
        # Initialize a node for the Monte Carlo Tree Search (MCTS) algorithm
        self.state = state  # The current state of the game
        self.parent = parent  # The parent node in the MCTS tree
        self.move = move  # The move made to reach this state
        # If move index is not provided but a move is, calculate the move index
        if move_index is None and move is not None:
            move_index = move_to_index(move)
        self.move_index = move_index  # Index of the move

        self.move_prob = 0  # Probability of choosing this move (used in MCTS)
        self.children = []  # Child nodes in the MCTS tree
        self.visits = 0  # Number of times this node has been visited during MCTS
        self.value = 0  # Value of the node as determined by MCTS

        # Generate a list of untried moves from this state
        self.untried_moves = self.get_possible_moves()

    def get_possible_moves(self):
        # Get all possible moves in the game
        all_moves = get_all_move()

        # Filter out the legal moves based on the current state
        legal_moves = [
            move
            for move in all_moves
            if self.is_legal_move(move[0][0], move[0][1], move[1])
        ]
        return legal_moves

    def is_legal_move(self, x, y, direction):
        # Determine if a move is legal in the current state
        player = self.state.player  # The current player
        board = self.state.board  # The game board

        # Get possible moves for the piece at the specified position
        possible_moves = possible_moves_for_piece(x, y)
        # Check if the move is legal (either an empty spot or the player's piece)
        if board[y][x] == player or board[y][x] == -1:
            # Check if the specified move is among the possible moves
            if ((x, y), direction) in possible_moves:
                return True
        return False

    def select_child(self, C=1.41):
        # Select the best child node using the Upper Confidence Bound (UCB) formula
        best_child = None
        best_ucb = float("-inf")  # Initialize the best UCB as negative infinity

        # Iterate through each child to find the child with the highest UCB
        for child in self.children:
            V = child.value  # The average value of the child node
            P = child.move_prob  # Probability of the move as predicted by the model
            N = self.visits if self.visits > 0 else 1  # Usa almeno 1 per evitare log(0)
            n = (
                child.visits if child.visits > 0 else 1
            )  # Usa almeno 1 per evitare divisione per 0

            # Calculate the UCB value
            ucb = V + C * P * math.sqrt(math.log(N) / n)
            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child

        return best_child  # Return the best child

    def expand(self, policies):
        while len(self.untried_moves) > 0:
            move = self.untried_moves.pop()
            # Check if the move is legal and, if so, add a new child
            if self.is_legal_move(move[0][0], move[0][1], move[1]):
                new_state = self.apply_move(move)  # Apply the move to get a new state
                index = move_to_index(move)
                new_child = MCTSNode(
                    state=new_state, move=move, move_prob=policies[index], parent=self
                )
                self.children.append(
                    new_child
                )  # Add the new child to the children list

    def apply_move(self, move):
        # Apply a move to the game and return the resulting game state
        game = Game(
            deepcopy(self.state.board), self.state.player
        )  # Create a new game instance
        game.move(move[0], move[1], self.state.player)  # Apply the move
        # Update the current player
        game.current_player_idx += 1
        game.current_player_idx %= 2
        # Get the new board and player
        board = game.get_board()
        player = game.get_current_player()
        return GameState(board, player)  # Return the new game state

    def update_value(self, value):
        # Update the value of the node based on the given value
        self.visits += 1  # Increment the visit count
        self.value += value  # Update the value

    def update_move_prob(self, move_prob):
        # Update the probability of choosing this move
        self.move_prob = move_prob

    def is_leaf(self):
        # Check if the node is a leaf node (no children)
        return len(self.children) == 0

    def is_terminal(self):
        game = Game(
            deepcopy(self.state.board), self.state.player
        )  # Create a new game instance
        return game.check_winner() >= 0

    def print(self):
        # Print the node's information
        print(f"Move: {self.move}")
        print(f"Move Index: {self.move_index}")
        print(f"Move Probability: {self.move_prob}")
        print(f"Visits: {self.visits}")
        print(f"Value: {self.value}")
        print(f"Untried Moves: {self.untried_moves}")
        print(f"Children: {self.children}")
        print(f"Is Leaf: {self.is_leaf()}")
        print(f"Is Terminal: {self.is_terminal()}")


class MonteCarloTreeSearch:
    def __init__(self, neural_network):
        # Initialize the Monte Carlo Tree Search with a neural network
        self.neural_network = neural_network  # Neural network for evaluating game states
        self.neural_network.to(device)
        self.dataset = QuixoDataset()  # Dataset for storing game states and labels

    def get_model(self):
        # Return the neural network model
        return self.neural_network

    def get_dataset(self):
        # Return the dataset
        return self.dataset

    def train(
        self,
        n_games,
        epochs,
        simulation_per_game,
        epochs_per_game=2,
        max_moves_per_game=100,
        max_moves_for_dataset=10_000,
    ):
        # Train the neural network over a specified number of epochs and games per epoch
        games = []  # List to store game states
        policies = []  # List to store policy labels
        values = []  # List to store value labels
        # Simulate games and collect data
        for i in range(n_games):
            print(f"Game {i+1}")
            # Initialize a new game board and set the starting player
            board = np.ones((5, 5), dtype=np.uint8) * -1
            player = 0
            root = MCTSNode(GameState(board, player))  # Create the root node for MCTS

            # Simulate a game using MCTS and collect the results
            (
                result,
                game_states,
                game_moves,
                game_state_with_player,
            ) = self.simulate_game(root, simulation_per_game, max_moves_per_game)

            games.extend(game_states)  # Add the game states to the dataset

            # Generate and add policy and value labels for the game states
            policy = generate_pol_labels(game_moves)
            value = generate_val_labels(game_state_with_player, result)
            policies.extend(policy)
            values.extend(value)

            # Add the collected game data to the dataset
            self.dataset.add_data_and_keep_fixed(
                games, values, policies, fixed_num=max_moves_for_dataset
            )

            # Create a DataLoader and train the neural network
            batch_size = 64 if len(games) > 64 else len(games)
            train_loader = get_data_loader(
                self.dataset, batch_size
            )  # Create a DataLoader
            train(self.neural_network, train_loader, epochs=epochs_per_game)  # Train the model
        train(self.neural_network, train_loader, epochs=epochs)  # Train the model

    def simulate_game(self, node, num_simulations=500, max_moves=100):
        # Simulate a game starting from a given node
        board = node.state.board
        player = node.state.player
        game = Game(board, player)  # Initialize the game state
        game_states = []  # Store the game states
        game_moves = []  # Store the moves made in the game
        game_state_with_player = []  # Store game states along with the current player

        progress_bar = tqdm(total=max_moves, desc="Game")

        # Continue the game until there is a winner
        while game.check_winner() < 0 and len(game_moves) < max_moves:
            # Run multiple MCTS simulations for each move
            for _ in range(num_simulations):
                self.run_mcts_simulation(node)

            # After simulations, choose the best move based on the MCTS results
            best_node = self.choose_best_node(node)

            # Determine the best move and apply it to the game
            best_move = best_node.move

            progress_bar.update(1)
            progress_bar.set_postfix(
                {
                    "Value": best_node.value,
                    "Player": game.get_current_player(),
                    "Move": best_move,
                }
            )

            game_moves.append(best_move)
            game.move(best_move[0], best_move[1], game.get_current_player())
            # Update the current player
            game.current_player_idx += 1
            game.current_player_idx %= 2

            # Prepare the next node for the next round of simulations
            node = best_node

            # Convert the board to a binary representation for the neural network
            binary_board = convert_board_to_binary(
                game.get_board(), game.get_current_player()
            )
            game_states.append(binary_board)
            game_state_with_player.append((binary_board, game.get_current_player()))

        progress_bar.close()

        # Return the result of the game and the states, moves, and state-player pairs
        return game.check_winner(), game_states, game_moves, game_state_with_player

    def make_move(self, node, num_simulations=500):
        # Simulate a game starting from a given node
        for _ in range(num_simulations):
            self.run_mcts_simulation(node)
        # After simulations, choose the best move based on the MCTS results
        best_node = self.choose_best_node(node, C = 0.71)

        # Determine the best move and apply it to the game
        best_move = best_node.move
        return best_move
        
    
    def choose_best_node(self, node, C=1.41):
        return node.select_child(C)

    def run_mcts_simulation(self, node):
        path = []
        current_node = node

        # Step 1: Selezione
        while not current_node.is_leaf():
            current_node = current_node.select_child()
            path.append(current_node)

        # Step 2: Espansione
        if not current_node.is_terminal():  # Verifica che il gioco non sia finito
            policies = self.get_policies(current_node)
            current_node.expand(policies)
            new_child = current_node.select_child()  # Seleziona uno dei nuovi figli
            path.append(new_child)
            current_node = new_child

        # Step 3: Simulazione/Rollout
        # Se usi una rete neurale per la valutazione, ottieni il valore dal modello
        # Altrimenti, esegui un rollout casuale o basato su euristica
        game_outcome = self.rollout(current_node)

        # Step 4: Backpropagation
        self.backpropagate_value(path, game_outcome)

    def get_policies(self, node):
        board = node.state.board
        player = node.state.player
        binary_board = convert_board_to_binary(board, player)
        # Convert the board to a tensor for the neural network
        board_tensor = torch.tensor(binary_board).float().unsqueeze(0).to(device)

        # Predict the policy and value for the current state without updating gradients
        with torch.no_grad():
            self.neural_network.eval()
            policy, _ = self.neural_network(board_tensor)
        return policy.squeeze(0).exp().cpu().numpy()

    def rollout(self, node):
        board = node.state.board
        player = node.state.player
        binary_board = convert_board_to_binary(board, player)

        board_tensor = torch.tensor(binary_board).float().unsqueeze(0).to(device)

        with torch.no_grad():
            self.neural_network.eval()
            _, val = self.neural_network(board_tensor)
        return val.item()

    def backpropagate_value(self, path, val):
        for node in reversed(path):
            node.update_value(val)
            val = -val

    def update_children(self, node, policy):
        # Update the move probabilities for the children of a node
        for child in node.children:
            # Update the move probability of the child based on the policy prediction
            child.update_move_prob(policy[child.move_index])


class AlphaZeroPlayer(Player):
    def __init__(self, mtcs, simulation = 500) -> None:
        super().__init__()
        self.mtcs = mtcs
        self.simulation = simulation

    def make_move(self, game: "Game") -> tuple[tuple[int, int], Move]:
        board = game.get_board()
        player = game.get_current_player()
         
        node = MCTSNode(GameState(board, player))
        best_move = self.mtcs.make_move(node, num_simulations=self.simulation)
        
        return best_move
