# AlphaZero Agent for Quixo Game
AlphaZero, a groundbreaking AI developed by DeepMind, has made significant advances in the field of artificial intelligence, notably defeating the world-renowned chess engine Stockfish. In this project, I adapt the principles of AlphaZero to develop a sophisticated agent for the Quixo game.

## Theoretical Background of AlphaZero
AlphaZero employs a unique combination of deep neural networks and a Monte Carlo Tree Search (MCTS) to master games like Chess, Go, and Shogi. Unlike traditional algorithms, it learns solely from self-play, without any prior knowledge or human intervention. This approach allows AlphaZero to discover and refine strategies, continuously improving its gameplay.

## Convolutional Neural Network Model
The core of AlphaZero implementation is a deep convolutional neural network, designed to evaluate Quixo game states and predict the best moves. The network comprises an initial convolutional layer followed by several residual blocks, which help in preserving information throughout the network. The network outputs both policy (move probabilities) and value (game outcome predictions) heads, crucial for guiding the MCTS.

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Adding the residual value
        return F.relu(out)

class QuixoNet(nn.Module):
    def __init__(self):
        super(QuixoNet, self).__init__()

        self.conv_initial = nn.Conv2d(2, 256, kernel_size=3, padding=1)
        self.bn_initial = nn.BatchNorm2d(256)

        # Creating 10 residual blocks
        self.res_blocks = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
        )

        # Defining the output layers
        self.fc_val = nn.Linear(256, 1)  # Value head
        self.fc_pol = nn.Linear(256, 44)  # Policy head

    def forward(self, x):
        x = F.relu(self.bn_initial(self.conv_initial(x)))
        x = self.res_blocks(x)
        x = F.avg_pool2d(x, x.size()[2:])  # Global average pooling
        x = x.view(x.size(0), -1)  # Flattening

        # Calculating value and policy predictions
        val = torch.tanh(self.fc_val(x))  # Value prediction
        pol = F.log_softmax(self.fc_pol(x), dim=1)  # Policy prediction

        return pol, val
```


## Concept of Self-Play
Self-play is a critical mechanism in AlphaZero's learning process, where the agent plays games against itself to improve its strategies. This method is revolutionary as it does not rely on historical data or pre-programmed knowledge. Instead, the agent develops its understanding of the game purely through the experience gained from playing against itself.

## Self-Play Process
1. **Initial Phase**: At the start, the neural network knows little about the game. The agent's decisions are mostly random, with a slight bias from the neural network's initial predictions.

2. **Data Generation**: During self-play, every move and decision made by the agent is recorded. This includes the game state, the chosen move, and the predicted outcome of the game at each step. 

3. **Policy and Value Data**: The neural network provides two key outputs for each game state - a policy (probability distribution over possible moves) and a value (an estimation of the game outcome from the current state). This data is crucial for understanding the agent's decision-making process and for training the neural network.

4. **Feedback Loop**: As games are played, the agent gathers a wealth of data. This data is then used to retrain the neural network, improving its move predictions and game outcome estimations.

5. **Iterative Improvement**: The improved network then plays more games against itself, generating better-quality data. This iterative process leads to a continuous cycle of self-improvement.

6. **Diverse Strategies**: Since the AlphaZero agent plays against itself, it encounters and adapts to a variety of strategies and board configurations. This diversity is key to developing a robust and versatile playing style.

## Benefits of Self-Play
- **Independence from Human Knowledge**: The agent is not limited by the extent of human strategies known for the game. It can discover novel strategies and tactics that human players might not have considered.
- **Adaptability and Learning**: The agent continuously adapts and learns from its own gameplay, leading to a deep understanding of the game's nuances.
- **Data Efficiency**: Self-play ensures that the data used for training is both relevant and highly efficient in improving the agent’s performance.

## Challenges in Self-Play
- **Computational Intensity**: The process of playing games, generating data, and retraining the neural network is computationally intensive and requires significant processing power.
- **Balancing Exploration and Exploitation**: Finding the right balance between exploring new strategies and exploiting known successful ones is crucial for effective learning.



## Child Selection in MCTS

In the Monte Carlo Tree Search (MCTS) used by AlphaZero implementation for Quixo, the selection of a child node is a critical step that determines the direction of the search. This process utilizes the Upper Confidence Bound (UCB) formula to weigh the trade-off between exploring new, potentially promising moves (exploration) and exploiting moves known to yield good outcomes (exploitation). The specific formula applied in the selection process is tailored to integrate both the value of the node and the policy predictions from the neural network.

### The Selection Formula
The formula used for selecting a child node in MCTS is as follows:

![UCB](image/UCB.png)

Where:
- V represents the value of the child node, which is the average result of simulations passing through that node.
- C is a constant that determines the balance between exploration and exploitation, set based on empirical testing or domain knowledge. In the provided code, C is set to 1.41, aligning with the square root of 2, a common choice in practice.
- P is the move probability assigned by the neural network, indicating the likelihood of choosing that move.
- N denotes the total number of visits to the current node (parent node of the child being evaluated).
- n is the number of visits to the child node.

### Detailed Explanation
- **Value (V) Component**: Reflects the historical success of the node. A higher value indicates a path that has led to favorable outcomes in past simulations.
- **Exploration Term**: The term C x P x sqrt(log(N)/n) balances exploration and exploitation. The multiplication by the move probability P ensures that moves deemed more promising by the neural network are preferred.
- **Adjustment for Parent and Child Visits**: The ratio of log(N) over n increases as the child node is visited less frequently compared to its siblings, promoting exploration of less-visited nodes.

```python
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
```

## MCTS Simulation
Each MCTS simulation involves four steps: selection, expansion, simulation (rollout), and backpropagation. The process starts at the root node and traverses down the tree based on UCB values until a leaf node is reached. The leaf node is then expanded, and the neural network evaluates the new nodes. The simulation step uses the neural network's value head to predict the game outcome, which is then backpropagated up the tree to update the node values.

## Training Data and Dataset
The neural network's training data, crucial for the AlphaZero agent's mastery of Quixo, is sourced from self-play—a process where the agent plays games against itself. This method ensures a steady influx of fresh game scenarios, capturing the agent's evolving strategies and decisions.

## Continuous Dataset Expansion and Pruning
As the agent accumulates experience, the dataset grows, incorporating new, strategically rich game states. To maintain high data quality and relevance, older entries—reflecting the agent's earlier, less sophisticated gameplay—are periodically pruned. This selective removal process keeps the dataset focused and efficient, containing only the most instructive game scenarios that mirror the agent's current understanding.

```python
def add_data_and_keep_fixed(self, new_game_states, new_val_labels, new_pol_labels, fixed_num):
        # Keep a fixed number of existing data randomly
        if len(self.game_states) > fixed_num:
            indices = list(range(len(self.game_states)))
            random.shuffle(indices)
            keep_indices = set(indices[:fixed_num])

            self.game_states = [self.game_states[i] for i in keep_indices]
            self.val_labels = [self.val_labels[i] for i in keep_indices]
            self.pol_labels = [self.pol_labels[i] for i in keep_indices]

        self.add_data(new_game_states, new_val_labels, new_pol_labels)
```

This dynamic approach to dataset management ensures that the neural network trains on high-quality, relevant data. By continually updating the dataset with recent games and pruning outdated information, the training material remains aligned with the agent's improving skill level, fostering an effective and focused learning environment.

## Results
I was unable to train AlphaZero due to hardware constraints. The computational power required to train a model using self-play is quite substantial. The AlphaZero paper reports that training the model on 5000 TPUs took 9 hours for 700,000 iterations. Moreover, to train the model effectively, a high number of simulations per move is necessary to get an accurate estimate of the winning probability. For these reasons, I was unable to fully train the model, but I still implemented the agent and conducted some simulations to test it. Given the limitations, I carried out a training regime that allowed me to verify the model's functionality. However, due to the limited number of epochs and simulations, the model achieved a winning rate of 50% against a random player.

It's important to remember that AlphaZero achieved superhuman results in games like chess and Go, so with adequate training, the model would be capable of beating any human player