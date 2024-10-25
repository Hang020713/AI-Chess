import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, 'chess_ai')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Define paths for all saved files
MODEL_PATH = os.path.join(MODEL_DIR, 'chess_model.keras')
MEMORY_PATH = os.path.join(MODEL_DIR, 'chess_memory.pkl')
HISTORY_PATH = os.path.join(MODEL_DIR, 'game_histories.pkl')

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.mixed_precision import set_global_policy
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
import random
import pickle
import time
import psutil
from multiprocessing import cpu_count
import gc
from collections import deque

# Configure CPU usage
physical_devices = tf.config.list_physical_devices('CPU')
cpu_count = os.cpu_count()
num_cores = max(1, int(cpu_count * 0.8))  # Use 80% of available cores

# Function to set CPU affinity
def set_cpu_affinity(target_percentage=80):
    try:
        # Get the number of CPUs
        global num_cores
        num_cpus = cpu_count
        # Calculate number of CPUs to use (80% by default)
        num_cores = max(1, int(num_cpus * (target_percentage / 100)))
        
        # Configure TensorFlow threading
        tf.config.threading.set_inter_op_parallelism_threads(num_cores)
        tf.config.threading.set_intra_op_parallelism_threads(num_cores)
        
        # Increase the number of parallel calls for data processing
        tf.data.experimental.AUTOTUNE = num_cores
        
        print(f"CPU Configuration:")
        print(f"Total CPUs: {num_cpus}")
        print(f"Using CPUs: {num_cores} ({target_percentage}%)")
        
    except Exception as e:
        print(f"Warning: Could not set CPU affinity: {e}")
        print("Falling back to default CPU configuration")

def get_cpu_usage():
    """Get CPU usage for the current process."""
    try:
        return psutil.cpu_percent(interval=0.1)
    except Exception as e:
        print(f"Error measuring CPU usage: {e}")
        return 0.0

# Set CPU usage to 80%
set_cpu_affinity(80)

# Optional: Set process priority to below normal to prevent system slowdown
try:
    process = psutil.Process()
    if os.name == 'nt':  # Windows
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    else:  # Unix-based
        process.nice(10)  # Higher nice value = lower priority
except Exception as e:
    print(f"Warning: Could not set process priority: {e}")

# Memory growth configuration
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print(f"Found {len(gpu_devices)} GPU(s)")

def cleanup_memory():
    gc.collect()
    tf.keras.backend.clear_session()

class ChessGame:
    EMPTY = 0
    # White pieces (positive)
    W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING = 1, 2, 3, 4, 5, 6
    # Black pieces (negative)
    B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING = -1, -2, -3, -4, -5, -6
    
    def __init__(self):
        self.board = self.init_board()
        self.move_history = []
        self.current_player = True  # True for White, False for Black
        self.done = False
        self.winner = None
        self.piece_count = self._count_pieces()
        
    def _count_pieces(self):
        pieces = defaultdict(int)
        for row in self.board:
            for piece in row:
                if piece != 0:
                    pieces[piece] += 1
        return pieces
        
    def init_board(self):
        board = np.zeros((8, 8), dtype=int)
        # Set up white pieces
        board[1, :] = self.W_PAWN
        board[0, [0, 7]] = self.W_ROOK
        board[0, [1, 6]] = self.W_KNIGHT
        board[0, [2, 5]] = self.W_BISHOP
        board[0, 3] = self.W_QUEEN
        board[0, 4] = self.W_KING
        
        # Set up black pieces
        board[6, :] = self.B_PAWN
        board[7, [0, 7]] = self.B_ROOK
        board[7, [1, 6]] = self.B_KNIGHT
        board[7, [2, 5]] = self.B_BISHOP
        board[7, 3] = self.B_QUEEN
        board[7, 4] = self.B_KING
        
        return board

    def get_state(self):
        state = np.zeros((8, 8, 12))  # 12 channels for 6 piece types × 2 colors
        piece_to_channel = {
            self.W_PAWN: 0, self.W_KNIGHT: 1, self.W_BISHOP: 2,
            self.W_ROOK: 3, self.W_QUEEN: 4, self.W_KING: 5,
            self.B_PAWN: 6, self.B_KNIGHT: 7, self.B_BISHOP: 8,
            self.B_ROOK: 9, self.B_QUEEN: 10, self.B_KING: 11
        }
        
        for i in range(8):
            for j in range(8):
                piece = self.board[i, j]
                if piece != 0:
                    channel = piece_to_channel[piece]
                    state[i, j, channel] = 1
                    
        return state.flatten()

    def is_valid_move(self, move):
        """Validate chess moves according to piece rules."""
        if not move:
            return False
            
        from_row, from_col, to_row, to_col = move
        
        # Basic boundary checks
        if not all(0 <= x < 8 for x in [from_row, from_col, to_row, to_col]):
            return False
            
        # Source and destination are different
        if (from_row, from_col) == (to_row, to_col):
            return False
            
        # Source square must contain a piece
        piece = self.board[from_row, from_col]
        if piece == 0:
            return False
            
        # Check if moving the correct color
        if self.current_player and piece < 0:
            return False
        if not self.current_player and piece > 0:
            return False
            
        # Check if destination contains same color piece
        dest_piece = self.board[to_row, to_col]
        if (piece > 0 and dest_piece > 0) or (piece < 0 and dest_piece < 0):
            return False
            
        # Get piece type and movement direction
        piece_type = abs(piece)
        direction = 1 if piece > 0 else -1
        
        # Calculate move deltas
        row_diff = to_row - from_row
        col_diff = to_col - from_col
        
        # Implement piece-specific rules
        if piece_type == abs(self.W_PAWN):  # Pawn
            # Forward movement (no capture)
            if col_diff == 0:
                if row_diff == direction:  # One square forward
                    return dest_piece == 0
                if ((piece > 0 and from_row == 1) or  # Initial two-square move
                    (piece < 0 and from_row == 6)) and row_diff == 2 * direction:
                    middle_row = from_row + direction
                    return (dest_piece == 0 and 
                        self.board[middle_row, from_col] == 0)
            # Capture diagonally
            elif abs(col_diff) == 1 and row_diff == direction:
                return dest_piece != 0 and dest_piece * piece < 0
            return False
            
        elif piece_type == abs(self.W_KNIGHT):  # Knight
            return ((abs(row_diff) == 2 and abs(col_diff) == 1) or
                    (abs(row_diff) == 1 and abs(col_diff) == 2))
                    
        elif piece_type == abs(self.W_BISHOP):  # Bishop
            if abs(row_diff) != abs(col_diff):
                return False
            return self._is_path_clear_diagonal(from_row, from_col, to_row, to_col)
            
        elif piece_type == abs(self.W_ROOK):  # Rook
            if (row_diff != 0 and col_diff != 0):
                return False
            return self._is_path_clear_straight(from_row, from_col, to_row, to_col)
            
        elif piece_type == abs(self.W_QUEEN):  # Queen
            if row_diff == 0 or col_diff == 0:  # Straight movement
                return self._is_path_clear_straight(from_row, from_col, to_row, to_col)
            elif abs(row_diff) == abs(col_diff):  # Diagonal movement
                return self._is_path_clear_diagonal(from_row, from_col, to_row, to_col)
            return False
            
        elif piece_type == abs(self.W_KING):  # King
            return abs(row_diff) <= 1 and abs(col_diff) <= 1
            
        return False

    def _is_path_clear_diagonal(self, from_row, from_col, to_row, to_col):
        """Check if diagonal path is clear."""
        row_step = 1 if to_row > from_row else -1
        col_step = 1 if to_col > from_col else -1
        current_row = from_row + row_step
        current_col = from_col + col_step
        
        while current_row != to_row:
            if self.board[current_row, current_col] != 0:
                return False
            current_row += row_step
            current_col += col_step
        return True

    def _is_path_clear_straight(self, from_row, from_col, to_row, to_col):
        """Check if straight path is clear."""
        if from_row == to_row:  # Horizontal movement
            step = 1 if to_col > from_col else -1
            for col in range(from_col + step, to_col, step):
                if self.board[from_row, col] != 0:
                    return False
        else:  # Vertical movement
            step = 1 if to_row > from_row else -1
            for row in range(from_row + step, to_row, step):
                if self.board[row, from_col] != 0:
                    return False
        return True

    def _is_king_in_check(self, king_pos):
        """Check if the king at given position is in check."""
        king_row, king_col = king_pos
        king_piece = self.board[king_row, king_col]
        is_white_king = king_piece > 0
        
        # Check for attacks from each direction
        directions = [
            (-1,-1), (-1,0), (-1,1),  # Top-left, top, top-right
            (0,-1),          (0,1),   # Left, right
            (1,-1),  (1,0),  (1,1)    # Bottom-left, bottom, bottom-right
        ]
        
        # Check knight attacks
        knight_moves = [
            (-2,-1), (-2,1), (-1,-2), (-1,2),
            (1,-2),  (1,2),  (2,-1),  (2,1)
        ]
        
        # Check knight attacks
        for row_off, col_off in knight_moves:
            row, col = king_row + row_off, king_col + col_off
            if 0 <= row < 8 and 0 <= col < 8:
                piece = self.board[row, col]
                if piece != 0 and abs(piece) == abs(self.W_KNIGHT):
                    if (is_white_king and piece < 0) or (not is_white_king and piece > 0):
                        return True
        
        # Check other directions
        for row_dir, col_dir in directions:
            row, col = king_row + row_dir, king_col + col_dir
            distance = 1
            
            while 0 <= row < 8 and 0 <= col < 8:
                piece = self.board[row, col]
                if piece != 0:
                    if (is_white_king and piece < 0) or (not is_white_king and piece > 0):
                        piece_type = abs(piece)
                        # Queen can attack in any direction
                        if piece_type == abs(self.W_QUEEN):
                            return True
                        # Rook can attack horizontally and vertically
                        if piece_type == abs(self.W_ROOK) and (row_dir == 0 or col_dir == 0):
                            return True
                        # Bishop can attack diagonally
                        if piece_type == abs(self.W_BISHOP) and row_dir != 0 and col_dir != 0:
                            return True
                        # Pawn can attack diagonally one square
                        if piece_type == abs(self.W_PAWN) and distance == 1:
                            if (is_white_king and row_dir == 1) or (not is_white_king and row_dir == -1):
                                return True
                    break
                row += row_dir
                col += col_dir
                distance += 1
        
        return False
        
    def make_move(self, move):
        if not self.is_valid_move(move):
            return -10  # Larger penalty for invalid moves
                
        from_row, from_col, to_row, to_col = move
        moving_piece = self.board[from_row, from_col]
        captured_piece = self.board[to_row, to_col]
        
        # Store current board state before making the move
        current_board_state = self.board.copy()
        
        # Store move in history with detailed information
        move_info = {
            'player': 'White' if self.current_player else 'Black',
            'move': move,
            'piece_moved': moving_piece,
            'piece_captured': captured_piece,
            'board_state': current_board_state,
            'from_square': f"{chr(97+from_col)}{from_row+1}",
            'to_square': f"{chr(97+to_col)}{to_row+1}"
        }
        self.move_history.append(move_info)
        
        # Execute move
        self.board[to_row, to_col] = moving_piece
        self.board[from_row, from_col] = 0
        
        # Update piece count
        if captured_piece != 0:
            self.piece_count[captured_piece] -= 1
        
        # Calculate reward
        reward = self._calculate_reward(captured_piece)
        
        # Add positional rewards
        if abs(moving_piece) == self.W_PAWN:
            # Reward pawn advancement
            direction = 1 if moving_piece > 0 else -1
            reward += 0.1 * (abs(to_row - from_row) * direction)
        
        # Control center bonus
        if 2 <= to_row <= 5 and 2 <= to_col <= 5:
            reward += 0.2
        
        # Check game end conditions
        self._check_game_end()
        
        # Switch player
        self.current_player = not self.current_player
        
        return reward
        
    def _calculate_reward(self, captured_piece):
        piece_values = {
            1: 1, -1: 1,    # Pawns
            2: 3, -2: 3,    # Knights
            3: 3, -3: 3,    # Bishops
            4: 5, -4: 5,    # Rooks
            5: 9, -5: 9,    # Queens
            6: 0, -6: 0     # Kings (no capture value as game ends)
        }
        
        if captured_piece != 0:
            return piece_values[captured_piece]
        return 0
        
    def _check_game_end(self):
        """Check game ending conditions: checkmate or stalemate."""
        # First, check if king is in check
        king_piece = self.W_KING if self.current_player else self.B_KING
        # Find king's position
        king_pos = None
        for row in range(8):
            for col in range(8):
                if self.board[row, col] == king_piece:
                    king_pos = (row, col)
                    break
            if king_pos:
                break
        
        if not king_pos:  # Should not happen in a legal game
            self.done = True
            self.winner = not self.current_player
            return

        # Check if any legal moves exist
        has_legal_moves = False
        for from_row in range(8):
            for from_col in range(8):
                piece = self.board[from_row, from_col]
                # Check only current player's pieces
                if (self.current_player and piece > 0) or (not self.current_player and piece < 0):
                    for to_row in range(8):
                        for to_col in range(8):
                            move = (from_row, from_col, to_row, to_col)
                            if self.is_valid_move(move):
                                # Make move temporarily
                                temp_board = self.board.copy()
                                self.board[to_row, to_col] = piece
                                self.board[from_row, from_col] = 0
                                
                                # Check if king is still in check
                                if not self._is_king_in_check(king_pos if piece != king_piece 
                                                            else (to_row, to_col)):
                                    has_legal_moves = True
                                
                                # Restore board
                                self.board = temp_board
                                if has_legal_moves:
                                    return

        if not has_legal_moves:
            self.done = True
            if self._is_king_in_check(king_pos):
                # Checkmate
                self.winner = not self.current_player
            else:
                # Stalemate
                self.winner = None

class OptimizedChessAI:
    def __init__(self):
        # Configure memory growth and mixed precision
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
        
        # Enable mixed precision for better performance
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        
        # Initialize paths and memory
        self.model_path = MODEL_PATH
        self.memory_path = MEMORY_PATH
        self.memory = deque(maxlen=int(1e6))  # 1 million transitions
        
        # Model dimensions
        self.state_size = 8 * 8 * 12  # Chess board state
        self.action_size = 8 * 8 * 8 * 8  # All possible moves
        
        # Optimized hyperparameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Exploration decay rate
        self.learning_rate = 3e-4  # Learning rate
        self.batch_size = 8  # Batch size for training
        self.num_parallel_calls = num_cores
        
        # Load or create model
        self.model = self._build_model()
        self.target_model = self._build_model()  # Target network for stable training
        self.update_target_network()
        
        # Load existing memory if available
        self.load_memory()

    def _build_model(self):
        """Build an optimized deep neural network model."""
        # Add this at the beginning of the method
        tf.config.experimental.enable_tensor_float_32_execution(False)  # Reduce CPU usage
        
        # Reduce model complexity slightly
        input_layer = tf.keras.layers.Input(shape=(self.state_size,))
        x = tf.keras.layers.Reshape((8, 8, 12))(input_layer)
        
        # First convolutional block - reduced filters
        x1 = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(x)
        x1 = tf.keras.layers.BatchNormalization()(x1)
        x1 = tf.keras.layers.ReLU()(x1)
        
        # Second convolutional block - reduced filters
        x2 = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(x1)
        x2 = tf.keras.layers.BatchNormalization()(x2)
        x2 = tf.keras.layers.ReLU()(x2)
        
        # Flatten and dense layers - reduced units
        x = tf.keras.layers.Flatten()(x2)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Output layer
        output_layer = tf.keras.layers.Dense(self.action_size)(x)
        
        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        
        # Use a more CPU-friendly optimizer configuration
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            clipnorm=1.0,
            epsilon=1e-7,  # Reduced precision for better CPU performance
        )
        
        # Compile with float32 for better CPU performance
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae'],
            run_eagerly=False  # Ensure graph mode for better performance
        )
        
        return model

    @tf.function(reduce_retracing=True)
    def _predict_step(self, state):
        """Optimized prediction step."""
        return self.model(state, training=False)

    @tf.function(reduce_retracing=True)
    def _train_step(self, states, targets):
        """Optimized training step."""
        with tf.GradientTape() as tape:
            predictions = self.model(states, training=True)
            loss = tf.keras.losses.Huber()(targets, predictions)
        
        # Get gradients and apply
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy."""
        if training and np.random.random() < self.epsilon:
            return self._random_action()
        
        state = np.expand_dims(state, axis=0)
        q_values = self._predict_step(state)
        return self._decode_action(np.argmax(q_values[0]))

    def replay(self, batch_size):
        """Train on a batch of experiences."""
        if len(self.memory) < batch_size:
            return
        
        # Sample batch efficiently
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        minibatch = [self.memory[i] for i in indices]
        
        # Prepare batch data efficiently
        states = np.array([m[0] for m in minibatch])
        actions = np.array([m[1] for m in minibatch])
        rewards = np.array([m[2] for m in minibatch])
        next_states = np.array([m[3] for m in minibatch])
        dones = np.array([m[4] for m in minibatch])
        
        # Convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        
        # Get target Q values using target network
        next_q_values = self.target_model(next_states, training=False)
        max_next_q = tf.reduce_max(next_q_values, axis=1)
        targets = rewards + (1 - dones) * self.gamma * max_next_q
        
        # Get current Q values and update with targets
        current_q = self.model(states, training=False)
        current_q = current_q.numpy()
        for i, action in enumerate(actions):
            action_idx = self._encode_action(action)
            current_q[i][action_idx] = targets[i]
        
        # Train the model
        loss = self._train_step(states, tf.convert_to_tensor(current_q))
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss

    def update_target_network(self):
        """Update target network weights."""
        self.target_model.set_weights(self.model.get_weights())

    def _encode_action(self, action):
        """Convert chess move to network output index."""
        from_row, from_col, to_row, to_col = action
        return from_row * 512 + from_col * 64 + to_row * 8 + to_col

    def _decode_action(self, index):
        """Convert network output index to chess move."""
        from_row = index // 512
        from_col = (index % 512) // 64
        to_row = (index % 64) // 8
        to_col = index % 8
        return (from_row, from_col, to_row, to_col)

    def _random_action(self):
        """Generate random valid chess move."""
        from_row = np.random.randint(0, 8)
        from_col = np.random.randint(0, 8)
        to_row = np.random.randint(0, 8)
        to_col = np.random.randint(0, 8)
        return (from_row, from_col, to_row, to_col)

    def save_model_and_memory(self):
        """Save model and memory state."""
        try:
            self.model.save(self.model_path)
            with open(self.memory_path, 'wb') as f:
                pickle.dump(list(self.memory), f)
            print(f"Saved model and {len(self.memory)} memories")
        except Exception as e:
            print(f"Error saving model and memory: {e}")

    def load_memory(self):
        """Load saved memory if available."""
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, 'rb') as f:
                    self.memory = deque(pickle.load(f), maxlen=self.memory.maxlen)
                print(f"Loaded {len(self.memory)} memories")
            except Exception as e:
                print(f"Error loading memory: {e}")

def train_chess_ai(episodes=1000, display_freq=10, batch_size=256, training_delay=0.0001):
    """
    Train the chess AI with controlled CPU usage.
    
    Args:
        episodes: Number of training episodes
        display_freq: How often to display progress
        batch_size: Size of training batches (increased for CPU efficiency)
        training_delay: Small delay between steps (reduced for higher CPU usage)
    """
    print(f"\nInitializing Training...")
    print(f"Using {num_cores} CPU cores")
    
    # Initialize process monitoring
    process = psutil.Process()
    process.cpu_percent()  # First call to initialize CPU monitoring
    print(f"Initial memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    
    # Initialize AI and tracking variables
    ai = OptimizedChessAI()
    game_histories = []
    best_reward = float('-inf')
    running_rewards = deque(maxlen=100)
    start_time = time.time()
    
    try:
        for episode in range(episodes):
            # Add minimal delay between episodes
            time.sleep(training_delay)
            
            # Memory cleanup every 100 episodes
            if episode % 100 == 0:
                gc.collect()
                tf.keras.backend.clear_session()
            
            episode_start = time.time()
            game = ChessGame()
            state = game.get_state()
            total_reward = 0
            move_count = 0
            
            # Episode loop with batch processing
            states_batch = []
            actions_batch = []
            rewards_batch = []
            next_states_batch = []
            dones_batch = []
            
            while not game.done and move_count < 100:
                # Minimal delay between moves
                time.sleep(training_delay)
                
                # Get action and execute
                action = ai.act(state)
                reward = game.make_move(action)
                next_state = game.get_state()
                
                # Store transition in batch
                states_batch.append(state)
                actions_batch.append(action)
                rewards_batch.append(reward)
                next_states_batch.append(next_state)
                dones_batch.append(game.done)
                
                # Process batch if full
                if len(states_batch) >= batch_size:
                    # Convert to numpy arrays for efficiency
                    states_np = np.array(states_batch, dtype=np.float32)
                    next_states_np = np.array(next_states_batch, dtype=np.float32)
                    
                    # Add to memory and train
                    for i in range(len(states_batch)):
                        ai.remember(
                            states_batch[i],
                            actions_batch[i],
                            rewards_batch[i],
                            next_states_batch[i],
                            dones_batch[i]
                        )
                    
                    if len(ai.memory) >= batch_size:
                        loss = ai.replay(batch_size)
                    
                    # Clear batches
                    states_batch = []
                    actions_batch = []
                    rewards_batch = []
                    next_states_batch = []
                    dones_batch = []
                
                state = next_state
                total_reward += reward
                move_count += 1
            
            # Process any remaining transitions
            if states_batch:
                for i in range(len(states_batch)):
                    ai.remember(
                        states_batch[i],
                        actions_batch[i],
                        rewards_batch[i],
                        next_states_batch[i],
                        dones_batch[i]
                    )
                if len(ai.memory) >= batch_size:
                    loss = ai.replay(batch_size)
            
            # Update tracking metrics
            running_rewards.append(total_reward)
            avg_reward = np.mean(running_rewards)
            
            # Save if performance improved significantly
            if avg_reward > best_reward:
                best_reward = avg_reward
                ai.save_model_and_memory()
                print(f"\nNew best model saved! Avg Reward: {avg_reward:.2f}")

            # Display progress
            if episode % display_freq == 0:
                current_time = time.time()
                elapsed = current_time - start_time
                episode_time = current_time - episode_start
                memory_usage = process.memory_info().rss / 1024 / 1024
                
                # Get more accurate CPU usage
                cpu_usage = get_cpu_usage()
                
                print("\n" + "="*50)
                print(f"Episode: {episode}/{episodes} ({episode/episodes*100:.1f}%)")
                print(f"Time: {elapsed:.2f}s (Episode: {episode_time:.2f}s)")
                print(f"Moves: {move_count}")
                print(f"Reward: {total_reward:.2f}")
                print(f"Running Reward: {avg_reward:.2f}")
                print(f"Best Reward: {best_reward:.2f}")
                print(f"Epsilon: {ai.epsilon:.3f}")
                print(f"Memory: {len(ai.memory)} transitions")
                print(f"Memory Usage: {memory_usage:.1f} MB")
                print(f"CPU Usage: {cpu_usage:.1f}%")
                print(f"Batch Size: {batch_size}")
                
                if game.winner is not None:
                    print(f"Winner: {'White' if game.winner else 'Black'}")
                
                # Save game history
                game_histories.append({
                    'episode': episode,
                    'moves': game.move_history,
                    'total_moves': move_count,
                    'total_reward': total_reward,
                    'running_reward': avg_reward,
                    'winner': game.winner,
                    'epsilon': ai.epsilon,
                    'time': episode_time,
                    'cpu_usage': cpu_usage,  # Updated to use new measurement
                    'memory_usage': memory_usage,
                    'batch_size': batch_size
                })
                
                # Save history periodically
                try:
                    if not safe_save_file(HISTORY_PATH, game_histories):
                        print("Warning: Could not save game histories")
                except Exception as e:
                    print(f"Warning: Could not save game history: {e}")
                
                # Optional: Early stopping if performance is good enough
                if avg_reward > 95:  # Adjust threshold as needed
                    print("\nSolved! Running reward is high enough.")
                    break
                
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving progress...")
        ai.save_model_and_memory()
        try:
            with open(HISTORY_PATH, 'wb') as f:
                pickle.dump(game_histories, f)
            print("Progress saved successfully!")
        except Exception as e:
            print(f"Error saving progress: {e}")
    
    finally:
        # Final cleanup and statistics
        end_time = time.time()
        total_time = end_time - start_time
        final_cpu = get_cpu_usage()
        final_memory = process.memory_info().rss / 1024 / 1024
        
        print("\nTraining Summary")
        print("="*50)
        print(f"Total Episodes: {episode + 1}")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Average Episode Time: {total_time/(episode + 1):.2f}s")
        print(f"Final Running Reward: {avg_reward:.2f}")
        print(f"Best Reward Achieved: {best_reward:.2f}")
        print(f"Final Memory Usage: {final_memory:.1f} MB")
        print(f"Final CPU Usage: {final_cpu:.1f}%")
        print(f"Final Batch Size: {batch_size}")
        
        # Final memory cleanup
        gc.collect()
        tf.keras.backend.clear_session()
    
    return game_histories

def evaluate_ai(ai, num_games=5):
    """Evaluate AI performance without exploration."""
    total_reward = 0
    for _ in range(num_games):
        game = ChessGame()
        state = game.get_state()
        move_count = 0
        while not game.done and move_count < 100:
            action = ai.act(state, training=False)
            reward = game.make_move(action)
            state = game.get_state()
            total_reward += reward
            move_count += 1
    return total_reward / num_games

def display_game(game_history):
    """Display a chess game replay with board visualization."""
    def piece_to_symbol(piece):
        symbols = {
            0: '·',    # Empty square
            1: '♙',    # White pieces
            2: '♘',
            3: '♗',
            4: '♖',
            5: '♕',
            6: '♔',
            -1: '♟',   # Black pieces
            -2: '♞',
            -3: '♝',
            -4: '♜',
            -5: '♛',
            -6: '♚'
        }
        return symbols.get(piece, '?')

    def print_board(board):
        print("\n     a   b   c   d   e   f   g   h")
        print("   ╔═══╤═══╤═══╤═══╤═══╤═══╤═══╤═══╗")
        
        for i in range(7, -1, -1):  # Print from rank 8 to 1
            print(f" {i+1} ║", end='')
            for j in range(8):
                symbol = piece_to_symbol(board[i][j])
                print(f" {symbol} ", end='')
                if j < 7:
                    print("│", end='')
            print(f"║ {i+1}")
            
            if i > 0:
                print("   ╟───┼───┼───┼───┼───┼───┼───┼───╢")
        
        print("   ╚═══╧═══╧═══╧═══╧═══╧═══╧═══╧═══╝")
        print("     a   b   c   d   e   f   g   h\n")

    if not game_history.get('moves'):
        print("No moves found in game history!")
        return

    print("\nGame Replay")
    print("="*50)
    
    # Create initial board
    game = ChessGame()  # This creates a new board with initial position
    print("\nInitial Position:")
    print_board(game.board)
    
    # Replay moves
    for move_num, move_data in enumerate(game_history['moves'], 1):
        if isinstance(move_data, dict) and 'move' in move_data:
            from_row, from_col, to_row, to_col = move_data['move']
            player = "White" if move_data.get('player', 'White') == 'White' else "Black"
            
            # Convert coordinates to chess notation
            from_square = f"{chr(97+from_col)}{from_row+1}"
            to_square = f"{chr(97+to_col)}{to_row+1}"
            
            print(f"\nMove {move_num}: {player}")
            print(f"From {from_square} to {to_square}")
            
            if move_data.get('piece_captured'):
                captured = piece_to_symbol(move_data['piece_captured'])
                print(f"Captured: {captured}")
            
            if 'board_state' in move_data:
                print_board(move_data['board_state'])
            else:
                print("(Board state not available)")
            
            print("-" * 50)
            time.sleep(0.5)  # Add delay between moves
        else:
            print(f"Invalid move data at move {move_num}")
    
    # Print game result
    if 'winner' in game_history:
        if game_history['winner'] is True:
            print("\nGame Result: White wins")
        elif game_history['winner'] is False:
            print("\nGame Result: Black wins")
        else:
            print("\nGame Result: Draw")
    
    print("\nGame Statistics:")
    print(f"Total Moves: {game_history.get('total_moves', 'N/A')}")
    print(f"Total Reward: {game_history.get('total_reward', 'N/A'):.2f}")
    print(f"Time Taken: {game_history.get('time', 'N/A'):.2f} seconds")

def analyze_games(game_histories):
    total_games = len(game_histories)
    if total_games == 0:
        print("No games to analyze.")
        return

    # Basic statistics
    white_wins = sum(1 for game in game_histories if game['winner'] is True)
    black_wins = sum(1 for game in game_histories if game['winner'] is False)
    draws = sum(1 for game in game_histories if game['winner'] is None)
    
    avg_moves = sum(game['total_moves'] for game in game_histories) / total_games
    avg_reward = sum(game['total_reward'] for game in game_histories) / total_games
    
    # Move analysis
    total_captures = 0
    piece_captures = defaultdict(int)
    
    for game in game_histories:
        for move in game['moves']:
            if move['piece_captured'] != 0:
                total_captures += 1
                piece_captures[move['piece_captured']] += 1
    
    # Print analysis
    print("\n=== Game Analysis ===")
    print(f"Total Games: {total_games}")
    print(f"\nWin Statistics:")
    print(f"White Wins: {white_wins} ({white_wins/total_games*100:.1f}%)")
    print(f"Black Wins: {black_wins} ({black_wins/total_games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/total_games*100:.1f}%)")
    
    print(f"\nGame Statistics:")
    print(f"Average Moves per Game: {avg_moves:.1f}")
    print(f"Average Reward per Game: {avg_reward:.1f}")
    print(f"Total Captures: {total_captures}")
    
    if total_captures > 0:
        print("\nCapture Distribution:")
        piece_names = {
            1: "White Pawn", -1: "Black Pawn",
            2: "White Knight", -2: "Black Knight",
            3: "White Bishop", -3: "Black Bishop",
            4: "White Rook", -4: "Black Rook",
            5: "White Queen", -5: "Black Queen"
        }
        for piece, count in sorted(piece_captures.items()):
            print(f"{piece_names.get(piece, 'Unknown')}: {count} ({count/total_captures*100:.1f}%)")

def safe_save_file(filepath, data, binary=True):
    """Safely save file with error handling."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the file
        mode = 'wb' if binary else 'w'
        with open(filepath, mode) as f:
            pickle.dump(data, f) if binary else f.write(data)
        return True
    except Exception as e:
        print(f"Error saving file {filepath}: {e}")
        # Try saving in current directory as fallback
        try:
            filename = os.path.basename(filepath)
            with open(filename, mode) as f:
                pickle.dump(data, f) if binary else f.write(data)
            print(f"Saved file in current directory: {filename}")
            return True
        except Exception as e2:
            print(f"Error saving in current directory: {e2}")
            return False

def safe_load_file(filepath, binary=True):
    """Safely load file with error handling."""
    try:
        mode = 'rb' if binary else 'r'
        with open(filepath, mode) as f:
            return pickle.load(f) if binary else f.read()
    except Exception as e:
        print(f"Error loading file {filepath}: {e}")
        # Try loading from current directory as fallback
        try:
            filename = os.path.basename(filepath)
            with open(filename, mode) as f:
                return pickle.load(f) if binary else f.read()
        except Exception as e2:
            print(f"Error loading from current directory: {e2}")
            return None

def replay_saved_games():
    """Review and analyze saved games."""
    histories = safe_load_file(HISTORY_PATH)
    if histories is None:
        print("No saved games found.")
        return
        
    print(f"Loaded {len(histories)} games")
    
    while True:
        print("\nOptions:")
        print("1. View game by number")
        print("2. View game analysis")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            game_idx = input(f"Enter game number (0-{len(histories)-1}): ")
            try:
                game_idx = int(game_idx)
                if 0 <= game_idx < len(histories):
                    display_game(histories[game_idx])
                else:
                    print("Invalid game number")
            except ValueError:
                print("Please enter a valid number")
        elif choice == '2':
            analyze_games(histories)
        elif choice == '3':
            break
        else:
            print("Invalid choice")

def main():
    print("Chess AI Training and Analysis Tool")
    print("===================================")
    
    while True:
        print("\nMain Menu:")
        print("1. Train new AI models")
        print("2. Review saved games")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            try:
                episodes = int(input("Enter number of training episodes (default 1000): ") or "1000")
                display_freq = int(input("Enter display frequency (default 1): ") or "1")
                batch_size = int(input("Enter batch size (default 8): ") or "8")
                training_delay = float(input("Enter training delay (default 0.002): ") or "0.002")
                
                histories = train_chess_ai(
                    episodes=episodes,
                    display_freq=display_freq,
                    batch_size=batch_size,
                    training_delay=training_delay
                )
                print("\nTraining completed successfully!")
                
            except ValueError as e:
                print(f"\nInvalid input: {e}")
                print("Using default values...")
                histories = train_chess_ai()
                
        elif choice == '2':
            replay_saved_games()
        elif choice == '3':
            break
        else:
            print("Invalid choice")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        raise