import os
import chess.pgn
from stockfish import Stockfish
import numpy as np
import random
from dotenv import load_dotenv

load_dotenv()
stockfish = Stockfish(path=os.getenv("STOCKFISH_PATH"), depth=18, parameters={"Threads": 2, "Minimum Thinking Time": 30})

def normalize_elo(elo, min_elo=800, max_elo=2800):
    return (elo - min_elo) / (max_elo - min_elo)

def encode_game_result(result):
    return 1 if result == "1-0" else 0 if result == "0-1" else 0.5

# function to encode a move as a 128-dimension vector
def move_to_vector(move):
    start_square = move.from_square
    end_square = move.to_square
    start_vector = np.zeros(64)
    end_vector = np.zeros(64)
    start_vector[start_square] = 1
    end_vector[end_square] = 1
    return np.concatenate((start_vector, end_vector))

def simulate_ai_game():
    board = chess.Board()
    game_moves = []
    prev_centipawn = 0
    cumulative_centipawn_loss = 0
    move_num = 0
    black_elo = random.randint(2000, 2800)
    white_elo = random.randint(1800, 2800)

    while not board.is_game_over():
        move_num += 1
        stockfish.set_fen_position(board.fen())
        if board.turn:
            elo = white_elo
        else:
            elo = black_elo
        stockfish.set_elo_rating(elo)
        best_move = stockfish.get_best_move()
        evaluation = stockfish.get_evaluation()
        
        if evaluation['type'] == 'cp':
            curr_centipawn = evaluation['value']
        else:
            curr_centipawn = 0  # if in checkmate scenario, set to 0 for simplicity
        
        centipawn_loss = curr_centipawn - prev_centipawn
        cumulative_centipawn_loss += centipawn_loss
        move = chess.Move.from_uci(best_move)

        move_data_entry = {
            "move_num": move_num,
            "centipawn_loss": centipawn_loss,
            "cumulative_centipawn_loss": cumulative_centipawn_loss,
            "player_type": 1,
            "normalized_elo": normalize_elo(elo),
            "game_result": encode_game_result("1/2-1/2"),
            "eco_encoded": [0, 0, 1, 0, 0],
            "move_vector": move_to_vector(move).tolist()
        }
        
        game_moves.append(move_data_entry)
        board.push(move)
        prev_centipawn = curr_centipawn

    print("board result " + board.result())

    # loop through game_moves and change all 'game_result' keys to board.result()
    for move in game_moves:
        move['game_result'] = encode_game_result(board.result())

    return game_moves

def simulate_ai_games(num_games=5):
    simulated_games = []
    for i in range(num_games):
        print("Simulating ai game " + str(i))
        simulated_games.append(simulate_ai_game())
    return simulated_games
