import os
import chess.pgn
from stockfish import Stockfish
import csv
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from concurrent.futures import ProcessPoolExecutor
from dotenv import load_dotenv

from simulate_ai_games import move_to_vector, simulate_ai_games

NUM_OF_HUMAN_GAMES = 20 
NUM_OF_AI_GAMES = 20

load_dotenv()
stockfish = Stockfish(path=os.getenv("STOCKFISH_PATH"), depth=18, parameters={"Threads": 2, "Minimum Thinking Time": 30})
move_data = []
pgn = open(os.getenv("PGN_PATH"))
min_elo, max_elo = 800, 2800
encoder = OneHotEncoder(sparse_output=False)
eco_codes = ["A00", "B01", "C42", "D02", "E02"]
encoder.fit(np.array(eco_codes).reshape(-1, 1))

def simulate_ai_games_wrapper(num_games):
    print('Simulating ai games...')
    simulated_games = simulate_ai_games(num_games)
    return [
        [move['move_num'], move['centipawn_loss'], move['cumulative_centipawn_loss'], move['player_type'],
         move['normalized_elo'], move['game_result'], *move['eco_encoded'], *move['move_vector']]
        for game in simulated_games for move in game
    ]

def process_pgn_games_wrapper(num_games):
    print('Processing pgn games...')
    local_move_data = []
    for i in range(num_games):
        game = chess.pgn.read_game(pgn)
        board = game.board()
        mainline_moves = list(game.mainline_moves())
        prev_centipawn = 0
        cumulative_centipawn_loss = 0
        print(game.headers['White'] + ' - ' + game.headers['Black'])

        for j, move in enumerate(mainline_moves):
            board.push(move)
            stockfish.set_fen_position(board.fen())
            evaluation = stockfish.get_evaluation()
            curr_centipawn = evaluation['value'] if evaluation['type'] == 'cp' else 0
            centipawn_loss = curr_centipawn - prev_centipawn
            cumulative_centipawn_loss += centipawn_loss
            move_vector = move_to_vector(move)

            player_elo = int(game.headers.get('WhiteElo' if j % 2 == 0 else 'BlackElo', min_elo))
            normalized_elo = (player_elo - min_elo) / (max_elo - min_elo)
            encoded_result = 1 if game.headers['Result'] == "1-0" else 0 if game.headers['Result'] == "0-1" else 0.5
            eco_code = game.headers['ECO']
            eco_encoded = encoder.transform([[eco_code]])[0] if eco_code in eco_codes else np.zeros(len(eco_codes))

            local_move_data.append([
                j+1, centipawn_loss, cumulative_centipawn_loss, 0,
                normalized_elo, encoded_result, *eco_encoded, *move_vector
            ])
            prev_centipawn = curr_centipawn
    return local_move_data

if __name__ == "__main__":
    with ProcessPoolExecutor() as executor:
        ai_future = executor.submit(simulate_ai_games_wrapper, 4)
        pgn_future = executor.submit(process_pgn_games_wrapper, 4)
        
        simulated_game_data = ai_future.result()
        processed_pgn_data = pgn_future.result()
        
        move_data = processed_pgn_data + simulated_game_data

    headers = ['move_num', 'centipawn_loss', 'cumulative_centipawn_loss', 'player_type', 
               'normalized_elo', 'game_result'] + [f'eco_{eco}' for eco in eco_codes] + [f'v{i}' for i in range(128)]

    with open('data/evaluation_data.csv', 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(headers)
        csv_writer.writerows(move_data)

    print("Data saved to 'data/evaluation_data.csv'")
