# .1 award for best script in the repo.
# doesn't really work and I was just trying to test with it but it's staying here.

import pandas as pd

# Load your CSV data
data = pd.read_csv("C:/Users/davudi/Desktop/projects/KnightWatch/data.csv")

# Group by each game and calculate cumulative sum for centipawn_loss
data['cumulative_centipawn_loss'] = data.groupby('game_id')['centipawn_loss'].cumsum()