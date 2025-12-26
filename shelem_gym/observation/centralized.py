# Centralized card observation for critic 
import numpy as np
from shelem_gym.utils.cards import NUM_CARDS

def build_centralized_state(
    *,
    hands,
    played,
    current_trick,
    trump,
    leader,
    team_tricks,
):
    hands_mat = np.zeros((4, NUM_CARDS), np.float32)
    for i, hand in enumerate(hands):
        for c in hand:
            hands_mat[i, c] = 1.0

    trick_mat = np.zeros((4, NUM_CARDS), np.float32)
    for pidx, cid in current_trick:
        trick_mat[pidx, cid] = 1.0

    trump_vec = np.eye(4, dtype=np.float32)[trump]
    leader_vec = np.eye(4, dtype=np.float32)[leader]

    return np.concatenate([
        hands_mat.reshape(-1),
        played.astype(np.float32),
        trick_mat.reshape(-1),
        trump_vec,
        leader_vec,
        team_tricks.astype(np.float32),
    ])
