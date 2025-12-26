# Policy Observation per-agent

import numpy as np
from shelem_gym.utils.cards import NUM_CARDS

def build_local_obs(
    *,
    player_idx,
    hands,
    played,
    current_trick,
    trump,
    leader,
    turn,
    team_tricks,
    num_tricks,
):
    hand_vec = np.zeros(NUM_CARDS, np.float32)
    for c in hands[player_idx]:
        hand_vec[c] = 1.0

    played_vec = played.astype(np.float32)

    trick_mat = np.zeros((4, NUM_CARDS), np.float32)
    for pidx, cid in current_trick:
        trick_mat[pidx, cid] = 1.0

    trump_vec = np.eye(4, dtype=np.float32)[trump]
    leader_vec = np.eye(4, dtype=np.float32)[leader]
    turn_vec = np.eye(4, dtype=np.float32)[turn]

    score_vec = np.array([
        team_tricks[0] / num_tricks,
        team_tricks[1] / num_tricks,
    ], dtype=np.float32)

    return np.concatenate([
        hand_vec,
        played_vec,
        trick_mat.reshape(-1),
        trump_vec,
        leader_vec,
        turn_vec,
        score_vec,
    ])
