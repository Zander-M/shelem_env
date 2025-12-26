# Game scoring logic 

def score_game(team_tricks):
    """
    team_tricks: np.ndarray shape (2,)
    returns:
        result: int  (0 = team0 wins, 1 = team1 wins, -1 = tie)
    """
    if team_tricks[0] > team_tricks[1]:
        return 0
    if team_tricks[1] > team_tricks[0]:
        return 1
    return -1
