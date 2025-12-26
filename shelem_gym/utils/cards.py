# Cards Utils

SUITS = 4
RANKS = 13
NUM_CARDS = 52

def card_id(suit: int, rank: int) -> int:
    return suit * 13 + rank

def suit_of(cid: int) -> int:
    return cid // 13

def rank_of(cid: int) -> int:
    return cid % 13
