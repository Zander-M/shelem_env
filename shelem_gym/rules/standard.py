# Game Rules
from shelem_gym.utils.cards import suit_of, rank_of

def legal_actions(hand, current_trick):
    """
    hand: set[int]
    current_trick: list[(player_idx, card_id)]
    returns: list[int]
    """
    if not current_trick:
        return list(hand)

    led_suit = suit_of(current_trick[0][1])
    follow = [c for c in hand if suit_of(c) == led_suit]
    return follow if follow else list(hand)


def resolve_trick_winner(current_trick, trump_suit):
    led_suit = suit_of(current_trick[0][1])

    def key(pidx_cid):
        pidx, cid = pidx_cid
        s = suit_of(cid)

        is_trump = (s == trump_suit)
        is_led = (s == led_suit)

        # Only trump or led-suit cards should compare by rank
        if is_trump or is_led:
            return (is_trump, is_led, rank_of(cid))
        else:
            # Always lose to any trump or led-suit card
            return (False, False, -1)

    return max(current_trick, key=key)[0]

