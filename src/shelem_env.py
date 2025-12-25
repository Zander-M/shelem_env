from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector

SUITS = 4
RANKS = 13
NUM_CARDS = 52

def card_id(suit: int, rank: int) -> int:
    return suit * 13 + rank  # rank: 0..12

def suit_of(cid: int) -> int:
    return cid // 13

@dataclass
class ShelemConfig:
    cards_per_player: int = 12
    num_tricks: int = 12
    shaped_rewards: bool = True
    seed: int | None = None

class ShelemAEC(AECEnv):
    metadata = {"name": "shelem_v0"}

    def __init__(self, cfg: ShelemConfig = ShelemConfig()):
        super().__init__()
        self.cfg = cfg
        self.possible_agents = ["p0","p1","p2","p3"]
        self.agents = []

        # Fixed action space: play any of 52 card ids
        self._action_spaces = {a: spaces.Discrete(NUM_CARDS) for a in self.possible_agents}

        # Observation: dict with vector + mask
        obs_dim = 52 + 52 + 4*52 + 4 + 4 + 4 + 2  # example design
        self._observation_spaces = {
            a: spaces.Dict({
                "obs": spaces.Box(0.0, 1.0, shape=(obs_dim,), dtype=np.float32),
                "action_mask": spaces.Box(0, 1, shape=(NUM_CARDS,), dtype=np.int8),
            })
            for a in self.possible_agents
        }

        self._rng = np.random.default_rng(cfg.seed)

        # Game state
        self.hands = None               # list of sets[int]
        self.trump = None               # int suit 0..3
        self.current_trick = None       # list[(agent_idx, card_id)]
        self.leader = None              # agent index 0..3
        self.turn = None                # agent index 0..3
        self.played = None              # np[52] bool
        self.tricks_won = None          # np[4] int
        self.team_tricks = None         # np[2] int

    # Required by PettingZoo
    def action_space(self, agent):
        return self._action_spaces[agent]

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.agents = self.possible_agents[:]
        self.rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

        deck = np.arange(NUM_CARDS)
        self._rng.shuffle(deck)

        # Deal cards_per_player each; ignore remaining (kitty) for now
        n = self.cfg.cards_per_player
        self.hands = [set(deck[i*n:(i+1)*n].tolist()) for i in range(4)]

        self.trump = int(self._rng.integers(0, 4))  # simple random trump for v0
        self.current_trick = []
        self.played = np.zeros(NUM_CARDS, dtype=np.int8)
        self.tricks_won = np.zeros(4, dtype=np.int32)
        self.team_tricks = np.zeros(2, dtype=np.int32)

        self.leader = int(self._rng.integers(0, 4))
        self.turn = self.leader

        self._agent_selector = agent_selector(self.agents)
        # Advance selector to leader
        while self._agent_selector.agent_selection != self.agents[self.turn]:
            self._agent_selector.next()

        self.agent_selection = self._agent_selector.agent_selection

    def observe(self, agent):
        idx = self.possible_agents.index(agent)

        hand_vec = np.zeros(52, dtype=np.float32)
        for c in self.hands[idx]:
            hand_vec[c] = 1.0

        played_vec = self.played.astype(np.float32)

        trick_mat = np.zeros((4,52), dtype=np.float32)
        for (pidx, cid) in self.current_trick:
            trick_mat[pidx, cid] = 1.0

        trump_vec = np.zeros(4, dtype=np.float32)
        trump_vec[self.trump] = 1.0

        leader_vec = np.zeros(4, dtype=np.float32)
        leader_vec[self.leader] = 1.0

        turn_vec = np.zeros(4, dtype=np.float32)
        turn_vec[self.turn] = 1.0

        score_vec = np.array([
            self.team_tricks[0] / self.cfg.num_tricks,
            self.team_tricks[1] / self.cfg.num_tricks,
        ], dtype=np.float32)

        obs = np.concatenate([
            hand_vec,
            played_vec,
            trick_mat.reshape(-1),
            trump_vec,
            leader_vec,
            turn_vec,
            score_vec
        ], axis=0)

        mask = self._legal_action_mask(idx)

        return {"obs": obs, "action_mask": mask}

    def _legal_action_mask(self, idx: int) -> np.ndarray:
        mask = np.zeros(NUM_CARDS, dtype=np.int8)
        hand = self.hands[idx]

        if len(self.current_trick) == 0:
            for c in hand:
                mask[c] = 1
            return mask

        led_suit = suit_of(self.current_trick[0][1])
        follow = [c for c in hand if suit_of(c) == led_suit]
        playable = follow if len(follow) > 0 else list(hand)
        for c in playable:
            mask[c] = 1
        return mask

    def step(self, action: int):
        agent = self.agent_selection
        idx = self.possible_agents.index(agent)

        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        mask = self._legal_action_mask(idx)
        if mask[action] == 0:
            raise ValueError(f"Illegal action {action} by {agent}")

        # Play card
        self.hands[idx].remove(action)
        self.played[action] = 1
        self.current_trick.append((idx, action))

        # If trick complete, resolve winner
        if len(self.current_trick) == 4:
            winner = self._resolve_trick_winner()
            self.tricks_won[winner] += 1
            team = 0 if winner in (0,2) else 1
            self.team_tricks[team] += 1

            # reward shaping per trick
            if self.cfg.shaped_rewards:
                r = 1.0 / self.cfg.num_tricks
                for a in self.agents:
                    aidx = self.possible_agents.index(a)
                    same_team = (aidx in (0,2)) == (team == 0)
                    self.rewards[a] = (r if same_team else -r)

            # new leader
            self.leader = winner
            self.turn = winner
            self.current_trick = []

            # end condition: no cards left
            if all(len(h) == 0 for h in self.hands):
                self._end_game()

            # move selector to new leader
            while self._agent_selector.agent_selection != self.agents[self.turn]:
                self._agent_selector.next()
            self.agent_selection = self._agent_selector.agent_selection
        else:
            # next player in order
            self.turn = (self.turn + 1) % 4
            self.agent_selection = self._agent_selector.next()

    def _resolve_trick_winner(self) -> int:
        led_suit = suit_of(self.current_trick[0][1])

        # Trick-taking ranking: trump beats non-trump; otherwise follow led suit
        def card_key(pidx_cid):
            pidx, cid = pidx_cid
            s = suit_of(cid)
            is_trump = (s == self.trump)
            is_led = (s == led_suit)
            rank = cid % 13
            # Higher rank should win; map rank order as you like (A high)
            # Here: 0..12; treat Ace (12) high if you encode that way.
            return (is_trump, is_led, rank)

        # winner is max by key (tuple comparison)
        return max(self.current_trick, key=card_key)[0]

    def _end_game(self):
        # terminal rewards if not shaping:
        if not self.cfg.shaped_rewards:
            if self.team_tricks[0] > self.team_tricks[1]:
                win_team = 0
            elif self.team_tricks[1] > self.team_tricks[0]:
                win_team = 1
            else:
                win_team = -1

            for a in self.agents:
                aidx = self.possible_agents.index(a)
                if win_team == -1:
                    self.rewards[a] = 0.0
                else:
                    same_team = (aidx in (0,2)) == (win_team == 0)
                    self.rewards[a] = 1.0 if same_team else -1.0

        for a in self.agents:
            self.terminations[a] = True
