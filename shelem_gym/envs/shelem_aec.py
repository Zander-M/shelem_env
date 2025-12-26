from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

from shelem_gym.observation.local import build_local_obs
from shelem_gym.rules.standard import legal_actions, resolve_trick_winner
from shelem_gym.rules.scoring import score_game
from shelem_gym.utils.cards import NUM_CARDS


# Matching pettingzoo conventions
def shelem_v0(**kwargs):
    env = ShelemAEC(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

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

        dummy = build_local_obs(
            player_idx=0,
            hands=[set() for _ in range(4)],
            played=np.zeros(NUM_CARDS, np.int8),
            current_trick=[],
            trump=0,
            leader=0,
            turn=0,
            team_tricks=np.zeros(2, np.int32),
            num_tricks=self.cfg.num_tricks,
        )
        obs_dim = int(dummy.shape[0])

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
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
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


    def _legal_action_mask(self, idx: int) -> np.ndarray:
        mask = np.zeros(NUM_CARDS, dtype=np.int8)
        for c in legal_actions(self.hands[idx], self.current_trick):
            mask[c] = 1
        return mask

    def observe(self, agent):
        idx = self.possible_agents.index(agent)

        obs = build_local_obs(
            player_idx=idx,
            hands=self.hands,
            played=self.played,
            current_trick=self.current_trick,
            trump=self.trump,
            leader=self.leader,
            turn=self.turn,
            team_tricks=self.team_tricks,
            num_tricks=self.cfg.num_tricks,
        )

        mask = self._legal_action_mask(idx)
        return {"obs": obs, "action_mask": mask}

    def step(self, action):
        # Reset Rewards
        for a in self.agents:
            self.rewards[a] = 0.0

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
            winner = resolve_trick_winner(self.current_trick, self.trump)
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
                self._accumulate_rewards()
                return

            # move selector to new leader
            while self._agent_selector.agent_selection != self.agents[self.turn]:
                self._agent_selector.next()
            self.agent_selection = self._agent_selector.agent_selection
        else:
            # next player in order
            self.turn = (self.turn + 1) % 4
            self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

    def _end_game(self):
        result = score_game(self.team_tricks)

        if not self.cfg.shaped_rewards:
            for a in self.agents:
                aidx = self.possible_agents.index(a)
                team = 0 if aidx in (0,2) else 1

                if result == -1:
                    self.rewards[a] = 0.0
                else:
                    self.rewards[a] = 1.0 if team == result else -1.0

        for a in self.agents:
            self.terminations[a] = True
            self.infos[a].update({
                "team_tricks": self.team_tricks.copy(),
                "winner": int(result)
            })
        