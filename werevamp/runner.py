from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Optional, Sequence, List, Iterable, Callable
from math import inf

from .game import Game, Action
from .player import BasePlayer


class INavigableGameStates(ABC):
    @abstractmethod
    def prev(self) -> Optional[Game]:
        pass

    @abstractmethod
    def cur(self) -> Game:
        pass

    @abstractmethod
    def next(self) -> Optional[Game]:
        pass

    @property
    @abstractmethod
    def cur_idx(self) -> int:
        pass

    @property
    @abstractmethod
    def num_states(self) -> int:
        pass


class GameHistory(List[Game], INavigableGameStates):
    def __init__(self, game_iterable:Iterable[Game] = ()):
        super().__init__(game_iterable)
        self._cur_idx: int = 0 if len(self) > 0 else -1
    
    @property
    def cur_idx(self) -> int:
        return self._cur_idx 

    @cur_idx.setter
    def cur_idx(self, idx) -> None:
        if idx < 0 or idx >= len(self):
            raise ValueError(f"`val` outside the range of the history: [0, {len(self)})")

    def prev(self) -> Optional[Game]:
        if self._cur_idx < 0:
            raise ValueError("No game state in the history")
        if self._cur_idx > 0:
            self._cur_idx -= 1
            return self.cur()
        else:
            return None
    
    def cur(self) -> Game:
        if self._cur_idx == -1:
            raise ValueError("No game state in the history")
        return self[self.cur_idx]

    def next(self) -> Optional[Game]:
        if self._cur_idx < len(self) - 1 and self._cur_idx > -1:
            self._cur_idx += 1
            return self.cur()
        else:
            return None
    
    @property
    def num_states(self) -> int:
        return len(self)

    def append_inc(self, game:Game) -> None:
        self._cur_idx += 1
        super().append(game)


class BaseRunner(INavigableGameStates):
    def __init__(self, initial_state: Optional[Game] = None):
        self.history = GameHistory((initial_state,)) if initial_state else GameHistory()

    @property
    def turn_count(self) -> int:
        return len(self.history)
    
    @property
    def last_state(self) -> Optional[Game]:
        return self.history[-1] if len(self.history) > 0 else None

    def winner(self) -> Optional[int]:
        return self.last_state.winner()

    @property
    def num_states(self) -> int:
        return len(self.history)

    @property
    def cur_idx(self) -> int:
        return self.history.cur_idx

    @cur_idx.setter
    def cur_idx(self, idx) -> None:
        self.history.cur_idx = idx

    def next(self) -> Optional[Game]:
        """Returns the next state in history, if non-existant try to generate it."""
        ret = self.history.next()
        if ret is None:
            ret = self.generate()
        return ret
    
    def cur(self) -> Game:
        """Returns the current state in history."""
        return self.history.cur()

    def prev(self) -> Optional[Game]:
        """Returns the previous state in history if existing, or None."""
        return self.history.prev()

    @abstractmethod
    def __call__(self) -> Optional[Game]:
        pass

    def generate(self) -> Game:
        new_state = self()
        if new_state is not None:
            self.history.append_inc(new_state)
        return new_state

    def run(self, max_turns:int=-1, break_if_winner=True) -> Game:
        """Compute all states until either one of the player wins or the turn limit is reached."""
        if max_turns < 0: max_turns = inf
        while self.turn_count < max_turns:
            new_state = self.generate()
            if break_if_winner and new_state.winner is not None:
                break
        return self.last_state

    def runit(self, max_turns:int=-1, break_if_winner=True) -> Game:
        """Generate and yield all states until either one of the player wins or the turn limit is reached."""
        if max_turns < 0: max_turns = inf
        while self.turn_count < max_turns:
            new_state = self.generate()
            yield new_state
            if break_if_winner and new_state.winner is not None:
                break
        return self.last_state


class CallableRuner(BaseRunner):
    def __init__(self, next_state_cb: Callable[[Optional[Game]], Optional[Game]], initial_state: Optional[Game] = None):
        super().__init__(initial_state)
        self.next_state_cb = next_state_cb

    def __call__(self) -> Optional[Game]:
        return self.next_state_cb(self.last_state)


class GameRunner2(BaseRunner):
    def __init__(self, initial_state: Game, player1: BasePlayer, player2: BasePlayer, player1_first: bool=True):
        self.player1 = player1
        self.player2 = player2
        self.player1_turn = player1_first
        self.history = GameHistory((initial_state,))

    def __call__(self) -> Optional[Game]:
        """If the game is not over or the turn limit is not exceeded, generate the next state, add it to the history and return it."""
        if self.winner() is not None:
            return None
        new_game = deepcopy(self.last_state)
        player = self.player1 if self.player1_turn else self.player2
        new_game.register_actions(player.faction, player.play(self.last_state))
        self.history.append_inc(new_game)
        self.player1_turn = not self.player1_turn 
        return new_game


class GameRunner(INavigableGameStates):
    def __init__(self, initial_state: Game, player1: BasePlayer, player2: BasePlayer, player1_first: bool=True, max_turns:int=-1):
        self.player1 = player1
        self.player2 = player2
        self.player1_turn = player1_first
        self.max_turns = max_turns if max_turns > 0 else inf
        self.history = GameHistory((initial_state,))

    @property
    def turn_count(self):
        return len(self.history)
    
    @property
    def last_state(self):
        return self.history[-1]

    def winner(self) -> Optional[int]:
        return self.last_state.winner()
    
    def __call__(self) -> Optional[Game]:
        """If the game is not over or the turn limit is not exceeded, generate the next state, add it to the history and return it."""
        if self.winner() is not None or self.turn_count == self.max_turns:
            return None
        new_game = deepcopy(self.last_state)
        player = self.player1 if self.player1_turn else self.player2
        new_game.register_actions(player.faction, player.play(self.last_state))
        self.history.append_inc(new_game)
        self.player1_turn = not self.player1_turn 
        return new_game

    @property
    def cur_idx(self) -> int:
        return self.history.cur_idx

    @cur_idx.setter
    def cur_idx(self, idx) -> None:
        self.history.cur_idx = idx

    def next(self) -> Optional[Game]:
        """Returns the next state in history, if non-existant try to generate it."""
        return self.history.next() or self()
    
    def cur(self) -> Game:
        """Returns the current state in history."""
        return self.history.cur()

    def prev(self) -> Optional[Game]:
        """Returns the previous state in history if existing, or None."""
        return self.history.prev()

    def run(self) -> Game:
        """Compute all states until either one of the player wins or the turn limit is reached."""
        while True:
            if self() is None:
                break
        return self.last_state


    def runit(self) -> Game:
        """Generate and yield all states until either one of the player wins or the turn limit is reached."""
        while True:
            state = self()
            if state is not None:
                yield state
            else:
                break
        return self.last_state


# class GameRunnerNoHistory():
#     def __init__(self, initial_state: Game, player1: BasePlayer, player2: BasePlayer, player1_first: bool=True, max_turns:int=-1):
#         self.player1 = player1
#         self.player2 = player2
#         self.max_turns = max_turns if max_turns > 0 else inf
#         self.player1_turn = player1_first
#         self.last_state = initial_state
#         self._turn_count = 0

#     @property
#     def turn_count(self):
#         return self._turn_count

#     def winner(self) -> Optional[int]:
#         return self.last_state.winner()

#     def __call__(self):
#         self._turn_count += 1
#         new_game = deepcopy(self.last_state)
#         player = self.player1 if self.player1_turn else self.player2
#         new_game.register_actions(player.faction, player.play(self.last_state))
#         self.last_state = new_game
#         self.player1_turn = not self.player1_turn 
#         return new_game

#     def run(self) -> Game:
#         """Compute all turns until either one of the player wins or the turn limit is reached."""
#         while self.winner() is None and self.turn_count < self.max_turns:
#             self()
#         return self.last_state


#     def runit(self) -> Game:
#         """Generate and yield all turns until either one of the player wins or the turn limit is reached."""
#         while self.winner() is None and self.turn_count < self.max_turns:
#             yield self()
#         return self.last_state


#     # def runit(self, game:Game, p1: BasePlayer, p2: BasePlayer, p1_first:bool=True, max_turns: int =1000):
#     #     nturn = 0
#     #     turn = p1_first

#     #     def next_actions():
#     #         if turn: return p1.player, p1.play(game)
#     #         else: return p2.player, p2.play(game)

#     #     while game.winner() is None and nturn < max_num_turns:
#     #         faction, actions = next_actions()
#     #         game.register_actions(faction, actions)
#     #         turn = not turn
#     #         nturn += 1
#     #         yield game
#     #     self.last_game_state = game
#     #     self.winner = game.winner()
#     #     return game


#     # def run(self, game:Game, p1, p2, p1_first:bool=True, max_num_turns=1000) -> Game:
#     #     timers = {
#     #         "choose": 0,
#     #         "register": 0
#     #     }
        
#     #     nturn = 0
#     #     turn = p1_first
#     #     def next_actions():
#     #         if turn: return p1.player, p1.play(game)
#     #         else: return p2.player, p2.play(game)

#     #     while game.winner() is None and nturn < max_num_turns:
#     #         faction, actions = next_actions()
#     #         game.register_actions(faction, actions)
#     #         turn = not turn
#     #         nturn += 1
#     #         t3 = time()
#     #     return game
