"""Correctness-first threaded PUCT MCTS."""

from __future__ import annotations

import math
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from zero_ttt.config import SearchConfig
from zero_ttt.game.rules import ACTION_SIZE
from zero_ttt.game.state import GameState
from zero_ttt.search.protocol import LeafEvaluator, SearchResult


@dataclass(slots=True)
class SearchEdge:
    prior: float
    visits: int = 0
    value_sum: float = 0.0
    virtual_loss: int = 0
    child: "SearchNode | None" = None


@dataclass(slots=True)
class SearchNode:
    state: GameState
    expanded: bool = False
    expanding: bool = False
    network_value: float = 0.0
    edges: dict[int, SearchEdge] = field(default_factory=dict)
    expansion_error: BaseException | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)
    condition: threading.Condition = field(init=False)

    def __post_init__(self) -> None:
        self.condition = threading.Condition(self.lock)


def terminal_value(state: GameState) -> float:
    result = state.score()
    if result.winner is None:
        return 0.0
    return 1.0 if result.winner is state.to_play else -1.0


class PythonMCTS:
    def __init__(self) -> None:
        self._root: SearchNode | None = None
        self._root_model_version: int | None = None

    def _prepare_root(self, state: GameState, model_version: int) -> SearchNode:
        if self._root_model_version != model_version:
            self._root = None
            self._root_model_version = model_version
        if self._root is not None:
            if self._root.state.identity() == state.identity():
                return self._root
            with self._root.lock:
                for edge in self._root.edges.values():
                    if edge.child is not None and edge.child.state.identity() == state.identity():
                        self._root = edge.child
                        return self._root
        self._root = SearchNode(state)
        return self._root

    @staticmethod
    def _expand(
        node: SearchNode,
        evaluator: LeafEvaluator,
        model_version: int,
    ) -> tuple[float, bool]:
        if node.state.is_terminal():
            return terminal_value(node.state), True
        with node.condition:
            if node.expanded:
                return node.network_value, False
            if node.expansion_error is not None:
                raise node.expansion_error
            if node.expanding:
                while node.expanding:
                    node.condition.wait()
                if node.expansion_error is not None:
                    raise node.expansion_error
                return node.network_value, False
            node.expanding = True
        try:
            evaluation = evaluator.evaluate(node.state, model_version)
            policy = np.asarray(evaluation.policy, dtype=np.float64)
            legal = np.asarray(node.state.legal_actions(), dtype=np.bool_)
            policy = np.where(legal, np.maximum(policy, 0.0), 0.0)
            total = float(policy.sum())
            if not math.isfinite(total) or total <= 0:
                policy = legal.astype(np.float64)
                total = float(policy.sum())
            policy /= total
            edges = {
                action: SearchEdge(prior=float(policy[action]))
                for action in np.flatnonzero(legal)
            }
            with node.condition:
                node.network_value = float(evaluation.value)
                node.edges = edges
                node.expanded = True
                node.expanding = False
                node.condition.notify_all()
            return node.network_value, True
        except BaseException as error:
            with node.condition:
                node.expansion_error = error
                node.expanding = False
                node.condition.notify_all()
            raise

    @staticmethod
    def _parent_q(node: SearchNode) -> float:
        visits = sum(edge.visits for edge in node.edges.values())
        if not visits:
            return node.network_value
        return sum(edge.value_sum for edge in node.edges.values()) / visits

    @classmethod
    def _select_edge(cls, node: SearchNode, config: SearchConfig) -> tuple[int, SearchEdge]:
        parent_visits = sum(edge.visits + edge.virtual_loss for edge in node.edges.values())
        parent_q = cls._parent_q(node)
        visited_prior_mass = sum(edge.prior for edge in node.edges.values() if edge.visits > 0)
        fpu = float(np.clip(
            parent_q - config.fpu_reduction * math.sqrt(visited_prior_mass),
            -1.0,
            1.0,
        ))
        best_action = -1
        best_edge: SearchEdge | None = None
        best_score = -math.inf
        for action, edge in node.edges.items():
            effective_visits = edge.visits + edge.virtual_loss
            if effective_visits:
                q_value = (edge.value_sum - edge.virtual_loss) / effective_visits
            else:
                q_value = fpu
            exploration = (
                config.c_puct
                * edge.prior
                * math.sqrt(max(parent_visits, 1))
                / (1 + effective_visits)
            )
            score = q_value + exploration
            if score > best_score or (score == best_score and action < best_action):
                best_action = action
                best_edge = edge
                best_score = score
        if best_edge is None:
            raise RuntimeError("expanded node has no legal edges")
        best_edge.virtual_loss += config.virtual_loss
        return best_action, best_edge

    @staticmethod
    def _undo_virtual(reserved: list[tuple[SearchNode, SearchEdge]], amount: int) -> None:
        for node, edge in reserved:
            with node.lock:
                edge.virtual_loss -= amount
                if edge.virtual_loss < 0:
                    raise RuntimeError("virtual loss underflow")

    def _simulate(
        self,
        root: SearchNode,
        evaluator: LeafEvaluator,
        config: SearchConfig,
        model_version: int,
    ) -> None:
        node = root
        reserved: list[tuple[SearchNode, SearchEdge]] = []
        try:
            while True:
                if node.state.is_terminal():
                    leaf_value = terminal_value(node.state)
                    break
                _, newly_expanded = self._expand(node, evaluator, model_version)
                if newly_expanded and node is not root:
                    leaf_value = node.network_value
                    break
                with node.lock:
                    action, edge = self._select_edge(node, config)
                    reserved.append((node, edge))
                    if edge.child is None:
                        edge.child = SearchNode(node.state.play(action))
                    node = edge.child

            while reserved:
                parent, edge = reserved.pop()
                parent_value = -leaf_value
                with parent.lock:
                    edge.virtual_loss -= config.virtual_loss
                    if edge.virtual_loss < 0:
                        raise RuntimeError("virtual loss underflow")
                    edge.visits += 1
                    edge.value_sum += parent_value
                leaf_value = parent_value
        except BaseException:
            self._undo_virtual(reserved, config.virtual_loss)
            raise

    @staticmethod
    def _add_root_noise(root: SearchNode, config: SearchConfig, rng: np.random.Generator) -> None:
        with root.lock:
            actions = sorted(root.edges)
            if not actions:
                return
            alpha = config.dirichlet_total_concentration / len(actions)
            noise = rng.dirichlet(np.full(len(actions), alpha, dtype=np.float64))
            for action, sample in zip(actions, noise, strict=True):
                edge = root.edges[action]
                edge.prior = (
                    (1.0 - config.dirichlet_weight) * edge.prior
                    + config.dirichlet_weight * float(sample)
                )

    @staticmethod
    def _statistics(root: SearchNode) -> tuple[np.ndarray, float, float, int]:
        visits = np.zeros(ACTION_SIZE, dtype=np.uint32)
        with root.lock:
            for action, edge in root.edges.items():
                visits[action] = edge.visits
        total = int(visits.sum())
        if total == 0:
            return visits, 0.0, 0.0, int(np.argmax(visits))
        probabilities = visits[visits > 0].astype(np.float64) / total
        legal_count = max(len(root.edges), 2)
        entropy = float(-(probabilities * np.log(probabilities)).sum() / math.log(legal_count))
        top = np.sort(visits)[-2:]
        gap = float((top[-1] - top[-2]) / total)
        leader = int(np.argmax(visits))
        return visits, entropy, gap, leader

    @staticmethod
    def _root_value(root: SearchNode) -> float:
        with root.lock:
            visits = sum(edge.visits for edge in root.edges.values())
            if not visits:
                return root.network_value
            return sum(edge.value_sum for edge in root.edges.values()) / visits

    @staticmethod
    def _should_stop(
        stage: int,
        leaders: list[int],
        entropy: float,
        gap: float,
        config: SearchConfig,
    ) -> bool:
        if stage == 0:
            return entropy <= config.entropy_threshold and gap >= config.gap_threshold_1
        if stage == 1:
            return leaders[-1] == leaders[-2] and gap >= config.gap_threshold_2
        if stage == 2:
            return leaders[-1] == leaders[-2] and gap >= config.gap_threshold_3
        return True

    @staticmethod
    def _select_action(
        visits: np.ndarray,
        state: GameState,
        config: SearchConfig,
        rng: np.random.Generator,
        selfplay: bool,
    ) -> int:
        if visits.sum() <= 0:
            raise RuntimeError("cannot choose from an unvisited root")
        if selfplay and state.move_number < config.temperature_moves:
            probabilities = visits.astype(np.float64) / visits.sum()
            return int(rng.choice(len(visits), p=probabilities))
        return int(np.argmax(visits))

    def search(
        self,
        state: GameState,
        evaluator: LeafEvaluator,
        config: SearchConfig,
        rng: np.random.Generator,
        model_version: int,
        selfplay: bool = True,
    ) -> SearchResult:
        if state.is_terminal():
            raise ValueError("cannot search a terminal state")
        root = self._prepare_root(state, model_version)
        self._expand(root, evaluator, model_version)
        if selfplay:
            self._add_root_noise(root, config, rng)
        completed = int(sum(edge.visits for edge in root.edges.values()))
        leaders: list[int] = []
        stop_reason = "budget_4"
        entropy = 0.0
        gap = 0.0
        with ThreadPoolExecutor(max_workers=config.num_threads, thread_name_prefix="zero-ttt-mcts") as pool:
            for stage, budget in enumerate(config.budgets):
                simulations = max(0, budget - completed)
                futures = [
                    pool.submit(self._simulate, root, evaluator, config, model_version)
                    for _ in range(simulations)
                ]
                for future in futures:
                    future.result()
                completed = budget
                visits, entropy, gap, leader = self._statistics(root)
                leaders.append(leader)
                should_stop = self._should_stop(stage, leaders, entropy, gap, config)
                if should_stop:
                    stop_reason = f"budget_{stage + 1}"
                    break
        visits, entropy, gap, _ = self._statistics(root)
        action = self._select_action(visits, state, config, rng, selfplay)
        return SearchResult(
            action=action,
            visit_counts=visits,
            root_value=self._root_value(root),
            simulations=int(visits.sum()),
            stop_reason=stop_reason,
            normalized_entropy=entropy,
            top_gap=gap,
        )
