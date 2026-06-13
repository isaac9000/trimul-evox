# EVOLVE-BLOCK-START
import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from skydiscover.config import DatabaseConfig
from skydiscover.search.base_database import Program, ProgramDatabase

logger = logging.getLogger(__name__)


@dataclass
class EvolvedProgram(Program):
    """Program for the evolved database."""


class EvolvedProgramDatabase(ProgramDatabase):
    """Adaptive search strategy: refine top performers, diverge when stuck."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.best_score_history: List[float] = []
        self.stagnation_count: int = 0
        self.last_best_score: float = 0.0
        self.iteration_count: int = 0

    def _get_score(self, program: EvolvedProgram) -> float:
        score = program.metrics.get("combined_score", 0.0)
        if isinstance(score, (int, float)):
            return float(score)
        return 0.0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any) -> str:
        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration if hasattr(self, 'last_iteration') else 0, iteration)

        self._update_best_program(program)

        # Track stagnation
        current_best = max((self._get_score(p) for p in self.programs.values()), default=0.0)
        if current_best - self.last_best_score > 0.01:
            self.stagnation_count = 0
            self.last_best_score = current_best
        else:
            self.stagnation_count += 1

        self.best_score_history.append(current_best)
        self.iteration_count += 1

        if self.config.db_path:
            self._save_program(program)

        logger.debug(f"Added program {program.id}, best={current_best:.4f}, stagnation={self.stagnation_count}")
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        # Sort by score descending, filter valid scores
        scored = [(self._get_score(p), p) for p in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)
        valid = [(s, p) for s, p in scored if s > 0]

        if not valid:
            parent = random.choice(candidates)
            context = random.sample(candidates, min(num_context_programs, len(candidates)))
            context = [p for p in context if p.id != parent.id][:num_context_programs]
            return {"": parent}, {"": context}

        scores_only = [s for s, _ in valid]
        programs_only = [p for _, p in valid]

        # Determine label based on stagnation
        stagnating = self.stagnation_count >= 2

        if stagnating:
            # Alternate between diverge (explore) and refine (exploit)
            if self.stagnation_count % 3 == 0:
                # Diverge from best to escape local optimum
                parent = programs_only[0]
                label = self.DIVERGE_LABEL
                # No context for targeted divergence
                return {label: parent}, {}
            else:
                # Refine top performer
                parent = programs_only[0]
                label = self.REFINE_LABEL
                # Context: diverse programs from mid-range
                mid_start = max(1, len(programs_only) // 4)
                mid_end = max(mid_start + 1, len(programs_only))
                mid_pool = programs_only[mid_start:mid_end]
                context = random.sample(mid_pool, min(num_context_programs, len(mid_pool)))
                context = [p for p in context if p.id != parent.id][:num_context_programs]
                return {label: parent}, {"": context}
        else:
            # Not stagnating: softmax-weighted selection among top half
            top_half = programs_only[:max(1, len(programs_only) // 2)]
            top_scores = scores_only[:len(top_half)]

            # Softmax weights
            min_s = min(top_scores)
            shifted = [s - min_s for s in top_scores]
            total = sum(shifted) + 1e-9
            weights = [s / total for s in shifted]

            r = random.random()
            cumulative = 0.0
            parent = top_half[-1]
            for w, p in zip(weights, top_half):
                cumulative += w
                if r <= cumulative:
                    parent = p
                    break

            # Context: diverse selection from rest of population
            rest = [p for p in programs_only if p.id != parent.id]
            if rest:
                context = random.sample(rest, min(num_context_programs, len(rest)))
            else:
                context = []

            return {"": parent}, {"": context}


# EVOLVE-BLOCK-END