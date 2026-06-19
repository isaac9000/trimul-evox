# EVOLVE-BLOCK-START
from skydiscover.search.base_database import Program, ProgramDatabase
from skydiscover.config import DatabaseConfig
from typing import Optional, Tuple, List, Dict
import logging
from dataclasses import dataclass
import random

logger = logging.getLogger(__name__)

@dataclass
class EvolvedProgram(Program):
    """Program for the evolved database."""
    

class EvolvedProgramDatabase(ProgramDatabase):
    """Initial search strategy database."""

    DIVERGE_LABEL = "diverge"
    REFINE_LABEL = "refine"

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self._sample_count = 0

        if config.random_seed is not None:
            random.seed(config.random_seed)
            logger.debug(f"Database: Set random seed to {config.random_seed}")

    def _score_of(self, program) -> float:
        """Safely extract a numeric combined_score from a program."""
        score = None
        metrics = getattr(program, "metrics", None)
        if isinstance(metrics, dict):
            score = metrics.get("combined_score", None)
        if score is None:
            score = getattr(program, "combined_score", None)
        try:
            score = float(score)
        except (TypeError, ValueError):
            return float("-inf")
        return score

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        """Add a program to the database."""
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program
        
        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)

        logger.debug(f"Added program {program.id} to the evolve database")
        return program.id

    def sample(
        self, 
        num_inspirations: Optional[int] = 4,
        **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        """
        Exploitation-heavy quality-biased sampling.

        Approach:
          1. Rank candidates by combined_score (best first).
          2. REFINE ~3/4 of samples: pick parent from a narrow top pool with
             quadratic bias toward rank #1 to incrementally improve the best.
          3. DIVERGE ~1/4 of samples: pick parent more broadly (rank-weighted)
             to escape local optima.
          4. Inspirations always anchor on the global best, add further top
             elites, and reserve one slot for a novel/random program.
        """
        candidates = list(self.programs.values())

        if len(candidates) == 0:
            raise ValueError("No candidates available for sampling")

        if num_inspirations is None:
            num_inspirations = 4

        self._sample_count += 1

        # Rank candidates: best (highest score) first.
        ranked = sorted(candidates, key=self._score_of, reverse=True)
        n = len(ranked)

        best = ranked[0]

        # Decide mode: strongly favor REFINE (exploit), since history shows the
        # best gains (116.8) came from incrementally improving top programs
        # while broad exploration regressed. Explore only ~1 in 4 to retain
        # diversity and escape local optima without wasting samples.
        diverge = (self._sample_count % 4 == 0)
        mode_label = self.DIVERGE_LABEL if diverge else self.REFINE_LABEL

        # ---- Parent selection ----
        if diverge:
            # Explore: bias toward the broader population (including weaker
            # programs) to escape local optima, but still weighted by rank.
            weights = [1.0 / (i + 1) ** 0.5 for i in range(n)]
            parent = random.choices(ranked, weights=weights, k=1)[0]
        else:
            # Exploit: focus tightly on the very best performers. A narrow top
            # pool with quadratic bias toward rank #1 mirrors the successful
            # incremental-improvement trajectory observed in history.
            top_k = max(1, min(n, max(2, (n + 4) // 5)))
            top_pool = ranked[:top_k]
            weights = [1.0 / (i + 1) ** 2 for i in range(len(top_pool))]
            parent = random.choices(top_pool, weights=weights, k=1)[0]

        # ---- Inspiration selection ----
        # Always anchor on the global best program (if not the parent) so its
        # strong building blocks are consistently available for transfer.
        examples = []
        chosen_ids = {parent.id}
        if num_inspirations > 0 and best.id not in chosen_ids:
            examples.append(best)
            chosen_ids.add(best.id)

        # Add additional top elites to share high-quality structure, but reserve
        # at least one slot for novelty when possible.
        elites = [p for p in ranked if p.id not in chosen_ids]
        num_elite = max(0, (num_inspirations - len(examples)) - 1)
        for p in elites[:num_elite]:
            examples.append(p)
            chosen_ids.add(p.id)

        # Fill remaining slots with diverse/random programs for novelty.
        remaining_pool = [p for p in candidates if p.id not in chosen_ids]
        random.shuffle(remaining_pool)
        for p in remaining_pool:
            if len(examples) >= num_inspirations:
                break
            examples.append(p)
            chosen_ids.add(p.id)

        examples = examples[:num_inspirations]

        parent_dict = {mode_label: parent}
        inspiration_programs_dict = {mode_label: examples}

        return parent_dict, inspiration_programs_dict

# EVOLVE-BLOCK-END
