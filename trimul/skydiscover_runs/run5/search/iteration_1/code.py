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
        Quality-biased sampling that balances exploitation and exploration.

        Approach:
          1. Rank candidates by combined_score (best first).
          2. Alternate between REFINE mode (exploit: pick parent from the top
             performers to incrementally improve them) and DIVERGE mode
             (explore: pick parent more broadly, including weaker/diverse
             programs to escape local optima).
          3. Inspiration set always includes the current best programs (to
             share strong building blocks) plus some random/diverse programs
             (to inject novelty), avoiding duplicates and the parent itself.
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

        # Decide mode: refine (exploit) most of the time, diverge periodically.
        # Roughly 1 in 3 samples explores to maintain diversity.
        diverge = (self._sample_count % 3 == 0)
        mode_label = self.DIVERGE_LABEL if diverge else self.REFINE_LABEL

        # ---- Parent selection ----
        if diverge:
            # Explore: bias toward the broader population (including weaker
            # programs) to escape local optima, but still weighted by rank.
            weights = [1.0 / (i + 1) ** 0.5 for i in range(n)]
            parent = random.choices(ranked, weights=weights, k=1)[0]
        else:
            # Exploit: pick from the top fraction of performers.
            top_k = max(1, min(n, (n + 3) // 4))
            top_pool = ranked[:top_k]
            # Stronger bias toward the very best within the top pool.
            weights = [1.0 / (i + 1) for i in range(len(top_pool))]
            parent = random.choices(top_pool, weights=weights, k=1)[0]

        # ---- Inspiration selection ----
        # Always include current best programs as inspirations (excluding parent).
        elites = [p for p in ranked if p.id != parent.id]
        num_elite = max(1, num_inspirations // 2) if num_inspirations > 0 else 0
        elite_picks = elites[:num_elite]

        # Fill remaining slots with diverse/random programs for novelty.
        chosen_ids = {parent.id} | {p.id for p in elite_picks}
        remaining_pool = [p for p in candidates if p.id not in chosen_ids]
        random.shuffle(remaining_pool)

        examples = list(elite_picks)
        for p in remaining_pool:
            if len(examples) >= num_inspirations:
                break
            examples.append(p)

        examples = examples[:num_inspirations]

        parent_dict = {mode_label: parent}
        inspiration_programs_dict = {mode_label: examples}

        return parent_dict, inspiration_programs_dict

# EVOLVE-BLOCK-END
