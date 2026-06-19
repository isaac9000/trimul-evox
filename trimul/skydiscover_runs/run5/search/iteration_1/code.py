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

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None

        if config.random_seed is not None:
            random.seed(config.random_seed)
            logger.debug(f"Database: Set random seed to {config.random_seed}")

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
        Picks parent via recency-weighted sampling and inspirations using
        REFINE_LABEL/DIVERGE_LABEL keys to guide exploitation vs exploration.
        Recent programs are preferred as parents (higher iteration_found = more weight).
        Inspirations are split: half from recent programs (refine), half random (diverge).
        """
        candidates = list(self.programs.values())

        if len(candidates) == 0:
            raise ValueError("No candidates available for sampling")

        # Weight candidates by recency (iteration_found)
        max_iter = max((getattr(p, 'iteration_found', 0) or 0) for p in candidates)
        weights = [1 + (getattr(p, 'iteration_found', 0) or 0) for p in candidates]
        parent = random.choices(candidates, weights=weights, k=1)[0]

        # Split inspirations: refine from recent, diverge from random
        num_refine = max(1, num_inspirations // 2)
        num_diverge = num_inspirations - num_refine

        # Recent candidates for refinement (top half by iteration)
        sorted_cands = sorted(candidates, key=lambda p: getattr(p, 'iteration_found', 0) or 0, reverse=True)
        recent = [p for p in sorted_cands if p.id != parent.id]
        old = [p for p in candidates if p.id != parent.id]

        refine_inspirations = recent[:min(num_refine, len(recent))]
        diverge_inspirations = random.sample(old, min(num_diverge, len(old))) if old else []

        parent_dict = {self.REFINE_LABEL: parent}
        inspiration_programs_dict = {
            self.REFINE_LABEL: refine_inspirations,
            self.DIVERGE_LABEL: diverge_inspirations,
        }

        return parent_dict, inspiration_programs_dict

# EVOLVE-BLOCK-END
