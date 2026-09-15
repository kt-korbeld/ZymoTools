"""BindCraft hotspot-screening pipeline.

A coherent codebase wrapping four formerly-loose steps:

  1. hotspots  - score residues and generate hotspot-patch BindCraft JSONs
  2. view-*    - Shiny viewers for curating hotspots / sizing linkers
  3. screen    - self-resubmitting SLURM controller that runs BindCraft
  4. linker    - shortest-solvent-path fusion-linker length estimate

See ``cli.py`` for the command-line interface and ``run`` orchestrator.
"""

__version__ = "1.0.0"
