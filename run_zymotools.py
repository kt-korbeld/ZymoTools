#!/usr/bin/env python3
"""
Wrapper to run the pipeline without installation.
Adds this directory to ``sys.path`` and calls the package CLI, so both
of these work from anywhere (including inside a SLURM ``--wrap`` job):

python zymotools.py <command> [options]
python -m zymotools <command> [options]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from zymotools.cli import main

if __name__ == "__main__":
    sys.exit(main())
