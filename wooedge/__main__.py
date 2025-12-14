"""
Entry point for running WOOEdge as a module.

Usage:
    python -m wooedge.cli run --seed 0 --steps 200 --render ascii
    python -m wooedge.cli benchmark --seeds 0 1 2 3 4
    python -m wooedge.cli demo
"""

from .cli import main

if __name__ == "__main__":
    main()
