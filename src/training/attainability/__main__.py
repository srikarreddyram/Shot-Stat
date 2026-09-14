"""CLI entrypoint: `python -m src.training.attainability --name ...`.

Split into its own file because `python -m <package>` runs a package's
__main__.py, never its __init__.py — the argparse block used to sit at the
bottom of the single attainability.py, and has to live here now that this
is a package.
"""
import argparse

from .train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the attainability model.")
    parser.add_argument("--name", default="attainability")
    parser.add_argument("--no-cast", action="store_true",
                        help="Ablate the leave-one-out supporting-cast features")
    parser.add_argument("--no-prior-diet", action="store_true",
                        help="Ablate the player's own prior-season shot diet")
    args = parser.parse_args()
    train(name=args.name, use_cast=not args.no_cast,
          use_prior_diet=not args.no_prior_diet)
