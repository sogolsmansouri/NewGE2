import argparse
import sys

import gege as ge


def main():
    parser = argparse.ArgumentParser(description="Configuration file based evaluation", prog="eval")

    parser.add_argument(
        "config",
        metavar="config",
        type=str,
        help=(
            "Path to YAML configuration file that describes the evaluation process. See documentation"
            " docs/config_interface for more details."
        ),
    )

    args = parser.parse_args()
    # Evaluation must not rewrite the checkpoint's full_config.yaml. Doing so
    # mutates saved checkpoints and can poison later training runs that reuse
    # checkpoint configs as templates.
    config = ge.config.loadConfig(args.config, save=False)
    ge.manager.gege_eval(config)


if __name__ == "__main__":
    sys.exit(main())
