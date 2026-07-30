"""AOSpectrum command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from aospectrum import __version__
from aospectrum.band import (
    compare_band_results,
    read_band_result,
    write_band_comparison,
)
from aospectrum.cli.inspect import print_inspection
from aospectrum.cli.view import serve_artifact
from aospectrum.errors import AOSpectrumError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aospectrum",
        description=(
            "Sparse Band and selected-state Orbital tools for localized "
            "atomic-orbital Hamiltonians."
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subcommands.add_parser(
        "inspect",
        help="inspect an operator bundle without solving",
    )
    inspect_parser.add_argument("input", type=Path)

    band_parser = subcommands.add_parser(
        "band",
        help="run an energy-window Band calculation",
    )
    band_parser.add_argument("--config", required=True, type=Path)
    band_parser.add_argument(
        "--resume",
        action="store_true",
        help="resume the exact interrupted run recorded beside the output",
    )

    orbital_parser = subcommands.add_parser(
        "orbital",
        help="run an indexed-state Orbital calculation",
    )
    orbital_parser.add_argument("--config", required=True, type=Path)
    orbital_parser.add_argument(
        "--resume",
        action="store_true",
        help="resume the exact interrupted eigensystem and state fields",
    )

    view_parser = subcommands.add_parser(
        "view",
        help="serve one generated result over local HTTP",
    )
    view_parser.add_argument("output_dir", type=Path)
    view_parser.add_argument("--host", default="127.0.0.1")
    view_parser.add_argument("--port", type=int, default=8000)

    compare_parser = subcommands.add_parser(
        "compare-band",
        help="compare same-path Band energies in eV",
    )
    compare_parser.add_argument("--new", required=True, type=Path)
    compare_parser.add_argument("--reference", required=True, type=Path)
    compare_parser.add_argument(
        "--output",
        type=Path,
        default=Path("band-comparison"),
    )
    compare_parser.add_argument("--midpoint-ev", type=float, default=1.0e-5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "inspect":
            print_inspection(arguments.input)
        elif arguments.command == "band":
            from aospectrum.cli.band import run_band_config

            run_band_config(arguments.config, resume=arguments.resume)
        elif arguments.command == "orbital":
            from aospectrum.cli.orbital import run_orbital_config

            run_orbital_config(arguments.config, resume=arguments.resume)
        elif arguments.command == "view":
            serve_artifact(
                arguments.output_dir,
                host=arguments.host,
                port=arguments.port,
            )
        elif arguments.command == "compare-band":
            new = read_band_result(arguments.new)
            reference = read_band_result(arguments.reference)
            comparison = compare_band_results(new, reference)
            output = write_band_comparison(
                comparison,
                new,
                arguments.output,
                midpoint_ev=arguments.midpoint_ev,
            )
            print(output)
        else:  # pragma: no cover - argparse owns command validation
            raise AssertionError(arguments.command)
    except AOSpectrumError as exc:
        print(f"aospectrum: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
