"""Command-line entry point for the multi-camera tracking baseline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.pipeline import MultiCameraTrackingPipeline
from src.utils.common import ProjectError, configure_logging, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline multi-camera person tracking demo")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML configuration")
    parser.add_argument(
        "--output-directory",
        help="Override output.output_directory without modifying the YAML file",
    )
    return parser.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()
    try:
        print("[setup] Loading configuration")
        config = load_config(args.config)
        if args.output_directory:
            config.setdefault("output", {})["output_directory"] = str(
                Path(args.output_directory).resolve()
            )
        summary = MultiCameraTrackingPipeline(config).run()
    except ProjectError as exc:
        print(f"\nERROR:\n{exc}")
        return 1
    except (KeyError, TypeError, ValueError) as exc:
        print(f"\nERROR:\nInvalid configuration or data: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\nProcessing cancelled by user.")
        return 130
    except Exception:
        logging.exception("Unexpected failure")
        return 1

    print("\n" + "=" * 44)
    print("Processing complete\n")
    for camera_id, count in summary.local_tracks_by_camera.items():
        camera_people = summary.camera_persons_by_camera[camera_id]
        print(f"{camera_id} local tracks: {count} -> camera persons: {camera_people}")
    print(f"Same-camera stitched links: {summary.intra_camera_matches}")
    print(f"Global persons: {summary.global_persons}")
    print(f"Cross-camera matches: {summary.cross_camera_matches}\n")
    print("Results:")
    print(summary.output_directory / "videos")
    print(summary.output_directory / "local_tracks.csv")
    print(summary.output_directory / "track_segments.csv")
    print(summary.output_directory / "camera_tracks.csv")
    print(summary.output_directory / "intra_camera_matches.csv")
    print(summary.output_directory / "global_tracks.csv")
    print(summary.output_directory / "tracking_diagnostics.csv")
    print(summary.output_directory / "similarity_matrix.csv")
    print("=" * 44)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
