from __future__ import annotations

import numpy as np

from src.pipeline import PipelineConfig, PipelineResult, run_pipeline


def main() -> None:
    data, case_boundaries = _example_sensor_log()
    result = run_pipeline(PipelineConfig(
        data=data,
        penalty=0.05,
        min_segment_size=10,
        var_names=["position", "pressure"],
        case_boundaries=case_boundaries,
        output_path="model.png",
        signature_debug_path="signature_debug.png",
    ))

    print(f"Detected {len(result.boundaries) - 1} segments")
    print(f"Synthesized {result.n_classes} rule activities")
    print()
    for rule in result.rules:
        print(f"  Rule {rule.class_id}: {len(rule.lo)} feature intervals")
    print()
    for idx, trace in enumerate(result.traces):
        print(f"  Trace {idx}: {trace}")
    print()
    print(f"Discovered Petri net: {len(result.net.transitions)} transitions, "
          f"{len(result.net.places)} places")
    print("Model saved to model.png")
    print("Signature debug view saved to signature_debug.png")


def _example_sensor_log() -> tuple[np.ndarray, list[int]]:
    case = np.vstack([
        np.tile([0.0, 0.0], (40, 1)),
        np.tile([1.0, 0.0], (40, 1)),
        np.tile([1.0, 1.0], (40, 1)),
    ])
    data = np.vstack([case, case])
    case_boundaries = [0, len(case), len(data)]
    return data, case_boundaries


if __name__ == "__main__":
    main()
