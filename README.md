# IoT-focused Process Discovery

This implementation follows the paper draft pipeline for exploratory process
discovery from raw IoT sensor values.

## Pipeline

1. Normalize raw sensor values.
2. Detect candidate change points.
3. Map each segment to envelope features and an `esig` truncated signature of
   the time-augmented path.
4. Merge segment occurrences that are separable by the interval grammar.
5. Synthesize one interval rule per segment class.
6. Build timestamped events and activity projections from rule activations.
7. Discover a Petri net with the Inductive Miner.
8. Optionally export a signature debug image.

## Repository Layout

- `main.py`: synthetic paper-shaped example.
- `main_real_test.py`: Future Factory raw-data entry point with cycle boundaries
  as case markers.
- `src/pipeline.py`: end-to-end discovery procedure.
- `src/preprocessing.py`: fitted min-max preprocessing.
- `src/changepoint.py`: PELT change point detection.
- `src/signatures.py`: `esig` segment signatures and debug image export.
- `src/merging.py`: grammar-aware segment-class merging.
- `src/synthesis.py`: interval-rule synthesis over segment features.
- `src/trace.py`: timestamped event-log construction.
- `src/discovery.py`: Inductive Miner process discovery.
- `tests/`: tests for the paper-aligned behavior.
- `Paper/`: paper draft and references.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

Run the synthetic example:

```bash
python main.py
```

This writes `model.png` and `signature_debug.png`.

Run the Future Factory example:

```bash
python main_real_test.py
```

This expects `data/Future_Factory/combined_[1-6].pkl` and writes
`real_model.png` plus `signature_debug.png`.  The script selects contiguous
`Q_Cell_CycleCount` runs as cases so repeated or reset cycle IDs are not merged,
then discovers an unsupervised clustered-interval rule model from the sensor
segments.

## Tests

```bash
pytest tests
```
