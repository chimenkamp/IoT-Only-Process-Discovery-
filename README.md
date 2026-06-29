# IoT-focused Process Discovery

This implementation follows the paper draft pipeline for exploratory process
discovery from raw IoT sensor values.

## Pipeline

1. Normalize raw sensor values.
2. Detect candidate change points.
3. Map each segment to envelope features and an `esig` truncated signature of
   the time-augmented path.
4. Merge segment occurrences that are separable by the interval grammar.
5. Synthesize one bounded SyGuS predicate per segment class.
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
- `src/synthesis.py`: CVC5-backed bounded SyGuS rule synthesis over segment
  features.
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

## SyGuS Backend

Rule synthesis uses the CVC5 Python API with `sygus=true` and an actual SyGuS
grammar. The implemented grammar is the paper's bounded interval fragment:
`Rule ::= true | Atom | (and Atom Rule)` and
`Atom ::= c <= xi | xi <= c`. Candidate constants are finite: for each feature
they are the expanded positive-class hull bounds. By default the grammar allows
up to `2 * n_features` predicate atoms; `rule_max_predicates` can make that
bound explicit.

There is no hidden Python enumerator fallback. If CVC5 is unavailable, returns
`UNKNOWN`, or cannot separate the positive and negative segment profiles within
the configured grammar, synthesis fails. A requested positive `rule_margin` is
part of the SyGuS specification; it is not silently reduced. Each rule records
the exact solver setup in its `SygusCertificate`.

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
then discovers an unsupervised SyGuS interval-rule model from the sensor
segments.

## Tests

```bash
pytest tests
```
