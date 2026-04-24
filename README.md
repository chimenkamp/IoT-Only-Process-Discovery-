# IoT-focused Process Discovery

This project discovers process models from multivariate sensor streams.
It segments time series data, merges compatible segments, synthesizes interval rules, and builds Petri nets.

## Pipeline

1. Detect change points in a sensor stream.
2. Build segment profiles from per-sensor value ranges.
3. Merge compatible segments into equivalence classes.
4. Synthesize one interval rule per class with SMT.
5. Convert rule activations into traces.
6. Discover a Petri net with pm4py.

## Repository Layout

- `main.py`: synthetic example and reusable pipeline entry point.
- `main_real_test.py`: real-data example for the Future Factory dataset.
- `src/changepoint.py`: change point detection with PELT.
- `src/merging.py`: segment profiling and clique-cover merging.
- `src/synthesis.py`: interval-rule synthesis.
- `src/trace.py`: trace construction from rules.
- `src/discovery.py`: process discovery and Petri net export.
- `tests/test_pipeline.py`: pipeline tests on synthetic data.
- `data/CASAS/`: CASAS dataset files.
- `data/Future_Factory/`: Future Factory data and notebook assets.
- `paper/`: paper sources.
- `docs/`: diagrams and notes.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

Run the synthetic example:

```bash
python main.py
```

This writes `model.png` in the project root.

Run the Future Factory example:

```bash
python main_real_test.py
```

This expects `data/Future_Factory/combined_[1-6].pkl`.
This writes `real_model.png` in the project root.

## Tests

Run the test suite:

```bash
pytest tests/test_pipeline.py
```

## Notes

The notebooks are exploratory and complement the Python entry points.
The pipeline code is centered in `main.py` and `src/`.