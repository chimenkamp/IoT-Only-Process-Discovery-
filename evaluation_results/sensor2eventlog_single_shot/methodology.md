# Single-shot Sensor2EventLog Benchmark

This benchmark compares against the evaluation protocol described in
`Paper/Sensor2EventLog.pdf`.

## Constraints Applied

- No Sensor2EventLog event-rule family was added to our method.
- No iterative planning/explaining/reviewing loop was used.
- No domain thresholds such as `Qin > tau_Q`, `T > 70`, `LIT101_diff_smooth < 0`,
  or `LIT101_stability > 0.8` were used by our pipeline.
- Reference labels are used only after discovery to compute the paper's rule
  coverage, precision, and effectiveness metrics.

## HACCP

The local HACCP dataset contains both `batch_id` and `state`, so the paper's
coverage metrics are computed directly on held-out batches. By default this
combined script copies the existing all-batch HACCP artifacts from
`evaluation_results/haccp_pasteurization`, because regenerating changepoints for
all 968 batches is slow. Set `RERUN_HACCP=1` to force regeneration.

## SWaT P1

The local SWaT workbook contains raw P1 sensor streams and controller/status
columns. It does not contain the prepared `batch_id` column used by the public
Sensor2EventLog implementation, but it does contain a `P1_STATE` column. The
benchmark therefore uses `P1_STATE` only after discovery to compute anonymous
rule coverage, precision, effectiveness, and a train-mapped timestamp accuracy.

For transparency, the SWaT run uses only the normal-operation window from the
provided data-collection note (2019-07-20T04:35:00+00:00 to
2019-07-20T06:50:00+00:00, GMT+0) and four equal contiguous pseudo-batches
for process-mining formatting. These pseudo-batches are not domain batches, so
the reported SWaT accuracy is still a single-shot workbook-label diagnostic
rather than a reproduction of the paper's teacher-guided HMM/K-means results.
