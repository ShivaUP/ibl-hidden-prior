# Event-delta audit (behavior-core)

created_utc: 2026-07-21T01:21:16.465239+00:00
n_eids: 10

## Pooled deltas (QC-pass trials only)

### go_minus_stim

- n: 9818
- median: 0.01600000000030377
- mean: 0.009499280233584371
- p05: 0.0008333333333325754
- p25: 0.0009666666669545521
- p75: 0.017000000000280124
- p95: 0.01979999999997517
- frac_within_100ms: 0.9998981462619678
- frac_negative: 0.0

### resp_minus_go

- n: 9818
- median: 0.4133850149351588
- mean: 1.2521059610801704
- p05: 0.1916999999998552
- p25: 0.29033775435800635
- p75: 0.9495433721324957
- p95: 4.575550000000042
- frac_within_100ms: 0.005296394377673661
- frac_negative: 0.0

### fb_minus_resp

- n: 9818
- median: 0.00010000000020227162
- mean: 0.00431826313385751
- p05: 8.497065181245489e-05
- p25: 9.999999974752427e-05
- p75: 0.0007571951335876292
- p95: 0.03049999999996089
- frac_within_100ms: 1.0
- frac_negative: 0.0

### off_minus_stim

- n: 4750
- median: 1.4835166666666737
- mean: 2.3762639719298253
- p05: 1.2666816666666363
- p25: 1.350233333333108
- p75: 2.3504583333331084
- p95: 5.55221833333329
- frac_within_100ms: 0.0
- frac_negative: 0.0

## Phase-map suggestion

```json
{
  "bin_size_ms": 100,
  "bin0": "stimOn_times",
  "notes": [
    "If median(go-stim) << 100ms, stimulus_right/contrast_high and response_window may both start in bin 0.",
    "response_made occupies the bin containing response_times - stimOn.",
    "reward occupies the bin containing feedback_times - stimOn.",
    "stimOff relative delay informs when stimulus channels turn off."
  ],
  "pooled_median_go_minus_stim_s": 0.01600000000030377,
  "pooled_median_resp_minus_go_s": 0.4133850149351588,
  "pooled_median_off_minus_stim_s": 1.4835166666666737
}
```
