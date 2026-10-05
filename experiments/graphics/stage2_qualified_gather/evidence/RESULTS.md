# Qualified stage-2 results

All errors are in scene-linear RGB. PSNR peak = 1 linear radiance unit.
Noise is the equal-N pair difference divided by sqrt(2); reference A is used.

| f-number | gather | region | MAE | RMSE | PSNR dB | reference noise RMS | combined noise / MAE |
|---:|---|---|---:|---:|---:|---:|---:|
| 1.2 | naive | whole | 0.0044990359 | 0.019692141 | 34.114 | 2.7194087e-05 | 0.6061% |
| 1.2 | weighted | whole | 0.0016380986 | 0.0079728202 | 41.968 | 2.7194087e-05 | 1.6648% |
| 1.2 | naive | depth_edges | 0.021950842 | 0.046600496 | 26.632 | 5.5544343e-05 | 0.2535% |
| 1.2 | weighted | depth_edges | 0.0063427372 | 0.018487186 | 34.663 | 5.5544343e-05 | 0.8776% |
| 22 | naive | whole | 0.000412999 | 0.00293858 | 50.637 | 1.3181509e-05 | 3.4298% |
| 22 | weighted | whole | 0.0002921051 | 0.0015370538 | 56.266 | 1.3181509e-05 | 4.8510% |
| 22 | naive | depth_edges | 0.0012795026 | 0.0070549325 | 43.03 | 2.4264459e-05 | 2.0982% |
| 22 | weighted | depth_edges | 0.00071626174 | 0.0035487919 | 48.998 | 2.4264459e-05 | 3.7503% |
