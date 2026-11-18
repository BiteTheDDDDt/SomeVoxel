# CUDA lattice storage and precision

The probe uses the dense 3 cm fixture and the local 1 cm fixture from the matched CUDA experiments. The force field moves once and then releases the grains. Push, gravity, paired normal and friction impulses, slip and the fixed wall all run in each variant. Arithmetic and render output use FP32. Travel is stored in cell widths so the face-crossing threshold is exactly one in every format.

## Reproduce

```sh
python -m pip install warp-lang==1.19.0
python experiments/cuda_motion_precision.py dense --repeats 4 --profile
python experiments/cuda_motion_precision.py local --scale 3 --repeats 4 --formats fp32 packed-cells packed-fp16-v packed-fp16 packed-cells-gather packed-fp16-v-gather
python experiments/cuda_motion_precision.py local --scale 3 --formats fp32 packed-cells-gather packed-fp16-v-gather packed-q13-v-gather
```

The JSON rows report CUDA event timings, logical buffer sizes, center of mass, occupancy and wall checks, and SHA-256 hashes of decoded cells and stored motion. The density difference measures total variation of normalized grain counts in 12 cm spatial bins. It is zero when the bins agree and reaches one when the distributions have disjoint support. Frame P95 is measured over frames 9 through 72 of each run. The table uses the median P95 across four runs on an RTX 5070 Ti Laptop GPU. Timings cover each 1/60-second simulation batch and GPU position packing. Drawing, application work, host transfers, compilation and validation are outside that interval.

| Storage and slip arbitration | 3 cm, 1,055,832 grains | 3 cm buffers | 1 cm, 840,000 grains | 1 cm buffers |
| --- | ---: | ---: | ---: | ---: |
| FP32, dense claims | 4.50 ms | 135.3 MiB | 11.17 ms | 1964.9 MiB |
| Packed integer cells, dense claims | 3.83 ms | 127.3 MiB | 10.72 ms | 1958.5 MiB |
| Packed integer cells, gathered claims | 4.03 ms | 92.8 MiB | 9.77 ms | 1002.5 MiB |
| Packed cells, FP16 velocity, dense claims | 3.30 ms | 121.2 MiB | 8.69 ms | 1953.7 MiB |
| Packed cells, FP16 velocity and travel, dense claims | 2.91 ms | 115.2 MiB | 7.67 ms | 1948.9 MiB |
| Packed cells, FP16 velocity, gathered claims | 3.33 ms | 86.8 MiB | 9.65 ms | 997.7 MiB |

Packing the integer cell coordinates into a 10/9/10-bit word uses four bytes per grain. The ordinary and gathered-claim FP32 variants both reproduced the full FP32 state hash in both scenes. The gathered variant scans four candidate sources for each proposed slip and preserves the highest-ID priority. This replaces a dense 4-byte claim buffer per voxel with one byte per grain and an additional dispatch. The local fixture saved 962 MiB of tracked buffers when this was combined with coordinate packing. Its median P95 was lower with overlapping timing ranges. The primary occupancy buffer still spans the entire fixture at 957 MiB.

FP16 velocity with FP32 travel produced a 0.076 cm center-of-mass difference and 4.22% density total variation at 3 cm. The same format produced a 9.73 cm difference and 11.32% density total variation at 1 cm. Quantizing both quantities to FP16 produced a 4.97 cm difference and 9.15% total variation at 3 cm, and 1.50 cm and 4.33% at 1 cm. The Q13 signed fixed-point velocity option behaved similarly to FP16 velocity in the 1 cm scene, with 9.67 cm and 10.96% differences. The dense Q14 travel run gave a 1.80 cm difference and 7.18% total variation. Its isolated 0.25 mm drift became 0.275 mm. Eight-bit travel rounded every increment of that same drift to zero. Its dense bulk test increased to 6.25 cm and 10.16% differences.

The isolated equal-mass oblique collision stayed within 0.00020 m/s per component in the tested FP16 formats. The wall restitution test returned −0.60010 m/s against an expected −0.60000 m/s after a 2.4 m/s approach. All runs retained their particle IDs and static walls, and each individual format reproduced its own hash. Use the exact coordinate and arbitration variants as the baseline for future tile paging. Motion quantization requires more scene-level error checks before use for gameplay contacts.
