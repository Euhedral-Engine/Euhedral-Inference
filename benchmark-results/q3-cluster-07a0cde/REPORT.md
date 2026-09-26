# Q3 opt-in CTA-cluster reuse experiment

## Outcome

Observed: no cluster composition wins for the measured tensors at 256 rows on NVIDIA GeForce RTX 5070 Ti, 595.84. The strategy remains opt-in; normal Q3 dispatch, ABI, and NVRTC C++14 source are unchanged. No production deployment was performed.

## Method

- Compared compile-time cluster shapes `1x1`, `2x1`, `4x1`, `1x2`, `1x4`, and `2x2` with matching CTA-local 32-row and 64-row tiles. A 2-D physical cluster owns per-output CTA accumulators; CTA `(m,0)` produces A and `(0,n)` produces B for each K group. Shared-memory remote borrowing is guarded by cluster barriers; consumer CTAs copy into their existing CTA-local execution tile.
- Read the actual compact Q3 tensors from `qwen3_5_27b_compact_q3.edrl`, layer 0: gate/up, down, GDN mixer output. Input activations were deterministic finite synthetic BF16, not a captured model trace. Packed tensor SHA-256, offsets, sizes, dimensions, GPU and driver are in the JSONL.
- For each pair, separate output buffers were initialized to `0xa5`; missing writes and bitwise differences fail before timing. All 36 pairs passed. CUDA events enclose eight asynchronous launches, with nine timed batches and three warmups; reported times are medians per launch. CTA baseline was resampled beside each cluster variant, alternating measurement order.
- The CTA baseline is the *matched tile*, not always production AUTO dispatch. AUTO uses the 64-row tile for gate/up and down at this shape, and the 32-row tile for mixer output. Timings are isolated operator measurements on a resident serving GPU, not end-to-end throughput. Temperature/frequency and cache effects may shift close ratios; these differences are not close.

## Latency (ms): matched CTA / cluster shapes

| Tensor | CTA rows | CTA | 1x1 | 2x1 | 4x1 | 1x2 | 1x4 | 2x2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gate_up | 32 | 11.986 | 27.204 | 38.569 | 39.875 | 33.966 | 34.579 | 44.010 |
| gate_up | 64 | 9.061 | 17.737 | 23.469 | 24.546 | 23.697 | 25.028 | 28.978 |
| down | 32 | 5.874 | 12.093 | 18.037 | 18.615 | 15.590 | 15.910 | 20.517 |
| down | 64 | 4.903 | 8.568 | 11.674 | 11.985 | 12.267 | 12.103 | 14.317 |
| output | 32 | 2.032 | 4.341 | 6.546 | 6.694 | 5.703 | 5.826 | 7.439 |
| output | 64 | 1.706 | 3.130 | 4.226 | 4.547 | 4.274 | 4.491 | 5.442 |

Observed: all 36 cluster/CTA ratios are slower, spanning 1.66x to 3.67x. B-only, A-only, and combined reuse each lose to the `1x1` cluster itself in every measured matrix/tile pair. Thus these results do not justify production selection.

## Synchronization and static resources

- Per 64-wide K group: a CTA barrier and cluster barrier precede the remote tile copies, a CTA barrier precedes MMA consumption, and a second cluster barrier protects the producer tile until all consumers finish. This means two cluster-wide barriers per K group plus additional shared-memory traffic. The measured `1x1` gap is a useful cluster/synchronization overhead control; it is not a claimed isolated barrier microbenchmark.
- Driver function attributes: 32-row CTA uses 88 registers/thread versus 96 for the cluster kernel, both 16 KiB static shared/CTA; 64-row CTA uses 114 versus 128 registers/thread, both 24 KiB shared/CTA. Driver reports zero local bytes/thread in all measured variants. Cluster shapes can further constrain residency because CTAs are scheduled together. No Nsight achieved-occupancy, stall, or bandwidth counters were collected; these are explanations to test, not measured bottleneck attributions.

## Reproduce

From the repository root with the pinned CUDA development package and a CUDA-capable GPU: `python benchmark-results/q3-cluster-07a0cde/real_weight_cluster.py`. The script reads the local artifact path named at its top and overwrites its adjacent JSONL. Run `python -m unittest discover -s native/tests -v` and `mise exec -- gradle nativeVerify :api:bootJar build`. The installed product includes cluster source but no production symbol lookup or dispatch for it.

Data: `real_weight_cluster.jsonl`. Source: `real_weight_cluster.py`. Previous pre-review measurements were retained locally only; this JSONL is the final complete-output run with Python `-O` to confirm numerical checks are not disabled.
