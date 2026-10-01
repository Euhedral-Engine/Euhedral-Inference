#!/usr/bin/env python3
"""summarize.py OUTDIR: per metric, the median over forks of each fork's median, control vs candidate,
and in how many forks the candidate was ahead. Warmups and failed iterations are excluded."""
import collections, glob, json, statistics as st, sys

out = sys.argv[1]
rows = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
failures = collections.Counter()
for path in glob.glob(f'{out}/*-*.jsonl'):
    arm, fork = path.rsplit('/', 1)[1][:-6].rsplit('-', 1)
    for line in open(path):
        r = json.loads(line)
        if r['status'] != 'success':
            failures[arm] += 1
            continue
        if r['warmup']:
            continue
        s, t, tm = r['scenario']['name'], r['throughput'], r['timings']
        rows[s + ' end-to-end ms'][arm][fork].append(tm['endToEnd'] / 1e6)
        if r['scenario']['kind'] == 'decode':
            rows[s + ' decode tok/s'][arm][fork].append(t['decodeTokensPerSecond'])
            rows[s + ' TTFT ms'][arm][fork].append(tm['timeToFirstToken'] / 1e6)
        else:
            rows[s + ' prefill tok/s'][arm][fork].append(t['prefillTokensPerSecond'])
if failures:
    print('FAILED iterations:', dict(failures))
print(f"{'metric':32s} {'control':>10s} {'candidate':>10s} {'change':>8s}  candidate ahead")
for metric in sorted(rows):
    c = {f: st.median(v) for f, v in rows[metric]['control'].items()}
    d = {f: st.median(v) for f, v in rows[metric]['candidate'].items()}
    if not c or not d:
        continue
    cm, dm = st.median(c.values()), st.median(d.values())
    higher_better = 'tok/s' in metric
    forks = sorted(set(c) & set(d))
    ahead = sum(1 for f in forks if (d[f] > c[f]) == higher_better and d[f] != c[f])
    print(f"{metric:32s} {cm:10.2f} {dm:10.2f} {100 * (dm - cm) / cm:+7.2f}%  {ahead}/{len(forks)}")
