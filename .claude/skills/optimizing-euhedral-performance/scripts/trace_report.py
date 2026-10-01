#!/usr/bin/env python3
"""trace_report.py SQLITE [TOP]: decode token anatomy from a profile.sh trace.

Tokens are delimited by the 1-row embedding launches (grid 20). Reports the token period, GPU busy time,
the idle gap at the token boundary (last kernel end -> next embedding start) and, per kernel, exclusive
time: end minus the later of its start and every earlier kernel's end. Raw kernel durations overstate
cost because a kernel launched with programmatic dependent launch starts early and waits inside
griddepcontrol.wait; exclusive time removes that wait (and overlap with other lanes).
"""
import collections, sqlite3, statistics, sys

db = sqlite3.connect(sys.argv[1])
top = int(sys.argv[2]) if len(sys.argv) > 2 else 20
rows = db.execute("""select k.start, k.end, s.value, k.gridX from CUPTI_ACTIVITY_KIND_KERNEL k
                     join StringIds s on s.id = k.shortName order by k.start""").fetchall()
tokens = [r for r in rows if r[2] == 'euhedral_q3_embedding' and r[3] == 20]
if len(tokens) < 3:
    sys.exit('no decode tokens (1-row embeddings) in this trace')
periods, gaps, busy = [], [], []
exclusive = collections.defaultdict(lambda: [0, 0])
counted = 0
index = 0
for a, b in zip(tokens, tokens[1:]):
    if b[0] - a[0] > 50e6:  # a run boundary, not a token
        continue
    while index < len(rows) and rows[index][0] < a[0]:
        index += 1
    segment = []
    j = index
    while j < len(rows) and rows[j][0] < b[0]:
        segment.append(rows[j]); j += 1
    counted += 1
    periods.append((b[0] - a[0]) / 1e3)
    last = a[0]
    for start, end, name, _ in sorted(segment, key=lambda r: r[1]):
        exclusive[name][0] += max(0, end - max(start, last)); exclusive[name][1] += 1
        last = max(last, end)
    gaps.append((b[0] - max(r[1] for r in segment)) / 1e3)
    spans = sorted((r[0], r[1]) for r in segment)
    total, (cs, ce) = 0, spans[0]
    for s, e in spans[1:]:
        if s > ce: total += ce - cs; cs, ce = s, e
        else: ce = max(ce, e)
    busy.append((total + ce - cs) / 1e3)
print(f'tokens {counted}  period {statistics.median(periods):.1f} us  busy {statistics.median(busy):.1f} us  '
      f'boundary gap {statistics.median(gaps):.1f} us  launches/token {sum(v[1] for v in exclusive.values()) / counted:.0f}')
print(f"{'kernel':48s} {'us/token':>9s} {'calls':>6s} {'us/call':>8s}")
for name, (ns, calls) in sorted(exclusive.items(), key=lambda kv: -kv[1][0])[:top]:
    print(f'{name[:48]:48s} {ns / counted / 1e3:9.1f} {calls / counted:6.1f} {ns / calls / 1e3:8.2f}')
