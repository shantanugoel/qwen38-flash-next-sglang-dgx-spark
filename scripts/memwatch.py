#!/usr/bin/env python3
"""Bounded, non-root UMA watchdog. Policy is separately testable without a GPU."""
import argparse
import json
import subprocess
import time
from pathlib import Path


def memory(path='/proc/meminfo'):
    values = {k: int(v.split()[0]) / 1048576 for k, v in
              (line.split(':', 1) for line in Path(path).read_text().splitlines())}
    return {'available_gib': values['MemAvailable'], 'free_gib': values['MemFree'],
            'swap_used_gib': values['SwapTotal'] - values['SwapFree']}


def nvidia_signals(text):
    """Xid is a GPU exception; NV_ERR_NO_MEMORY is often a one-shot alloc probe."""
    lines = [ln for ln in text.splitlines()
             if 'NVRM: Xid' in ln or 'NV_ERR_NO_MEMORY' in ln or 'Out of memory [NV_ERR' in ln]
    xid = any('NVRM: Xid' in ln for ln in lines)
    alloc = any('NV_ERR_NO_MEMORY' in ln or 'Out of memory [NV_ERR' in ln for ln in lines)
    return xid, alloc, lines


class Policy:
    def __init__(self, available=6, free=0.5, free_gate=10, sustain=15, swap_growth=0.5):
        self.available, self.free, self.free_gate = available, free, free_gate
        self.sustain, self.swap_growth = sustain, swap_growth
        self.since = None
        self.initial_swap = None

    def check(self, sample, now, xid=False, alloc=False):
        if self.initial_swap is None:
            self.initial_swap = sample['swap_used_gib']
        if xid:
            return ['new NVIDIA Xid']
        reasons = []
        if sample['available_gib'] < self.available:
            reasons.append('MemAvailable below floor')
        if sample['free_gib'] < self.free and sample['available_gib'] < self.free_gate:
            reasons.append('MemFree below gated floor')
        if sample['swap_used_gib'] - self.initial_swap > self.swap_growth:
            reasons.append('swap growth above budget')
        if alloc:
            reasons.append('new NVIDIA allocation error')
        if reasons:
            if self.since is None:
                self.since = now
            if now - self.since >= self.sustain:
                return reasons
        else:
            self.since = None
        return []


def run(args):
    policy = Policy(args.available, args.free, args.free_gate, args.sustain, args.swap_growth)
    since = time.time()
    with Path(args.log).open('x') as log:
        while not Path(args.done).exists():
            now = time.time()
            sample = memory()
            # Only inspect messages since the previous sample; old failures cannot trip it.
            p = subprocess.run(['journalctl', '-k', '--since', f'@{since:.6f}',
                                '--no-pager', '-o', 'short-iso'], capture_output=True,
                               text=True, timeout=5)
            since = now
            journal_ok = p.returncode == 0
            xid = alloc = False
            matches = []
            if journal_ok:
                xid, alloc, matches = nvidia_signals(p.stdout)
            reasons = policy.check(sample, time.monotonic(), xid=xid, alloc=alloc)
            row = {'time': now, **sample, 'journal_access': journal_ok,
                   'action': 'stop' if reasons else 'observe', 'reasons': reasons}
            if matches:
                row['journal_matches'] = matches[:20]
            log.write(json.dumps(row) + '\n'); log.flush()
            if reasons:
                Path(args.trip).write_text(json.dumps(row, indent=2))
                # Capture evidence before stopping only this experiment's container.
                tail = subprocess.run(['docker', 'logs', '--tail', '80', args.container],
                                      capture_output=True, text=True, timeout=10)
                Path(args.log + '.server-tail').write_text(tail.stdout + tail.stderr)
                stopped = subprocess.run(['docker', 'stop', '-t', '30', args.container],
                                         capture_output=True, text=True, timeout=40)
                log.write(json.dumps({'stop_returncode': stopped.returncode}) + '\n')
                return 2
            time.sleep(args.interval)
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('container', 'log', 'done', 'trip'):
        p.add_argument('--' + name, required=True)
    for name, default in [('available', 6), ('free', .5), ('free-gate', 10),
                          ('sustain', 15), ('swap-growth', .5), ('interval', 5)]:
        p.add_argument('--' + name, type=float, default=default)
    args = p.parse_args()
    if min(args.available, args.free, args.free_gate, args.sustain,
           args.swap_growth, args.interval) <= 0:
        p.error('thresholds and intervals must be positive')
    raise SystemExit(run(args))
