#!/usr/bin/env python3
"""One locked, bounded serving experiment or benchmark-only suite."""
import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from inventory import ROOT, bind_identity, command, inventory, ple_identity


def bounded(args, log, seconds, env=None, guard=None):
    """Kill the complete child process group on deadline or guard failure."""
    with Path(log).open('w') as output:
        proc = subprocess.Popen(args, stdout=output, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
        deadline = time.monotonic() + seconds
        try:
            while proc.poll() is None:
                if guard:
                    guard()
                if time.monotonic() >= deadline:
                    raise TimeoutError('process deadline exceeded')
                time.sleep(.2)
            return proc.returncode
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=5)


def validate(name, report):
    if name in ('quality', 'longctx'):
        rows = report['results']
        key = 'pass' if name == 'quality' else 'needle_pass'
        return bool(rows) and report['failed'] == 0 and all(r[key] is True for r in rows)
    if name == 'decode':
        modes = report['results']
        return (len(modes) == 2 and {m['mode'] for m in modes} == {'thinking_on', 'thinking_off'}
                and all(m['tasks'] and all(t['samples'] and all(s['completion_tokens'] > 0
                    and s['seconds'] > 0 for s in t['samples']) for t in m['tasks'].values())
                    for m in modes))
    if name.startswith('agentic'):
        s = report['summary']
        return (s['invalid_tool_calls'] == 0 and s['late_recall_pass'] is True
                and s['turns'] > 0 and len(report['turns']) == s['turns'])
    return False


def suites(out, env, guard=None):
    summary = {'status': 'running', 'suites': []}
    path = out / 'suite_status.json'
    tasks = [('smoke', ['bash', str(ROOT / 'scripts/smoke.sh')], {})]
    tasks += [('quality', [sys.executable, str(ROOT / 'bench/quality.py')], {'EFFORT': '1'}),
              ('decode', [sys.executable, str(ROOT / 'bench/decode.py')], {'THINKING': 'both', 'N': env.get('N', '3')})]
    if env.get('QUICK') != '1':
        tasks += [('longctx', [sys.executable, str(ROOT / 'bench/longctx.py')], {'SIZES': env.get('SIZES', '8k,32k')})]
        for mode in ('off', 'on'):
            tasks.append(('agentic_' + mode, [sys.executable, str(ROOT / 'bench/agentic.py')],
                          {'THINKING': mode, 'TURNS': env.get('TURNS', '40')}))
    for name, args, overrides in tasks:
        print('START suite:', name, flush=True)
        child_env = {**env, **overrides, 'OUT': str(out / (name + '.json'))}
        row = {'name': name, 'status': 'error'}
        start = time.monotonic()
        try:
            rc = bounded(args, out / (name + '.log'), float(env.get('SUITE_TIMEOUT', '900')),
                         child_env, guard)
            row['exit_code'] = rc
            if name == 'smoke':
                passed = rc == 0
            else:
                passed = validate(name, json.loads((out / (name + '.json')).read_text()))
            row['status'] = 'pass' if rc == 0 and passed else 'fail'
        except (TimeoutError, RuntimeError):
            row['status'] = 'aborted'
            summary['suites'].append(row)
            summary['status'] = 'fail'
            path.write_text(json.dumps(summary, indent=2))
            raise
        except (OSError, ValueError, KeyError, TypeError) as exc:
            row['error'] = type(exc).__name__
        row['seconds'] = round(time.monotonic() - start, 3)
        summary['suites'].append(row)
        path.write_text(json.dumps(summary, indent=2))
        print('RESULT suite:', name, row['status'], flush=True)
    summary['status'] = 'pass' if all(x['status'] == 'pass' for x in summary['suites']) else 'fail'
    path.write_text(json.dumps(summary, indent=2))
    return 0 if summary['status'] == 'pass' else 1


def ready(base):
    try:
        with urllib.request.urlopen(base + '/health', timeout=3) as r:
            return r.status == 200
    except OSError:
        return False


SAFE_FLAGS = {'model_path', 'revision', 'served_model_name', 'context_length',
              'max_total_tokens', 'max_running_requests', 'mem_fraction_static',
              'chunked_prefill_size', 'mamba_radix_cache_strategy', 'mamba_track_interval',
              'max_mamba_cache_size', 'mamba_ssm_dtype', 'page_size', 'quantization',
              'speculative_algorithm', 'speculative_num_steps', 'speculative_eagle_topk',
              'speculative_num_draft_tokens', 'speculative_draft_model_quantization',
              'prefill_attention_backend', 'decode_attention_backend', 'enable_gdn_replayssm_spec',
              'disable_radix_cache', 'disable_prefill_cuda_graph', 'cuda_graph_max_bs_decode',
              'reasoning_parser', 'tool_call_parser', 'preferred_sampling_params',
              'fp4_gemm_backend', 'moe_runner_backend', 'kv_cache_dtype',
              'enable_linear_replayssm_spec', 'enable_gdn_replayssm_spec',
              'enable_linear_replayssm', 'enable_gdn_replayssm', 'max_total_num_tokens'}


def safe_server_info(server):
    args = server.get('server_args', server)
    return {k: v for k, v in args.items() if k in SAFE_FLAGS}


def effective(container, base, out):
    # Never dump inspect's Env or unfiltered /get_server_info (may contain API keys).
    info = json.loads(command(['docker', 'inspect', container]))[0]
    argv = info['Config']['Cmd']
    flags = {}
    for i, a in enumerate(argv):
        key = a.removeprefix('--').replace('-', '_')
        if a.startswith('--') and key in SAFE_FLAGS:
            flags[key] = argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith('--') else True
    result = {'image_id': info['Image'], 'launch_flags': flags}
    server = {}
    if base:
        with urllib.request.urlopen(base + '/server_info', timeout=10) as r:
            server = json.load(r)
    result['effective_flags'] = safe_server_info(server)
    result['server_version'] = server.get('version')
    package_code = ('import importlib.metadata as m,json; '
                    'names=["sglang","sglang-kernel","sgl-kernel","torch","triton","transformers",'
                    '"flashinfer-python","nvidia-modelopt"]; '
                    'print(json.dumps({d.metadata["Name"]:d.version for d in m.distributions() '
                    'if d.metadata["Name"].lower() in names}))')
    result['packages'] = json.loads(command(['docker', 'exec', container, 'python3', '-c', package_code], 20))
    result['source_commit'] = command(['docker', 'exec', container, 'sh', '-c',
        'git -c safe.directory=/sgl-workspace/sglang -C /sgl-workspace/sglang rev-parse HEAD 2>/dev/null || true']) or 'unavailable'
    (out / ('effective.json' if base else 'launch.json')).write_text(json.dumps(result, indent=2))


def run(out, env):
    container = env['CONTAINER']
    base = env.get('BASE', 'http://127.0.0.1:' + env.get('PORT', '30000')).rstrip('/')
    env = {**env, 'BASE': base, 'MODEL': env['RECIPE_MODEL'], 'PYTHONUNBUFFERED': '1'}
    initial = inventory(env['IMAGE'], container)
    (out / 'inventory_before.json').write_text(json.dumps(initial, indent=2))
    preserved = out / 'preserved'; preserved.mkdir()
    for folder in ('build', 'patches', 'scripts', 'bench'):
        dst = preserved / folder; dst.mkdir()
        for p in (ROOT / folder).glob('*'):
            if p.is_file() and p.suffix in ('.py', '.sh', '.txt'):
                shutil.copy2(p, dst / p.name)
    # Local tag retains the exact original image without exporting multi-GB layers.
    command(['docker', 'tag', initial['image_id'], 'qwen38-u0-preserved:' + env['TAG']])
    identity = ple_identity(env['SNAPSHOT'], env['PLE_DIR'], env['RECIPE_MODEL'], env['REVISION'])
    bind_identity(identity, ROOT / 'results/ple-identities')
    (out / 'ple_identity.json').write_text(json.dumps(identity, indent=2))
    env['IMAGE'] = initial['image_id']  # serve cannot accidentally follow a moved tag
    env['EXPERIMENT_IMAGE_ID'] = initial['image_id']
    env['EXPERIMENT_REVISION'] = env['REVISION']
    env['EXPERIMENT_PLE_DIR'] = env['PLE_DIR']
    # POST before occupying the GPU; do not start alongside any remaining occupant.
    request = urllib.request.Request(env.get('UNLOAD_URL', 'http://127.0.0.1:8080/api/models/unload'), method='POST')
    with urllib.request.urlopen(request, timeout=10) as response:
        response.read()
    # Stop only our prior recipe instance, before verifying that the GPU is idle.
    names = command(['docker', 'ps', '-a', '--format', '{{.Names}}']).splitlines()
    if container in names:
        command(['docker', 'stop', '-t', '30', container], 40)
    occupants = command(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'])
    if occupants:
        raise RuntimeError('GPU remains occupied after unload; refusing launch')
    watch = None
    done, trip = out / 'watch.done', out / 'watch.trip.json'
    owned = False
    boot_start = time.monotonic()
    result = {'status': 'error'}
    def guard():
        if trip.exists():
            raise RuntimeError('memory watchdog tripped')
        if watch is not None and watch.poll() is not None:
            raise RuntimeError('watchdog exited unexpectedly')
    try:
        watch_log = (out / 'watch-process.log').open('w')
        args = [sys.executable, str(ROOT / 'scripts/memwatch.py'), '--container', container,
                '--log', str(out / 'memory.jsonl'), '--done', str(done), '--trip', str(trip)]
        for key, flag in [('MEMWATCH_AVAILABLE', 'available'), ('MEMWATCH_FREE', 'free'),
                          ('MEMWATCH_FREE_GATE', 'free-gate'), ('MEMWATCH_SUSTAIN', 'sustain')]:
            if key in env:
                args += ['--' + flag, env[key]]
        watch = subprocess.Popen(args, stdout=watch_log, stderr=subprocess.STDOUT, start_new_session=True)
        watch_log.close()
        owned = True
        if bounded(['bash', str(ROOT / 'scripts/serve.sh')], out / 'serve.log', 90, env, guard):
            raise RuntimeError('serve failed; inspect serve.log')
        effective(container, None, out)  # versions survive even a failed warmup
        deadline = boot_start + float(env.get('BOOT_WAIT', '1500'))
        while not ready(base):
            guard()
            if time.monotonic() >= deadline:
                raise TimeoutError('boot deadline exceeded')
            state = command(['docker', 'inspect', '-f', '{{.State.Running}}', container])
            if state != 'true':
                raise RuntimeError('container died during startup')
            time.sleep(5)
        result['boot_seconds'] = round(time.monotonic() - boot_start, 2)
        print('RESULT boot:', result['boot_seconds'], flush=True)
        effective(container, base, out)
        boot_proc = subprocess.run(['docker', 'logs', '--tail', '2000', container],
            capture_output=True, text=True, timeout=15)
        boot_log = boot_proc.stdout + boot_proc.stderr
        (out / 'boot-facts.txt').write_text('\n'.join(line for line in boot_log.splitlines()
            if any(key in line for key in ('shards already on disk', 'KV Cache',
                'max_total_num_tokens=', 'CUDA graph', 'Load weight end'))))
        env['MODEL'] = env['SERVED_NAME']
        rc = suites(out, env, guard)
        result['status'] = 'pass' if rc == 0 else 'fail'
    finally:
        if owned:
            tail = subprocess.run(['docker', 'logs', '--tail', '200', container],
                                  capture_output=True, text=True, timeout=15)
            (out / 'server-tail.log').write_text(tail.stdout + tail.stderr)
            # U0 experiments own their server lifecycle; never leave an unguarded
            # or known-old server serving unattended after the benchmark ends.
            stopped = subprocess.run(['docker', 'stop', '-t', '30', container],
                                     capture_output=True, text=True, timeout=40)
            result['stop_exit_code'] = stopped.returncode
        done.touch()
        if watch:
            try:
                watch.wait(timeout=45)
            except subprocess.TimeoutExpired:
                watch.terminate(); watch.wait(timeout=5)
        result['watchdog_tripped'] = trip.exists()
        if trip.exists() or result.get('stop_exit_code', 1) != 0:
            result['status'] = 'fail'
        (out / 'experiment_status.json').write_text(json.dumps(result, indent=2))
    return 0 if result['status'] == 'pass' else 1


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt('experiment interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    env = dict(os.environ)
    tag = env.get('TAG', '')
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,99}', tag):
        raise SystemExit('set a unique TAG containing only letters/digits/._-')
    results = ROOT / 'results'; results.mkdir(exist_ok=True)
    with (results / '.bench.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('another experiment/benchmark holds results/.bench.lock')
        out = results / tag
        out.mkdir()  # refuse to overwrite evidence from any prior attempt
        if env.get('PROFILE') == 'u0':
            if env.get('SIZES', '8k,32k') != '8k,32k' or int(env.get('TURNS', '40')) > 40:
                raise SystemExit('U0 restricts baseline to 8k/32k and <=40 tool turns')
            env['MAX_PROMPT_TOKENS'] = '32768'
        try:
            if sys.argv[1:] == ['run']:
                return run(out, env)
            if sys.argv[1:] == ['suite']:
                return suites(out, env)
            raise ValueError('expected run or suite')
        except Exception as exc:
            (out / 'failure.json').write_text(json.dumps({'type': type(exc).__name__, 'message': str(exc)}, indent=2))
            print('RESULT experiment: FAIL', type(exc).__name__, str(exc), flush=True)
            return 1


if __name__ == '__main__':
    raise SystemExit(main())
