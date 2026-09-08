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


def default_if_blank(env, key, value):
    """Set env[key] when missing or whitespace-only (exported empty strings count)."""
    if not str(env.get(key) or '').strip():
        env[key] = value


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
    if name == 'effort_probe':
        s = report['summary']
        return s.get('n', 0) > 0 and len(report.get('results') or []) == s['n']
    if name == 'ple_spec':
        rows = report['results']
        return bool(rows) and report['failed'] == 0 and all(r['pass'] is True for r in rows)
    if name == 'gsm8k':
        s = report['summary']
        return (s.get('n', 0) > 0 and 'accuracy' in s
                and len(report.get('results') or []) == s['n'])
    if name == 'prefill':
        rows = report['results']
        return (bool(rows) and report['failed'] == 0
                and all(r.get('ttft_s', 0) > 0 and r.get('prompt_tokens', 0) > 0
                        and r.get('needle_pass') is True for r in rows))
    if name == 'streams':
        rows = report['results']
        wanted = {1, 2, 4}
        got = {r.get('concurrency') for r in rows}
        return (bool(rows) and report['failed'] == 0 and wanted.issubset(got)
                and all(r.get('ok') is True and r.get('median_aggregate_tok_s', 0) > 0
                        for r in rows))
    if name == 'mixedload':
        rows = report['results']
        return (bool(rows) and report['failed'] == 0
                and all(r.get('ok') is True and r.get('prefill_ttft_s', 0) > 0
                        and r.get('prefill_prompt_tokens', 0) > 0
                        and r.get('prefill_needle_pass') is True
                        and r.get('decode_streams', 0) >= 2 for r in rows))
    if name in ('soak', 'growsoak'):
        s = report['summary']
        target = float(s.get('target_seconds') or 0)
        extra = True if name != 'growsoak' else s.get('recall_fail', 1) == 0
        return (extra and s.get('errors', 1) == 0 and s.get('ok', 0) > 0
                and s.get('requests', 0) == len(report.get('results') or [])
                and (target <= 0 or float(s.get('seconds') or 0) >= 0.9 * target))
    return False


def suites(out, env, guard=None):
    summary = {'status': 'running', 'suites': []}
    path = out / 'suite_status.json'
    tasks = [('smoke', ['bash', str(ROOT / 'scripts/smoke.sh')], {})]
    if env.get('PREFILL_BENCH') == '1':
        tasks += [('prefill', [sys.executable, str(ROOT / 'bench/prefill.py')],
                   {'SIZES': env.get('SIZES', '8k,32k'),
                    'N': env.get('PREFILL_N', env.get('N', '3'))})]
    tasks += [('quality', [sys.executable, str(ROOT / 'bench/quality.py')], {'EFFORT': '1'}),
              ('decode', [sys.executable, str(ROOT / 'bench/decode.py')], {'THINKING': 'both', 'N': env.get('N', '3')})]
    if env.get('EFFORT_PROBE') == '1':
        # Descriptive: how stable is the one quality case that flips between
        # boots of the same accepted configuration? Never gates a decision.
        tasks += [('effort_probe', [sys.executable, str(ROOT / 'bench/effort_probe.py')],
                   {'N': env.get('EFFORT_PROBE_N', '20'),
                    'THINKING': env.get('EFFORT_PROBE_THINKING', 'off')})]
    if env.get('STREAMS') == '1':
        tasks += [('streams', [sys.executable, str(ROOT / 'bench/streams.py')],
                   {'N': env.get('STREAMS_N', env.get('N', '3')),
                    'CONCURRENCIES': env.get('CONCURRENCIES', '1,2,4')})]
    if env.get('MIXEDLOAD') == '1':
        tasks += [('mixedload', [sys.executable, str(ROOT / 'bench/mixedload.py')],
                   {'N': env.get('MIXEDLOAD_N', env.get('N', '3')),
                    'MIXEDLOAD_TOKENS': env.get('MIXEDLOAD_TOKENS', '64000'),
                    'REQUEST_TIMEOUT': env.get('MIXEDLOAD_REQUEST_TIMEOUT', '1800')})]
    if env.get('QUICK') != '1':
        tasks += [('longctx', [sys.executable, str(ROOT / 'bench/longctx.py')], {'SIZES': env.get('SIZES', '8k,32k')})]
        for mode in ('off', 'on'):
            tasks.append(('agentic_' + mode, [sys.executable, str(ROOT / 'bench/agentic.py')],
                          {'THINKING': mode, 'TURNS': env.get('TURNS', '40')}))
    if env.get('PROFILE') == 'u2':
        if env.get('SPEC', 'nextn') != 'off':
            tasks += [('ple_spec', [sys.executable, str(ROOT / 'bench/ple_spec.py')], {})]
        if env.get('GSM8K_N'):
            tasks += [('gsm8k', [sys.executable, str(ROOT / 'bench/gsm8k.py')],
                       {'N': env['GSM8K_N'], 'THINKING': env.get('GSM8K_THINKING', 'off')})]
    if env.get('SOAK_SECONDS'):
        tasks += [('soak', [sys.executable, str(ROOT / 'bench/soak.py')],
                   {'SOAK_SECONDS': env['SOAK_SECONDS']})]
    if env.get('GROWSOAK') == '1':
        tasks += [('growsoak', [sys.executable, str(ROOT / 'bench/growsoak.py')],
                   {'SOAK_SECONDS': env.get('SOAK_SECONDS', '3600')})]
    if only := env.get('ONLY'):
        wanted = {name.strip() for name in only.split(',') if name.strip()}
        tasks = [t for t in tasks if t[0] in wanted]
        if not tasks:
            raise ValueError('ONLY matched no suites: ' + only)
    for name, args, overrides in tasks:
        print('START suite:', name, flush=True)
        child_env = {**env, **overrides, 'OUT': str(out / (name + '.json'))}
        row = {'name': name, 'status': 'error'}
        start = time.monotonic()
        try:
            timeout = float(env.get('SUITE_TIMEOUT', '900'))
            if name == 'gsm8k':
                timeout = float(env.get('GSM8K_TIMEOUT', str(timeout)))
            if name in ('soak', 'growsoak'):
                timeout = float(env.get('SOAK_SECONDS', '3600')) + 300
            if name == 'mixedload':
                timeout = float(env.get('MIXEDLOAD_TIMEOUT', str(max(timeout, 1800))))
            rc = bounded(args, out / (name + '.log'), timeout, child_env, guard)
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
              'speculative_token_map',
              'prefill_attention_backend', 'decode_attention_backend', 'enable_gdn_replayssm_spec',
              'disable_radix_cache', 'disable_prefill_cuda_graph', 'cuda_graph_max_bs_decode',
              'cuda_graph_bs_decode',
              'reasoning_parser', 'tool_call_parser', 'preferred_sampling_params',
              'fp4_gemm_backend', 'moe_runner_backend', 'kv_cache_dtype',
              'enable_linear_replayssm_spec', 'enable_gdn_replayssm_spec',
              'enable_linear_replayssm', 'enable_gdn_replayssm', 'max_total_num_tokens',
              'ple_offload_embedding', 'ple_offload_backend', 'ple_offload_dir'}


LIST_FLAGS = {'cuda_graph_bs_decode'}


def launch_flags_from_argv(argv):
    """Parse SAFE_FLAGS from a docker Cmd list. List-valued flags keep all args."""
    flags = {}
    i = 0
    while i < len(argv):
        a = argv[i]
        key = a.removeprefix('--').replace('-', '_')
        if a.startswith('--') and key in SAFE_FLAGS:
            vals = []
            j = i + 1
            while j < len(argv) and not str(argv[j]).startswith('--'):
                vals.append(argv[j])
                j += 1
            if not vals:
                flags[key] = True
            elif key in LIST_FLAGS:
                flags[key] = vals
            else:
                flags[key] = vals[0]
            i = j
            continue
        i += 1
    return flags


def safe_server_info(server):
    args = server.get('server_args', server)
    return {k: v for k, v in args.items() if k in SAFE_FLAGS}


def effective(container, base, out):
    # Never dump inspect's Env or unfiltered /get_server_info (may contain API keys).
    info = json.loads(command(['docker', 'inspect', container]))[0]
    argv = info['Config']['Cmd']
    flags = launch_flags_from_argv(argv)
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
        shutil.copytree(ROOT / folder, preserved / folder,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.log'),
                        dirs_exist_ok=True)
    # Local tag retains the exact original image without exporting multi-GB layers.
    command(['docker', 'tag', initial['image_id'], 'qwen38-u0-preserved:' + env['TAG']])
    # PLE_FIRST_BUILD=1 is for a checkpoint whose backing table does not exist
    # yet. The directory is still bound to this checkpoint before the build, so
    # a second checkpoint cannot land in the same file, and the table is sampled
    # after the boot instead.
    first_build = env.get('PLE_FIRST_BUILD') == '1'
    identity = ple_identity(env['SNAPSHOT'], env['PLE_DIR'], env['RECIPE_MODEL'],
                            env['REVISION'], require_backing=not first_build)
    bind_identity(identity, ROOT / 'results/ple-identities')
    (out / 'ple_identity.json').write_text(json.dumps(identity, indent=2))
    if first_build and identity.get('state') == 'present':
        raise RuntimeError('PLE_FIRST_BUILD is set but the backing file already exists')
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
        if env.get('QSA_KERNEL_CHECK') == '1':
            print('START qsa kernel check', flush=True)
            if bounded(['bash', str(ROOT / 'scripts/bench_qsa_kernels.sh')],
                       out / 'qsa_kernel.log', float(env.get('KERNEL_TIMEOUT', '600')), env, guard):
                raise RuntimeError('QSA SM121 kernel check failed; inspect qsa_kernel.log')
            print('RESULT qsa kernel check: pass', flush=True)
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
        if first_build:
            # The table exists now; sample it against the checkpoint and record
            # the identity a later reuse boot will be checked against.
            built = ple_identity(env['SNAPSHOT'], env['PLE_DIR'], env['RECIPE_MODEL'],
                                 env['REVISION'])
            (out / 'ple_identity_built.json').write_text(json.dumps(built, indent=2))
            print('RESULT ple build:', built['bytes'], built['sample_sha256'], flush=True)
        effective(container, base, out)
        boot_proc = subprocess.run(['docker', 'logs', '--tail', '2000', container],
            capture_output=True, text=True, timeout=15)
        boot_log = boot_proc.stdout + boot_proc.stderr
        facts = '\n'.join(line for line in boot_log.splitlines()
            if any(key in line for key in ('shards already on disk', 'KV Cache',
                'max_total_num_tokens=', 'CUDA graph', 'Load weight end',
                'Capture target', 'Capture draft', 'num_tokens_per_req=',
                'KDA Qwen3.8 QSA', 'Using the Codex/Kimi',
                'Triton SM121 QSA', 'committing PLE n-gram',
                'file-backed mmap', 'reusing recipe backing file',
                'WILLNEED prefetch', 'RSS trimmer', 'resident set capped',
                'trimmed resident', 'speculative-token-map')))
        (out / 'boot-facts.txt').write_text(facts)
        token_map = env.get('SPECULATIVE_TOKEN_MAP', '').strip()
        if token_map:
            flags = json.loads((out / 'effective.json').read_text()).get('launch_flags', {})
            if flags.get('speculative_token_map') != '/speculative-token-map.pt':
                raise RuntimeError('SPECULATIVE_TOKEN_MAP was set but launch flags omitted it')
            report = Path(token_map).with_suffix('.report.json')
            if not report.is_file():
                report = Path(token_map).with_name(Path(token_map).stem + '.report.json')
            if report.is_file():
                shutil.copy2(report, out / 'draft_vocab.report.json')
        if env.get('QSA_KERNEL_CHECK') == '1':
            if 'KDA Qwen3.8 QSA' in facts or 'Using the Codex/Kimi' in facts:
                raise RuntimeError('KDA SM121 kernel selected; Triton-only serving is required')
            if 'Triton SM121 QSA' not in facts:
                raise RuntimeError('Triton SM121 QSA route did not log at boot')
        prefetch = str(env.get('SGLANG_QWEN4_PLE_FILE_PREFETCH', '0')).strip().lower()
        if prefetch not in ('0', '', 'false', 'no') and 'WILLNEED prefetch' not in facts:
            raise RuntimeError('PLE file prefetch did not log at boot')
        try:
            rss_budget = float(env.get('SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB', '0') or 0)
        except ValueError:
            rss_budget = 0.0
        if rss_budget > 0 and 'resident set capped' not in facts:
            raise RuntimeError('PLE RSS trimmer did not log at boot')
        env['MODEL'] = env['SERVED_NAME']
        rc = suites(out, env, guard)
        result['status'] = 'pass' if rc == 0 else 'fail'
        try:
            with urllib.request.urlopen(base + '/metrics', timeout=10) as response:
                metrics_txt = response.read().decode('utf-8', 'replace')
            metrics = {}
            for line in metrics_txt.splitlines():
                if line.startswith('#') or ' ' not in line:
                    continue
                key, _, val = line.rpartition(' ')
                metric = key.split('{', 1)[0]
                if metric in ('sglang:spec_accept_length', 'sglang:cache_hit_rate'):
                    metrics[metric] = float(val)
            result['metrics'] = metrics
        except (OSError, ValueError):
            result['metrics'] = {}
        if env.get('PROFILE') in ('u2', 'u3') and env.get('SPEC', 'nextn') != 'off':
            # First verify is in boot-facts. A later --tail of a long soak can
            # scroll that line off the log window; do not treat that as missing.
            if 'committing PLE n-gram/short-conv state' not in facts:
                served_proc = subprocess.run(['docker', 'logs', '--tail', '8000', container],
                                             capture_output=True, text=True, timeout=20)
                served = served_proc.stdout + served_proc.stderr
                if 'committing PLE n-gram/short-conv state' not in served:
                    raise RuntimeError(
                        'ReplaySSM PLE commit did not log; patch is not on the serving path')
    finally:
        if owned:
            tail = subprocess.run(['docker', 'logs', '--tail', '200', container],
                                  capture_output=True, text=True, timeout=15)
            (out / 'server-tail.log').write_text(tail.stdout + tail.stderr)
            trim = subprocess.run(['docker', 'logs', '--tail', '20000', container],
                                  capture_output=True, text=True, timeout=20)
            served = trim.stdout + trim.stderr
            (out / 'rss-trim.log').write_text('\n'.join(
                line for line in served.splitlines()
                if 'resident set' in line or 'trimmed resident' in line))
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
        elif env.get('PROFILE') == 'u1':
            env.setdefault('TURNS', '120')
            env.setdefault('SIZES', '8k,32k')
            env.setdefault('SUITE_TIMEOUT', '1800')
            env.setdefault('REQUEST_TIMEOUT', '900')
            env.setdefault('QSA_KERNEL_CHECK', '1')
        elif env.get('PROFILE') == 'u2':
            env.setdefault('TURNS', '120')
            env.setdefault('SIZES', '8k,32k')
            env.setdefault('SUITE_TIMEOUT', '1800')
            env.setdefault('REQUEST_TIMEOUT', '900')
            env.setdefault('GSM8K_N', '200')
            env.setdefault('GSM8K_TIMEOUT', '7200')
            if 'NGRAM' in env.get('EXTRA_ARGS', '').upper():
                raise SystemExit('U2 isolates the PLE commit; do not enable NGRAM')
        elif env.get('PROFILE') == 'u3':
            env.setdefault('TURNS', '120')
            env.setdefault('SIZES', '8k,32k')
            env.setdefault('SUITE_TIMEOUT', '1800')
            env.setdefault('REQUEST_TIMEOUT', '900')
            env.setdefault('QSA_KERNEL_CHECK', '1')
            env.setdefault('SOAK_SECONDS', '3600')
            default_if_blank(env, 'PLE_OFFLOAD_BACKEND', 'file')
            env.setdefault('SGLANG_QWEN4_PLE_FILE_PREFETCH', '0')
            env.setdefault('SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB', '0')
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
