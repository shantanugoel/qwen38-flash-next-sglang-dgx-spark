import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'bench'))
from experiment import bounded, suites, validate, safe_server_info, run as experiment_run
from inventory import bind_identity, ple_identity
from memwatch import Policy, run as watch_run
from client import chat, chat_stream


class WatchTests(unittest.TestCase):
    def sample(self, a=12, f=1, swap=0):
        return dict(available_gib=a, free_gib=f, swap_used_gib=swap)

    def test_reclaimable_cache_does_not_trip_free_floor(self):
        p = Policy()
        self.assertFalse(p.check(self.sample(a=30, f=.1), 0))
        self.assertFalse(p.check(self.sample(a=30, f=.1), 50))

    def test_sustained_pressure_and_recovery(self):
        p = Policy()
        self.assertFalse(p.check(self.sample(a=5), 0))
        self.assertFalse(p.check(self.sample(a=12), 10))
        self.assertFalse(p.check(self.sample(a=5), 20))
        self.assertFalse(p.check(self.sample(a=5), 34))
        self.assertTrue(p.check(self.sample(a=5), 35))

    def test_gated_free_swap_and_driver_errors(self):
        p = Policy()
        self.assertFalse(p.check(self.sample(a=8, f=.2), 0))
        self.assertTrue(p.check(self.sample(a=8, f=.2), 16))
        p = Policy()
        p.check(self.sample(), 0)
        p.check(self.sample(swap=1), 1)
        self.assertTrue(p.check(self.sample(swap=1), 17))
        self.assertTrue(Policy().check(self.sample(), 0, xid=True))
        p = Policy()
        self.assertFalse(p.check(self.sample(), 0, alloc=True))
        self.assertFalse(p.check(self.sample(), 10, alloc=False))
        self.assertFalse(p.check(self.sample(), 20, alloc=True))
        self.assertTrue(p.check(self.sample(), 35, alloc=True))

    def test_watchdog_stops_only_named_container_and_writes_evidence(self):
        import argparse
        with tempfile.TemporaryDirectory() as d:
            args = argparse.Namespace(available=6, free=.5, free_gate=10,
                sustain=15, swap_growth=.5, interval=.01, log=d+'/memory.jsonl',
                done=d+'/done', trip=d+'/trip', container='only-this-experiment')
            calls=[]
            def fake_run(cmd, **kwargs):
                calls.append(cmd)
                return subprocess.CompletedProcess(cmd, 0,
                    'NVRM: Xid 79' if cmd[0]=='journalctl' else 'test evidence', '')
            with patch('memwatch.memory', return_value=self.sample()), patch('memwatch.subprocess.run', side_effect=fake_run):
                self.assertEqual(watch_run(args),2)
            trip=json.loads(Path(args.trip).read_text())
            self.assertEqual(trip['reasons'], ['new NVIDIA Xid'])
            self.assertEqual(trip['journal_matches'], ['NVRM: Xid 79'])
            self.assertIn(['docker','stop','-t','30','only-this-experiment'],calls)
            self.assertTrue(Path(args.log+'.server-tail').exists())

    def test_oneshot_allocation_error_does_not_stop_watch_run(self):
        import argparse
        with tempfile.TemporaryDirectory() as d:
            args = argparse.Namespace(available=6, free=.5, free_gate=10,
                sustain=15, swap_growth=.5, interval=.01, log=d+'/memory.jsonl',
                done=d+'/done', trip=d+'/trip', container='only-this-experiment')
            n={'j':0}
            def fake_run(cmd, **kwargs):
                if cmd[0]=='journalctl':
                    n['j'] += 1
                    out='NV_ERR_NO_MEMORY' if n['j']==1 else ''
                    return subprocess.CompletedProcess(cmd, 0, out, '')
                return subprocess.CompletedProcess(cmd, 0, '', '')
            def fake_memory():
                if n['j']>=3:
                    Path(args.done).touch()
                return self.sample()
            with patch('memwatch.memory', side_effect=fake_memory), patch('memwatch.subprocess.run', side_effect=fake_run):
                self.assertEqual(watch_run(args),0)
            self.assertFalse(Path(args.trip).exists())



class HarnessTests(unittest.TestCase):
    def test_effective_config_supports_flat_and_nested_without_secrets(self):
        raw={'context_length':262144,'api_key':'DO-NOT-LOG','env':{'HF_TOKEN':'DO-NOT-LOG'}}
        self.assertEqual(safe_server_info(raw),{'context_length':262144})
        self.assertEqual(safe_server_info({'server_args':raw}),{'context_length':262144})

    def test_boot_timeout_stops_owned_container_without_running_suites(self):
        import io
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in ('build','patches','scripts','bench','output'):(root/name).mkdir()
            env={'CONTAINER':'test-only','IMAGE':'local','RECIPE_MODEL':'model',
                 'REVISION':'rev','SNAPSHOT':d,'PLE_DIR':d,'TAG':'test','BOOT_WAIT':'0.000001'}
            watcher=Mock();watcher.poll.return_value=None
            completed=subprocess.CompletedProcess([],0,'','')
            with patch('experiment.ROOT',root), patch('experiment.inventory',return_value={'image_id':'sha256:test'}), patch('experiment.command',return_value=''), patch('experiment.ple_identity',return_value={}), patch('experiment.bind_identity'), patch('experiment.urllib.request.urlopen',return_value=io.BytesIO()), patch('experiment.subprocess.Popen',return_value=watcher), patch('experiment.subprocess.run',return_value=completed) as calls, patch('experiment.bounded',return_value=0), patch('experiment.ready',return_value=False), patch('experiment.effective'), patch('experiment.suites') as suite:
                with self.assertRaises(TimeoutError):experiment_run(root/'output',env)
                suite.assert_not_called()
                self.assertIn(['docker','stop','-t','30','test-only'],[c.args[0] for c in calls.call_args_list])
            result=json.loads((root/'output/experiment_status.json').read_text())
            self.assertEqual(result['status'],'error')

    def test_shutdown_failure_cannot_report_success(self):
        import io
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in ('build','patches','scripts','bench','output'):(root/name).mkdir()
            env={'CONTAINER':'test-only','IMAGE':'local','RECIPE_MODEL':'model','SERVED_NAME':'served',
                 'REVISION':'rev','SNAPSHOT':d,'PLE_DIR':d,'TAG':'test'}
            watcher=Mock();watcher.poll.return_value=None
            def completed(cmd,**kw):
                return subprocess.CompletedProcess(cmd,1 if cmd[:2]==['docker','stop'] else 0,'','')
            with patch('experiment.ROOT',root), patch('experiment.inventory',return_value={'image_id':'sha256:test'}), patch('experiment.command',return_value=''), patch('experiment.ple_identity',return_value={}), patch('experiment.bind_identity'), patch('experiment.urllib.request.urlopen',return_value=io.BytesIO()), patch('experiment.subprocess.Popen',return_value=watcher), patch('experiment.subprocess.run',side_effect=completed), patch('experiment.bounded',return_value=0), patch('experiment.ready',return_value=True), patch('experiment.effective'), patch('experiment.suites',return_value=0):
                self.assertEqual(experiment_run(root/'output',env),1)
            result=json.loads((root/'output/experiment_status.json').read_text())
            self.assertEqual(result['status'],'fail')

    def test_semantic_failure_despite_exit_zero(self):
        self.assertFalse(validate('quality', {'failed': 0, 'results': [{'pass': False}]}))
        self.assertFalse(validate('longctx', {'failed': 0, 'results': []}))
        self.assertFalse(validate('agentic_off', {'summary': {'invalid_tool_calls': 1, 'late_recall_pass': True}}))

    def test_process_deadline(self):
        with tempfile.TemporaryDirectory() as d:
            begin = time.monotonic()
            with self.assertRaises(TimeoutError):
                bounded([sys.executable, '-c', 'import time; time.sleep(30)'], Path(d)/'log', .2)
            self.assertLess(time.monotonic() - begin, 4)

    def test_guard_aborts_process(self):
        def guard():
            raise RuntimeError('watchdog failed')
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):
                bounded([sys.executable, '-c', 'import time; time.sleep(30)'], Path(d)/'log', 10, guard=guard)

    def test_process_deadline_terminates_descendants(self):
        with tempfile.TemporaryDirectory() as d:
            pidfile=Path(d)/'pid'
            code=('import subprocess,sys,time; from pathlib import Path; '
                  'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"]); '
                  'Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(30)')
            with self.assertRaises(TimeoutError):
                bounded([sys.executable,'-c',code,str(pidfile)],Path(d)/'log',.4)
            pid=int(pidfile.read_text())
            stat=Path(f'/proc/{pid}/stat')
            time.sleep(.1)
            self.assertTrue(not stat.exists() or stat.read_text().split()[2]=='Z')

    def test_suite_failure_and_missing_report_propagate(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            def fake(args, log, seconds, env, guard):
                if 'quality.py' in args[-1]:
                    Path(env['OUT']).write_text(json.dumps({'failed': 1, 'results': [{'pass': False}]}))
                return 0
            with patch('experiment.bounded', side_effect=fake):
                self.assertEqual(suites(out, {'QUICK': '1'}), 1)
            result = json.loads((out/'suite_status.json').read_text())
            self.assertEqual([r['status'] for r in result['suites']], ['pass', 'fail', 'error'])

    def test_second_process_cannot_take_lock(self):
        import fcntl
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'lock'
            with path.open('w') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                code = 'import fcntl,sys; f=open(sys.argv[1],"w"); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)'
                p = subprocess.run([sys.executable, '-c', code, str(path)], capture_output=True)
                self.assertNotEqual(p.returncode, 0)

    def test_ple_identity_samples_and_rejects_conflicting_revision(self):
        import struct
        with tempfile.TemporaryDirectory() as d:
            p = Path(d); snapshot=p/'snapshot'; snapshot.mkdir(); ple=p/'ple';ple.mkdir()
            tensor='model.ngram_embedding.shard_0.weight'
            header=json.dumps({tensor:{'dtype':'F8_E4M3','shape':[10], 'data_offsets':[0,10]}}).encode()
            (snapshot/'weight.safetensors').write_bytes(struct.pack('<Q',len(header))+header+b'abcdefghij')
            (snapshot/'config.json').write_text('{}')
            (snapshot/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{tensor:'weight.safetensors'}}))
            backing=ple/'ple_table_10_10.bin';backing.write_bytes(b'abcdefghij')
            record=ple_identity(snapshot,ple,'model','revision-a')
            bind_identity(record,p/'registry');bind_identity(record,p/'registry')
            with self.assertRaises(RuntimeError):
                bind_identity({**record,'revision':'revision-b'},p/'registry')
            backing.write_bytes(b'abcdefghiX')
            with self.assertRaises(RuntimeError):
                ple_identity(snapshot,ple,'model','revision-a')


class RequestDeadlineTests(unittest.TestCase):
    def test_oversize_prompt_is_rejected_before_generation(self):
        import io
        response=io.BytesIO(json.dumps({'count':40000}).encode())
        with patch.dict(os.environ, {'MAX_PROMPT_TOKENS':'32768','REQUEST_TIMEOUT':'1'}), patch('client.urllib.request.urlopen',return_value=response) as request:
            with self.assertRaises(ValueError):chat([{'role':'user','content':'test'}])
            self.assertEqual(request.call_count,1)
            self.assertTrue(request.call_args.args[0].full_url.endswith('/tokenize'))

    def test_dribbling_stream_has_total_deadline(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args): pass
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                self.send_response(200); self.send_header('Content-Type','text/event-stream'); self.end_headers()
                try:
                    for _ in range(40):
                        self.wfile.write(b'data: {"choices":[]}\n\n'); self.wfile.flush();time.sleep(.03)
                except (BrokenPipeError,ConnectionResetError):pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with patch.dict(os.environ, {'BASE':f'http://127.0.0.1:{server.server_port}','REQUEST_TIMEOUT':'.2','MAX_PROMPT_TOKENS':''}):
                for fn in (chat,chat_stream):
                    with self.assertRaises(TimeoutError):fn([{'role':'user','content':'test'}])
        finally:server.shutdown();server.server_close()


class QsaPatchTests(unittest.TestCase):
    def load_patcher(self):
        spec = importlib.util.spec_from_file_location(
            'qsa_sm121_kda', ROOT / 'patches/qsa_sm121_kda.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_kda_route_rejects_widened_gate_and_is_idempotent(self):
        mod = self.load_patcher()
        src = ('def _resolve_flash_attn_varlen_func():\n'
               '    try:\n'
               '        from flash_attn import flash_attn_varlen_func\n'
               '        return flash_attn_varlen_func\n'
               '    except ImportError:\n'
               '        pass\n')
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'backend.py'
            path.write_text(src)
            self.assertEqual(mod.main(str(path)), 0)
            patched = path.read_text()
            self.assertIn('kda_kernels.qwen38_qsa_sm121', patched)
            self.assertIn('qsa.sm121_varlen', patched)
            self.assertNotIn('is_sm100_supported() or is_sm120_supported()', patched)
            self.assertEqual(mod.main(str(path)), 0)
            wide = Path(d) / 'wide.py'
            wide.write_text('if not (is_sm100_supported() or is_sm120_supported()):\n' + src)
            self.assertEqual(mod.main(str(wide)), 1)

    def test_triton_route_rejects_kda_and_widened_gate(self):
        spec = importlib.util.spec_from_file_location(
            'qsa_sm121_triton', ROOT / 'patches/qsa_sm121_triton.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        src = ('def _resolve_flash_attn_varlen_func():\n'
               '    try:\n'
               '        from flash_attn import flash_attn_varlen_func\n'
               '        return flash_attn_varlen_func\n'
               '    except ImportError:\n'
               '        pass\n')
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'backend.py'
            path.write_text(src)
            self.assertEqual(mod.main(str(path)), 0)
            patched = path.read_text()
            self.assertIn('_qsa_sm121_triton_varlen', patched)
            self.assertIn('qsa.sm121_varlen', patched)
            self.assertNotIn('kda_kernels.qwen38_qsa_sm121', patched)
            self.assertEqual(mod.main(str(path)), 0)
            kda = Path(d) / 'kda.py'
            kda.write_text('from sglang.kernels.kda_kernels.qwen38_qsa_sm121 import x\n' + src)
            self.assertEqual(mod.main(str(kda)), 1)
            wide = Path(d) / 'wide.py'
            wide.write_text('if not (is_sm100_supported() or is_sm120_supported()):\n' + src)
            self.assertEqual(mod.main(str(wide)), 1)

    def test_triton_replaces_bundled_kda_route(self):
        spec = importlib.util.spec_from_file_location(
            'qsa_sm121_triton', ROOT / 'patches/qsa_sm121_triton.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        src = ('from sglang.srt.utils import is_sm121\n'
               + mod.KDA_BUNDLED
               + '    try:\n        from flash_attn import flash_attn_varlen_func\n')
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'backend.py'
            path.write_text(src)
            self.assertEqual(mod.main(str(path)), 0)
            patched = path.read_text()
            self.assertIn('_qsa_sm121_triton_varlen', patched)
            self.assertIn('qsa.sm121_varlen', patched)
            self.assertNotIn('qwen38_qsa_sm121_varlen', patched)
            self.assertEqual(mod.main(str(path)), 0)

    def test_u3_profile_adds_soak(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            def fake_bounded(args, log, seconds, env=None, guard=None):
                name = Path(env['OUT']).stem
                if name == 'smoke':
                    return 0
                payload = {'results': [{'pass': True, 'needle_pass': True}], 'failed': 0}
                if name == 'decode':
                    payload = {'results': [
                        {'mode': 'thinking_on', 'tasks': {'c': {'samples': [{'completion_tokens': 1, 'seconds': 1}]}}},
                        {'mode': 'thinking_off', 'tasks': {'c': {'samples': [{'completion_tokens': 1, 'seconds': 1}]}}},
                    ]}
                elif name == 'soak':
                    payload = {'results': [{'pass': True}], 'failed': 0,
                               'summary': {'ok': 1, 'errors': 0, 'requests': 1,
                                           'seconds': 1.0, 'target_seconds': 1}}
                Path(env['OUT']).write_text(json.dumps(payload))
                return 0
            with patch('experiment.bounded', side_effect=fake_bounded):
                self.assertEqual(suites(out, {
                    'QUICK': '1', 'SUITE_TIMEOUT': '1', 'SOAK_SECONDS': '1',
                }), 0)
            names = [r['name'] for r in json.loads((out / 'suite_status.json').read_text())['suites']]
            self.assertEqual(names, ['smoke', 'quality', 'decode', 'soak'])

    def test_only_runs_named_suites(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            with patch('experiment.bounded', return_value=0):
                self.assertEqual(suites(out, {'ONLY': 'smoke', 'SUITE_TIMEOUT': '1'}), 0)
            result = json.loads((out / 'suite_status.json').read_text())
            self.assertEqual([r['name'] for r in result['suites']], ['smoke'])
            self.assertEqual(result['status'], 'pass')
            with self.assertRaises(ValueError):
                suites(out, {'ONLY': 'no-such-suite'})

    def test_u2_profile_adds_ple_spec_and_gsm8k(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            def fake_bounded(args, log, seconds, env=None, guard=None):
                name = Path(env['OUT']).stem
                if name == 'smoke':
                    return 0
                payload = {'results': [{'pass': True, 'needle_pass': True}], 'failed': 0}
                if name == 'decode':
                    payload = {'results': [
                        {'mode': 'thinking_on', 'tasks': {'c': {'samples': [{'completion_tokens': 1, 'seconds': 1}]}}},
                        {'mode': 'thinking_off', 'tasks': {'c': {'samples': [{'completion_tokens': 1, 'seconds': 1}]}}},
                    ]}
                elif name == 'gsm8k':
                    payload = {'summary': {'n': 2, 'accuracy': 50.0},
                               'results': [{'i': 0, 'pass': True}, {'i': 1, 'pass': False}]}
                Path(env['OUT']).write_text(json.dumps(payload))
                return 0
            with patch('experiment.bounded', side_effect=fake_bounded):
                self.assertEqual(suites(out, {
                    'PROFILE': 'u2', 'GSM8K_N': '200', 'QUICK': '1', 'SUITE_TIMEOUT': '1',
                }), 0)
            names = [r['name'] for r in json.loads((out / 'suite_status.json').read_text())['suites']]
            self.assertEqual(names, ['smoke', 'quality', 'decode', 'ple_spec', 'gsm8k'])


class ReplaySsmPleCommitTests(unittest.TestCase):
    def load_patcher(self):
        spec = importlib.util.spec_from_file_location(
            'replayssm_ple_commit', ROOT / 'patches/replayssm_ple_commit.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_patch_commits_ple_on_fold_and_ring_returns(self):
        mod = self.load_patcher()
        src = (
            'import logging\nlogger = logging.getLogger(__name__)\n'
            + mod.FOLD_ANCHOR + '\n# other branch\n' + mod.RING_ANCHOR
        )
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'spec_utils.py'
            path.write_text(src)
            self.assertEqual(mod.main(str(path)), 0)
            patched = path.read_text()
            self.assertEqual(patched.count('attn_backend._update_ple_state_after_mtp_verify('), 2)
            self.assertIn('ReplaySSM verify: committing PLE n-gram/short-conv state', patched)
            fold_idx = patched.index('commit_gdn_replayssm_fold_after_verify(')
            fold_return = patched.index('\n        return\n', fold_idx)
            self.assertIn('_update_ple_state_after_mtp_verify', patched[fold_idx:fold_return])
            self.assertNotIn('_linearize_chain', patched)
            self.assertEqual(mod.main(str(path)), 0)

    def test_patch_refuses_ngram_leak_and_missing_anchor(self):
        mod = self.load_patcher()
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / 'ngram.py'
            bad.write_text(mod.FOLD_ANCHOR + mod.RING_ANCHOR + '\nNgramCorpus = 1\n')
            self.assertEqual(mod.main(str(bad)), 1)
            missing = Path(d) / 'missing.py'
            missing.write_text('return\n')
            self.assertEqual(mod.main(str(missing)), 1)

    def test_accept_and_reject_select_last_correct_ple_history(self):
        prompt = [1, 2, 3]
        # per-draft intermediate histories written during TARGET_VERIFY
        steps = [
            [1, 2, 10],
            [2, 10, 11],
            [10, 11, 12],
            [11, 12, 13],
        ]

        def commit(last_correct_step):
            return steps[last_correct_step]

        self.assertEqual(commit(3), [11, 12, 13])  # all drafts accepted
        self.assertEqual(commit(0), [1, 2, 10])    # all drafts rejected, bonus only
        self.assertEqual(prompt, [1, 2, 3])        # freeze bug leaves prompt n-grams


class NativePleAndQsaDropTests(unittest.TestCase):
    def test_mmap_skips_native_file_backend(self):
        spec = importlib.util.spec_from_file_location(
            'ple_mmap', ROOT / 'patches/ple_mmap.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'qwen4_exp.py'
            path.write_text('def allocate_ple_host_table():\n    return None\n')
            self.assertEqual(mod.main(str(path)), 0)
            self.assertNotIn('_alloc_ple_table', path.read_text())

    def test_reuse_applies_without_mmap_helper(self):
        spec = importlib.util.spec_from_file_location(
            'ple_reuse', ROOT / 'patches/ple_reuse.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        src = ('def allocate_ple_host_table():\n    return None\n'
               + 'class Qwen4ExpPinnedHostEmbedding(VocabParallelEmbedding):\n'
               + '    pass\n'
               + mod.ORIG
               + '        self.logged_params = loaded_params\n')
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'qwen4_exp.py'
            path.write_text(src)
            self.assertEqual(mod.main(str(path)), 0)
            self.assertIn('_ple_shard_matches', path.read_text())

    def test_file_compat_prefers_recipe_backing_name(self):
        spec = importlib.util.spec_from_file_location(
            'ple_file_compat', ROOT / 'patches/ple_file_compat.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        src = 'def allocate_ple_host_table():\n    pass\n' + mod.ORIG
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'qwen4_exp_ple_table.py'
            path.write_text(src)
            self.assertEqual(mod.main(str(path)), 0)
            patched = path.read_text()
            self.assertIn('reusing recipe backing file', patched)
            self.assertIn('ple_table_%d_%d.bin', patched)
            self.assertEqual(mod.main(str(path)), 0)

    def test_qsa_drop_is_noop_when_is_sm121_is_only_the_kernel_gate(self):
        spec = importlib.util.spec_from_file_location(
            'qsa_drop', ROOT / 'patches/qsa_drop_sm121_sdpa.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        src = '    if is_sm121():\n        return qwen38_qsa_sm121_varlen\n'
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'backend.py'
            path.write_text(src)
            self.assertEqual(mod.main(str(path)), 0)
            self.assertEqual(path.read_text(), src)

    def test_blank_env_does_not_block_u3_file_backend(self):
        from experiment import default_if_blank
        env = {'PLE_OFFLOAD_BACKEND': ''}
        default_if_blank(env, 'PLE_OFFLOAD_BACKEND', 'file')
        self.assertEqual(env['PLE_OFFLOAD_BACKEND'], 'file')
        default_if_blank(env, 'PLE_OFFLOAD_BACKEND', 'pinned')
        self.assertEqual(env['PLE_OFFLOAD_BACKEND'], 'file')


if __name__ == '__main__': unittest.main()
