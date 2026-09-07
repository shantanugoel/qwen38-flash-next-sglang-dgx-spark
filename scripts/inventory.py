#!/usr/bin/env python3
"""Sanitized experiment inventory and sampled, read-only FP8 PLE identity check."""
import hashlib
import json
import re
import struct
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def command(args, timeout=10):
    p = subprocess.run(args, text=True, capture_output=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError(f'{args[0]} failed ({p.returncode})')
    return p.stdout.strip()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ple_identity(snapshot, directory, model, revision):
    """Validate every shard's ends and midpoint without loading the full table.

    Sampling is evidence, not a full checksum. Refuse a known identity mismatch
    even if samples coincide. The identity record belongs in results/, never
    alongside the user's existing backing file.
    """
    snapshot, directory = Path(snapshot), Path(directory)
    index = snapshot / 'model.safetensors.index.json'
    weights = json.loads(index.read_text())['weight_map']
    for filename in set(weights.values()):
        if not (snapshot / filename).is_file():
            raise RuntimeError('incomplete checkpoint: missing weight shard')
    shards = sorted((int(m.group(1)), key, value) for key, value in weights.items()
                    if (m := re.search(r'ngram_embedding\.shard_(\d+)\.weight$', key)))
    if not shards or [x[0] for x in shards] != list(range(len(shards))):
        raise RuntimeError('PLE shards missing or unsupported layout')
    layout = []
    headers = {}
    total = 0
    for _, key, filename in shards:
        if filename not in headers:
            with (snapshot / filename).open('rb') as f:
                size = struct.unpack('<Q', f.read(8))[0]
                if size > 64 * 1024 * 1024:
                    raise RuntimeError('unexpected safetensors header size')
                headers[filename] = (8 + size, json.loads(f.read(size)))
        start, header = headers[filename]
        tensor = header[key]
        if tensor['dtype'] != 'F8_E4M3':
            raise RuntimeError('PLE identity checker currently supports FP8 E4M3 only')
        a, b = tensor['data_offsets']
        layout.append((filename, start + a, b - a, total))
        total += b - a
    backing = directory / f'ple_table_{total}_{total}.bin'
    if not backing.is_file() or backing.stat().st_size != total:
        raise RuntimeError('expected existing PLE backing file missing or wrong size')
    digest = hashlib.sha256()
    with backing.open('rb') as dst:
        for filename, src_start, size, dst_start in layout:
            with (snapshot / filename).open('rb') as src:
                width = min(size, 4096)
                for offset in sorted({0, (size - width) // 2, size - width}):
                    src.seek(src_start + offset); a = src.read(width)
                    dst.seek(dst_start + offset); b = dst.read(width)
                    if len(a) != width or a != b:
                        raise RuntimeError('PLE sample mismatch; refusing reuse')
                    digest.update(a)
    st = backing.stat()
    return {'model': model, 'revision': revision, 'index_sha256': sha(index),
            'config_sha256': sha(snapshot / 'config.json'), 'backing': str(backing.resolve()),
            'device': st.st_dev, 'inode': st.st_ino, 'bytes': total,
            'shards': len(shards), 'sample_sha256': digest.hexdigest(),
            'validation': '3 byte windows per shard; not a full-file checksum'}


def bind_identity(record, registry):
    registry = Path(registry)
    registry.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(record['backing'].encode()).hexdigest()
    p = registry / (key + '.json')
    if p.exists():
        old = json.loads(p.read_text())
        for field in ('model', 'revision', 'index_sha256', 'config_sha256', 'bytes'):
            if old[field] != record[field]:
                raise RuntimeError('PLE directory already bound to a different checkpoint')
    else:
        with p.open('x') as f:
            json.dump(record, f, indent=2)


def inventory(image, container):
    im = json.loads(command(['docker', 'image', 'inspect', image]))[0]
    result = {'git_commit': command(['git', 'rev-parse', 'HEAD']),
              'git_status': command(['git', 'status', '--short']),
              'image_id': im['Id'], 'repo_digests': im.get('RepoDigests'),
              'image_created': im['Created'], 'architecture': im['Architecture'],
              'kernel': command(['uname', '-r']),
              'gpu': command(['nvidia-smi', '--query-gpu=name,driver_version,clocks.current.graphics,clocks.max.graphics', '--format=csv']),
              'gpu_processes': command(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv']),
              'meminfo': Path('/proc/meminfo').read_text(),
              'disk': command(['df', '-B1', str(ROOT)]),
              'sources': {str(p.relative_to(ROOT)): sha(p) for p in
                          [*sorted((ROOT / 'scripts').glob('*.*')),
                           *sorted((ROOT / 'bench').glob('*.py')),
                           *sorted((ROOT / 'patches').rglob('*.py')),
                           *sorted((ROOT / 'build').rglob('*.py'))] if p.is_file()}}
    return result
