#!/usr/bin/env python3
"""Build a reduced NEXTN draft-token map for SGLang --speculative-token-map.

SGLang NEXTN is an EAGLE alias. The worker clones the target lm_head, keeps
only these rows for drafting, argmaxes in the reduced space, then remaps with
hot_token_id[topk] so draft token IDs stay in the target vocabulary. Target
verify and sampling are unchanged; tokens outside the subset can still be
emitted when a draft is rejected. Do not enable
--speculative-use-rejection-sampling with a reduced map (upstream raises).

The frequency ranking uses a code/multilingual/tool corpus that is not the
quality/decode/agentic/longctx/gsm8k eval files. Those files are held-out
coverage only. If the corpus yields fewer than --size distinct IDs, remaining
rows are filled in tokenizer ID order and counted as fill, not as measured
traffic. Bandwidth savings are not resident-memory savings: the sliced head is
an extra clone.

Writes a 1-D int64 torch tensor that load_token_map() can torch.load.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

HELD_OUT_NAMES = {
    'quality.py', 'decode.py', 'agentic.py', 'longctx.py', 'gsm8k.py',
    'prefill.py', 'soak.py', 'growsoak.py', 'ple_spec.py',
}


def load_tokenizer(snapshot: Path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(snapshot), trust_remote_code=True)


def vocab_size_of(snapshot: Path, tokenizer) -> int:
    model_cfg = snapshot / 'config.json'
    size = int(getattr(tokenizer, 'vocab_size', 0) or 0)
    try:
        size = max(size, len(tokenizer))
    except TypeError:
        pass
    if model_cfg.is_file():
        cfg = json.loads(model_cfg.read_text())
        size = max(size, int((cfg.get('text_config') or {}).get('vocab_size') or 0))
        size = max(size, int(cfg.get('vocab_size') or 0))
    if size <= 0:
        raise SystemExit('could not determine vocabulary size')
    return size


def special_ids(snapshot: Path, tokenizer, vocab_size: int) -> list[int]:
    ids = set()
    cfg_path = snapshot / 'tokenizer_config.json'
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text())
        for key in cfg.get('added_tokens_decoder') or {}:
            ids.add(int(key))
        for name in ('bos_token', 'eos_token', 'pad_token', 'unk_token'):
            tok = cfg.get(name)
            if isinstance(tok, str):
                converted = tokenizer.convert_tokens_to_ids(tok)
                if converted is not None and int(converted) >= 0:
                    ids.add(int(converted))
    model_cfg = snapshot / 'config.json'
    if model_cfg.is_file():
        cfg = json.loads(model_cfg.read_text())
        for key, value in cfg.items():
            if key.endswith('_token_id') and isinstance(value, int):
                ids.add(value)
        text = cfg.get('text_config') or {}
        for key, value in text.items():
            if key.endswith('_token_id') and isinstance(value, int):
                ids.add(value)
    extra = getattr(tokenizer, 'all_special_ids', None) or []
    for tid in extra:
        ids.add(int(tid))
    return sorted(i for i in ids if 0 <= i < vocab_size)


def iter_text_files(roots: list[Path], skip_names: set[str]):
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            yield root
            continue
        for path in root.rglob('*'):
            if not path.is_file() or path.name in skip_names:
                continue
            if path.suffix.lower() not in {
                '.py', '.txt', '.md', '.json', '.jsonl', '.js', '.ts', '.rs',
                '.go', '.c', '.h', '.cc', '.cpp', '.java', '.sh', '.sql',
                '.toml', '.yml', '.yaml', '.xml', '.html', '.css', '.jinja',
            }:
                continue
            yield path


def read_text(path: Path, limit: int = 2_000_000) -> str:
    try:
        data = path.read_bytes()[:limit]
    except OSError:
        return ''
    return data.decode('utf-8', 'replace')


def wiki_extracts(langs: list[str], pages: int, timeout: float) -> str:
    chunks = []
    opener = urllib.request.build_opener()
    opener.addheaders = [(
        'User-Agent',
        'Qwen38FlashNextRecipe/0.1 (draft-vocab corpus; local experiment)',
    )]
    for lang in langs:
        url = (
            f'https://{lang}.wikipedia.org/w/api.php?action=query'
            f'&generator=random&grnnamespace=0&prop=extracts&explaintext=1'
            f'&exchars=2000&grnlimit={pages}&format=json'
        )
        try:
            with opener.open(url, timeout=timeout) as response:
                payload = json.loads(response.read().decode('utf-8', 'replace'))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            print(f'wiki skip {lang}: {type(exc).__name__}', flush=True)
            continue
        pages_obj = (payload.get('query') or {}).get('pages') or {}
        for page in pages_obj.values():
            title = page.get('title') or ''
            extract = page.get('extract') or ''
            chunks.append(f'# {title}\n{extract}\n')
        print(f'wiki {lang}: {len(pages_obj)} pages', flush=True)
        time.sleep(0.2)
    return '\n'.join(chunks)


def encode_texts(tokenizer, texts: list[str]) -> Counter:
    counts: Counter = Counter()
    for i, text in enumerate(texts):
        if not text.strip():
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        counts.update(int(x) for x in ids)
        if (i + 1) % 200 == 0:
            print(f'encoded {i + 1}/{len(texts)} files', flush=True)
    return counts


def rank_ids(special: list[int], counts: Counter, vocab_size: int, size: int,
             fill: bool) -> tuple[list[int], dict]:
    chosen = []
    seen = set()
    for tid in special:
        if 0 <= tid < vocab_size and tid not in seen:
            chosen.append(tid)
            seen.add(tid)
    ranked = [tid for tid, _ in counts.most_common() if 0 <= tid < vocab_size]
    corpus_added = 0
    for tid in ranked:
        if tid in seen:
            continue
        chosen.append(tid)
        seen.add(tid)
        corpus_added += 1
        if len(chosen) >= size:
            break
    n_corpus = len(chosen)
    n_fill = 0
    if fill:
        for tid in range(vocab_size):
            if len(chosen) >= size:
                break
            if tid not in seen:
                chosen.append(tid)
                seen.add(tid)
                n_fill += 1
    chosen = chosen[:size]
    report = {
        'requested_size': size,
        'vocab_size': vocab_size,
        'n_special': len(special),
        'n_after_specials_and_corpus': n_corpus,
        'n_corpus_rank_added': corpus_added,
        'n_fill': n_fill,
        'n_final': len(chosen),
        'filled_to_size': fill and n_fill > 0,
    }
    return chosen, report


def coverage(tokenizer, paths: list[Path], chosen: set[int]) -> dict:
    total = 0
    hit = 0
    missing: Counter = Counter()
    for path in paths:
        text = read_text(path)
        if not text:
            continue
        ids = [int(x) for x in tokenizer.encode(text, add_special_tokens=False)]
        total += len(ids)
        for tid in ids:
            if tid in chosen:
                hit += 1
            else:
                missing[tid] += 1
    return {
        'tokens': total,
        'in_map': hit,
        'coverage': (hit / total) if total else 0.0,
        'missing_distinct': len(missing),
        'missing_top': missing.most_common(20),
        'files': [str(p) for p in paths],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--corpus', action='append', default=[])
    parser.add_argument('--held-out', action='append', default=[])
    parser.add_argument('--size', type=int, default=65536)
    parser.add_argument('--out', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--wiki-langs', default='en,es,zh,ja,ar,hi,fr,de,ko,ru')
    parser.add_argument('--wiki-pages', type=int, default=20)
    parser.add_argument('--no-wiki', action='store_true')
    parser.add_argument('--no-fill', action='store_true')
    args = parser.parse_args(argv)

    snapshot = Path(args.snapshot)
    tokenizer = load_tokenizer(snapshot)
    vocab_size = vocab_size_of(snapshot, tokenizer)
    special = special_ids(snapshot, tokenizer, vocab_size)
    texts = []
    sources = []
    for raw in args.corpus:
        path = Path(raw)
        files = list(iter_text_files([path], HELD_OUT_NAMES))
        sources.extend(str(p) for p in files)
        texts.extend(read_text(p) for p in files)
    if not args.no_wiki:
        langs = [x.strip() for x in args.wiki_langs.split(',') if x.strip()]
        wiki = wiki_extracts(langs, args.wiki_pages, timeout=20)
        if wiki:
            texts.append(wiki)
            sources.append('wikipedia_random_extracts')
    print(f'encoding {len(texts)} texts, vocab={vocab_size}, specials={len(special)}',
          flush=True)
    counts = encode_texts(tokenizer, texts)
    chosen, stats = rank_ids(special, counts, vocab_size, args.size,
                             fill=not args.no_fill)
    chosen_set = set(chosen)
    held_paths = []
    for raw in args.held_out:
        path = Path(raw)
        if path.is_file():
            held_paths.append(path)
        elif path.is_dir():
            held_paths.extend(p for p in path.glob('*.py') if p.name in HELD_OUT_NAMES)
    cov = coverage(tokenizer, held_paths, chosen_set) if held_paths else {}
    report = {
        'snapshot': str(snapshot),
        'sources': sources,
        'corpus_distinct': len(counts),
        'corpus_occurrences': int(sum(counts.values())),
        **stats,
        'held_out_coverage': cov,
        'special_ids': special,
        'first_16': chosen[:16],
        'last_16': chosen[-16:],
    }
    import torch
    tensor = torch.tensor(chosen, dtype=torch.int64)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, out)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in (
        'n_final', 'n_after_specials_and_corpus', 'n_fill', 'corpus_distinct',
        'corpus_occurrences')}, indent=2), flush=True)
    if cov:
        print('held-out coverage', round(cov['coverage'], 4),
              'tokens', cov['tokens'], flush=True)
    return 0


if __name__ == '__main__':
    # Allow `python3 -c` import of rank_ids without transformers.
    if os.environ.get('RANK_IDS_ONLY') == '1':
        sys.exit(0)
    raise SystemExit(main())
