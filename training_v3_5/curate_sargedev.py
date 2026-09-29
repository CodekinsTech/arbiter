"""Curate 25k noul (T/F) rows from SargeDev/jev-distill-corpus-v3.

Actual schema (verified):
  id, kind, options (stringified list), target (stringified soft-label list),
  state (plain string context), question, domain, family, source

Filters:
  - kind == 'noul'
  - source in {yuri_v3, openjev_v2}  (drop yuri_v1 per plan)
  - options == ['false', 'true']
  - max(target) >= 0.6 (teacher confidence gate)
  - length 50-4000 chars
  - fuzzy dedup on (state + question)[:200]

70% yuri_v3, 30% openjev_v2 (matches v3 successful ratio).

Output schema: {prompt, answer_slot} — slot 0 = T (true), slot 1 = F (false).
"""
import os, json, random, hashlib, ast
from pathlib import Path
os.environ['HF_HUB_DISABLE_XET'] = '1'

from datasets import load_dataset

OUT_DIR = Path(__file__).parent
OUT_PATH = OUT_DIR / 'train_70k.jsonl'

TARGETS = {'noul': 40000, 'choice': 20000, 'score': 10000}  # total 70k
YURI_RATIO = 0.70
CONF_THRESHOLD = 0.60
LEN_MIN, LEN_MAX = 50, 4000

# Slot mapping (must match train.py VERBALIZER exactly):
# 0=T, 1=F, 2-17=A..P (up to 16 choice options), 18-23=0..5 (score)
SLOT_TF = {'true': 0, 'false': 1}
SLOT_CHOICE = {chr(ord('A')+i): 2+i for i in range(16)}      # A..P -> 2..17
SLOT_SCORE = {str(i): 18+i for i in range(6)}                 # '0'..'5' -> 18..23

random.seed(20260930)

def parse_list(s):
    if isinstance(s, list): return s
    if isinstance(s, str):
        try: return ast.literal_eval(s)
        except Exception: return None
    return None

def to_canonical(row):
    """Return {prompt, answer_slot, kind, source} or None."""
    kind = row.get('kind')
    if kind not in ('noul', 'choice', 'score'): return None
    opts = parse_list(row.get('options'))
    target = parse_list(row.get('target'))
    if not isinstance(opts, list) or not isinstance(target, list): return None
    if len(opts) != len(target): return None
    conf = max(target)
    if conf < CONF_THRESHOLD: return None
    state = (row.get('state') or '').strip()
    question = (row.get('question') or '').strip()
    if not state or not question: return None
    winning_idx = target.index(conf)

    if kind == 'noul':
        if opts != ['false', 'true']: return None
        slot = SLOT_TF['true'] if winning_idx == 1 else SLOT_TF['false']  # 0=T, 1=F
        prompt = (f'State: {state}\n\nQuestion: {question}\n\n'
                  f'Options:\nT. Yes / True\nF. No / False\n\nAnswer:')
    elif kind == 'choice':
        if not (2 <= len(opts) <= 16): return None
        # options come as arbitrary strings — map to A..P letters
        letters = [chr(ord('A')+i) for i in range(len(opts))]
        slot = SLOT_CHOICE[letters[winning_idx]]
        opt_lines = '\n'.join(f'{L}. {o}' for L, o in zip(letters, opts))
        prompt = f'State: {state}\n\nQuestion: {question}\n\nOptions:\n{opt_lines}\n\nAnswer:'
    elif kind == 'score':
        if opts != [str(i) for i in range(len(opts))]: return None
        if len(opts) != 6: return None    # 0..5 scale
        slot = SLOT_SCORE[str(winning_idx)]
        prompt = (f'State: {state}\n\nQuestion: {question}\n\n'
                  f'Options:\n0\n1\n2\n3\n4\n5\n\nAnswer:')
    else:
        return None

    if not (LEN_MIN <= len(prompt) <= LEN_MAX): return None
    return {'prompt': prompt, 'answer_slot': slot, 'kind': kind, 'source': row.get('source', '')}


def main():
    # per-kind, per-source quota
    per_kind_yuri = {k: int(n * YURI_RATIO) for k, n in TARGETS.items()}
    per_kind_openjev = {k: TARGETS[k] - per_kind_yuri[k] for k in TARGETS}
    print(f'Per-kind targets:')
    for k in TARGETS:
        print(f'  {k:<8}: yuri_v3={per_kind_yuri[k]}, openjev_v2={per_kind_openjev[k]}')

    print('\nStreaming SargeDev/jev-distill-corpus-v3...')
    ds = load_dataset('SargeDev/jev-distill-corpus-v3', split='train', streaming=True)

    kept = {(k, s): [] for k in TARGETS for s in ('yuri_v3', 'openjev_v2')}
    seen = set()
    total_seen = skipped = 0

    def all_full():
        return all(
            len(kept[(k, 'yuri_v3')]) >= per_kind_yuri[k] and
            len(kept[(k, 'openjev_v2')]) >= per_kind_openjev[k]
            for k in TARGETS
        )

    for row in ds:
        total_seen += 1
        if total_seen % 50000 == 0:
            counts = {k: sum(len(kept[(k, s)]) for s in ('yuri_v3','openjev_v2')) for k in TARGETS}
            print(f'  scanned {total_seen}  noul={counts["noul"]}  choice={counts["choice"]}  score={counts["score"]}')
        if all_full(): break

        src = row.get('source', '')
        if src not in ('yuri_v3', 'openjev_v2'):
            skipped += 1; continue

        item = to_canonical(row)
        if item is None:
            skipped += 1; continue

        k = item['kind']
        quota = per_kind_yuri[k] if src == 'yuri_v3' else per_kind_openjev[k]
        if len(kept[(k, src)]) >= quota:
            continue

        h = hashlib.md5(item['prompt'][:200].encode('utf-8')).hexdigest()
        if h in seen: continue
        seen.add(h)

        kept[(k, src)].append(item)

    print(f'\nScanned {total_seen} rows total, skipped {skipped}')
    for k in TARGETS:
        y = len(kept[(k, 'yuri_v3')]); o = len(kept[(k, 'openjev_v2')])
        print(f'  {k:<8}: yuri_v3={y}, openjev_v2={o}, total={y+o}')

    all_rows = [it for bucket in kept.values() for it in bucket]
    random.shuffle(all_rows)

    print(f'\nSlot distribution:')
    slot_counts = {}
    for r in all_rows: slot_counts[r['answer_slot']] = slot_counts.get(r['answer_slot'], 0) + 1
    for s in sorted(slot_counts): print(f'  slot {s:>2}: {slot_counts[s]}')

    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'\nWrote {len(all_rows)} rows to {OUT_PATH}')
    print(f'  file size: {OUT_PATH.stat().st_size/1e6:.1f} MB')


if __name__ == '__main__':
    main()
