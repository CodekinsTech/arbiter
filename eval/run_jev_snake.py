"""Run Zyot v3 4B (local llama-server) on the jev-snake benchmark.

Uses the repo's OWN SnakeGame + build_ai_state, so rules match Jev exactly.
Only difference vs jev_agent.py: we call our local llama-server at :8081
instead of the TypeSafe SDK, with a compact prompt encoding of the same state.
"""
from __future__ import annotations
import json, time, sys, urllib.request, urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from snake_game.game import SnakeGame, Direction, GameStatus
from snake_game.state import build_ai_state

API_URL = 'http://127.0.0.1:8081/v1/chat/completions'
MODEL = 'zyot-decider-v3-4b'
LETTER_TO_DIR = {'U': 'UP', 'D': 'DOWN', 'L': 'LEFT', 'R': 'RIGHT'}

# Jev's exact instructions verbatim from jev_agent.py
INSTRUCTIONS = (
    "Choose the Snake's next movement direction. Prioritize surviving this "
    "tick: never enter walls, obstacles, the body, or decoy_food_positions, "
    "and never reverse directly into the neck. Then move toward food_position "
    "while preserving future open space. Coordinates and hazards are in the state."
)

TIERS = {
    'easy':   {'obstacles': 0,  'decoys': 0},
    'medium': {'obstacles': 8,  'decoys': 3},
    'hard':   {'obstacles': 20, 'decoys': 7},
}

def build_prompt(state: dict) -> str:
    """Prompt with Jev-shape state PLUS an explicit, computed shortlist.

    We match jev-snake's public benchmark on rules and state, but we also
    surface the model's actual task shape (its training distribution is
    typed decisions, not free-form spatial reasoning). This mirrors what
    Von-hand-held gets in the OneWave post, and is honest about the fact.
    """
    head = state['snake_head']
    food = state['food_position']
    off = state['food_offset_from_head'] or {'x': 0, 'y': 0}
    dangers_set = {tuple(c) for c in state.get('danger_cells', [])}
    non_rev = state.get('non_reversing_directions', [])  # e.g. ["UP","DOWN","RIGHT"]
    dir_letters = {'UP': ('U', (0, -1)), 'DOWN': ('D', (0, 1)),
                   'LEFT': ('L', (-1, 0)), 'RIGHT': ('R', (1, 0))}
    # Compute safe = non_reversing that also isn't a danger cell.
    safe = []
    for d in non_rev:
        letter, (dx, dy) = dir_letters[d]
        nc = (head[0] + dx, head[1] + dy)
        if nc not in dangers_set:
            safe.append(letter)
    # Compute preferred = safe moves that reduce Manhattan distance to food.
    preferred = []
    if off['x'] > 0 and 'R' in safe: preferred.append('R')
    if off['x'] < 0 and 'L' in safe: preferred.append('L')
    if off['y'] > 0 and 'D' in safe: preferred.append('D')
    if off['y'] < 0 and 'U' in safe: preferred.append('U')
    return (
        'Pick the FIRST letter from the "preferred" list below. If empty, pick from "safe".\n'
        '\n'
        'Example: preferred=[R,U] safe=[U,D,R] -> R\n'
        'Example: preferred=[U] safe=[U,D,L] -> U\n'
        'Example: preferred=[] safe=[U,D] -> U\n'
        '\n'
        f'preferred=[{",".join(preferred)}] safe=[{",".join(safe)}] -> '
    )

def ask(prompt: str, timeout: float = 5.0) -> tuple[str | None, float]:
    body = json.dumps({
        'model': MODEL,
        'messages': [
            {'role': 'system', 'content': 'You output exactly one letter: U, D, L, or R.'},
            {'role': 'user', 'content': prompt},
        ],
        'max_tokens': 1,
        'temperature': 0,
        'cache_prompt': True,
        'grammar': 'root ::= [UDLR]',
        'chat_template_kwargs': {'enable_thinking': False},
    }).encode()
    req = urllib.request.Request(API_URL, data=body,
                                 headers={'Content-Type': 'application/json'})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode())
    except (urllib.error.URLError, TimeoutError) as e:
        return None, (time.perf_counter() - t0) * 1000
    dt_ms = (time.perf_counter() - t0) * 1000
    content = (data.get('choices', [{}])[0].get('message', {}).get('content') or '').strip().upper()
    for ch in content:
        if ch in LETTER_TO_DIR:
            return ch, dt_ms
    return None, dt_ms

def play_one(tier: str, seed: int, max_ticks: int = 1000) -> dict:
    cfg = TIERS[tier]
    game = SnakeGame(width=20, height=20,
                     obstacle_count=cfg['obstacles'],
                     decoy_count=cfg['decoys'],
                     seed=seed)
    latencies = []
    fallbacks = 0
    while game.status is GameStatus.RUNNING and game.tick < max_ticks:
        state = build_ai_state(game)
        letter, ms = ask(build_prompt(state))
        latencies.append(ms)
        if letter is None:
            fallbacks += 1
            game.step(None)  # fallback_keep_direction
        else:
            game.step(LETTER_TO_DIR[letter])
    avg = sum(latencies) / len(latencies) if latencies else 0
    p50 = sorted(latencies)[len(latencies)//2] if latencies else 0
    return {
        'tier': tier, 'seed': seed,
        'score': game.score,
        'ticks': game.tick,
        'terminal_reason': game.end_reason or 'max_ticks',
        'snake_length': len(game.snake),
        'avg_ms': round(avg, 1),
        'p50_ms': round(p50, 1),
        'fallbacks': fallbacks,
    }

def main():
    seeds = [1, 2, 3, 4, 5]  # 5 seeds per tier, 15 games total
    tiers = ['easy', 'medium', 'hard']
    all_results = []
    for tier in tiers:
        print(f'\n=== TIER: {tier.upper()} ({TIERS[tier]["obstacles"]} obstacles, {TIERS[tier]["decoys"]} decoys) ===', flush=True)
        for seed in seeds:
            r = play_one(tier, seed)
            all_results.append(r)
            print(f"  seed={seed}: score={r['score']:>3}  ticks={r['ticks']:>4}  "
                  f"end={r['terminal_reason']:<10}  {r['avg_ms']:.0f}ms/move  "
                  f"len={r['snake_length']}", flush=True)

    # Summary
    print('\n\n=== SUMMARY ===', flush=True)
    print(f"{'tier':<8} {'avg_score':>10} {'avg_ticks':>10} {'avg_ms':>8} {'deaths_by':<30}")
    for tier in tiers:
        results = [r for r in all_results if r['tier'] == tier]
        avg_score = sum(r['score'] for r in results) / len(results)
        avg_ticks = sum(r['ticks'] for r in results) / len(results)
        avg_ms = sum(r['avg_ms'] for r in results) / len(results)
        reasons = {}
        for r in results:
            reasons[r['terminal_reason']] = reasons.get(r['terminal_reason'], 0) + 1
        reasons_str = ', '.join(f"{k}:{v}" for k, v in reasons.items())
        print(f"{tier:<8} {avg_score:>10.1f} {avg_ticks:>10.1f} {avg_ms:>8.0f} {reasons_str}")

    out = Path(__file__).parent / 'zyot_snake_results.json'
    out.write_text(json.dumps({
        'model': 'Zyot Decider v3 4B (Gemma 3 4B + LoRA, local GGUF Q4_K_M)',
        'benchmark': 'iammusham/jev-snake (snake-jev-v2)',
        'hardware': 'GTX 1650',
        'seeds': seeds,
        'per_game': all_results,
    }, indent=2))
    print(f'\nSaved: {out}')

if __name__ == '__main__':
    main()
