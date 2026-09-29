"""Zyot v3 4B — build GGUF adapter for llama.cpp/Ollama runtime loading.

Steps:
  1. Download unsloth/gemma-3-4b-it-GGUF Q4_K_M as base (~2.5GB)
  2. Convert our LoRA safetensors → LoRA GGUF (small, ~130MB)
  3. Smoke-test: load base + LoRA in llama-cli, verify one prediction works
  4. Stage files for HF upload as hiteshluke/zyot-decider-v3-4b-gguf

Requires: llama.cpp tools already cloned at zyot_decider_v2/llama_cpp_tools/
"""
import os, subprocess, sys
from pathlib import Path

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

V2_TOOLS = Path(r'C:\Users\Nishan\Desktop\projects\klexdecide\zyot_decider_v2\llama_cpp_tools')
LLAMA_CPP = V2_TOOLS / 'llama.cpp'
LLAMA_BIN = V2_TOOLS / 'llama-bin'

ADAPTER = Path(r'C:\Users\Nishan\Desktop\projects\klexdecide\zyot_decider_v3\adapter_v3\zyot-v3-minimal\final')
GGUF_OUT = Path(r'C:\Users\Nishan\Desktop\projects\klexdecide\zyot_decider_v3\gguf')
GGUF_OUT.mkdir(exist_ok=True)

BASE_REPO = 'unsloth/gemma-3-4b-it-GGUF'
BASE_QUANT = 'gemma-3-4b-it-Q4_K_M.gguf'

# --- 1. Download base ---
print('=== 1. Download Gemma 3 4B Q4_K_M base ===', flush=True)
from huggingface_hub import hf_hub_download
base_path = GGUF_OUT / BASE_QUANT
if not base_path.exists():
    print(f'Fetching {BASE_QUANT} from {BASE_REPO}...', flush=True)
    fetched = hf_hub_download(BASE_REPO, filename=BASE_QUANT, local_dir=str(GGUF_OUT))
    print(f'  Saved: {fetched} ({os.path.getsize(fetched)/1e9:.2f} GB)', flush=True)
else:
    print(f'  Already have: {base_path}', flush=True)

# --- 2. Convert LoRA to GGUF ---
print('\n=== 2. Convert LoRA safetensors → GGUF ===', flush=True)
lora_gguf = GGUF_OUT / 'zyot-decider-v3-4b-lora.gguf'
subprocess.check_call([
    sys.executable, str(V2_TOOLS / 'convert_lora_to_gguf.py'),
    '--outfile', str(lora_gguf),
    '--outtype', 'f16',
    str(ADAPTER),
], cwd=str(LLAMA_CPP))
print(f'  Wrote: {lora_gguf} ({os.path.getsize(lora_gguf)/1e6:.1f} MB)', flush=True)

# --- 3. Smoke test ---
print('\n=== 3. Smoke test with llama-cli ===', flush=True)
smoke_prompt = ('State: The sky is generally blue during daytime.\n\n'
                'Question: Based on the passage, is the answer to "is the sky blue" yes?\n\n'
                'Options:\nT. Yes / True\nF. No / False\n\nAnswer:')
cmd = [
    str(LLAMA_BIN / 'llama-cli.exe'),
    '-m', str(base_path),
    '--lora', str(lora_gguf),
    '-p', smoke_prompt,
    '-n', '3',
    '-st',
    '--no-warmup',
    '-ngl', '25',
]
print(f'  Running: {" ".join(cmd[:3])} ... (smoke test)', flush=True)
result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
print('--- llama-cli output ---')
print(result.stdout[-500:] if result.stdout else '(no stdout)')
if result.returncode != 0:
    print('STDERR:', (result.stderr or '')[-500:])
    print('WARNING: smoke test non-zero exit', file=sys.stderr)

# --- 4. Write Modelfile for Ollama users (once export-lora is done manually) ---
print('\n=== 4. Write Modelfile ===', flush=True)
modelfile = GGUF_OUT / 'Modelfile'
modelfile.write_text('''FROM ./gemma-3-4b-it-Q4_K_M.gguf
ADAPTER ./zyot-decider-v3-4b-lora.gguf

TEMPLATE """{{- if .System }}<start_of_turn>user
{{ .System }}

{{ .Prompt }}<end_of_turn>
{{- else }}<start_of_turn>user
{{ .Prompt }}<end_of_turn>
{{- end }}
<start_of_turn>model
"""

PARAMETER stop "<start_of_turn>"
PARAMETER stop "<end_of_turn>"
PARAMETER num_ctx 2048
PARAMETER num_gpu 25
PARAMETER temperature 0.1

SYSTEM """You are Zyot Decider — a soft-labeled decision assistant."""
''', encoding='utf-8')
print(f'  Wrote: {modelfile}')

# --- Final report ---
print('\n=== DONE. Files ready in', GGUF_OUT, '===')
for p in sorted(GGUF_OUT.iterdir()):
    if p.is_file():
        print(f'  {p.name}  ({p.stat().st_size/1e6:.1f} MB)')

print('\nNext step: push to HF as hiteshluke/zyot-decider-v3-4b-gguf')
