# Running Nemotron on MAX

The Nemotron architecture is registered as **NemotronForCausalLM** (non-legacy). You must set **`use_legacy_module=false`** so the registry resolves to this implementation.

## Build

From the repo root:

```bash
./bazelw build //max/python/max/entrypoints:pipelines
```

(Or build the full pipelines tree: `./bazelw build //max/python/max/pipelines/...`)

## Generate (CLI)

```bash
./bazelw run //max/python/max/entrypoints:pipelines -- \
  generate \
  --model-path nvidia/Nemotron-Mini-4B-Instruct \
  --use-legacy-module false \
  --prompt "Hello, world" \
  --max-new-tokens 64
```

## Serve

```bash
max serve \
  --model-path nvidia/Nemotron-Mini-4B-Instruct \
  --use-legacy-module false
```

If using a config file, set:

```yaml
use_legacy_module: false
model:
  model_path: nvidia/Nemotron-Mini-4B-Instruct
```

## Integration logit verification

```bash
# MAX logits
./bazelw run //max/tests/integration:generate_llm_logits -- \
  --device gpu --framework max --pipeline nvidia/Nemotron-Mini-4B-Instruct \
  --encoding bfloat16 --output /tmp/max-logits.json

# (Pipeline config used by the test harness must have use_legacy_module=false for Nemotron.)
```
