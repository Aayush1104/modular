# Nemotron-4 for MAX (Modular)

This directory contains a migration of the HuggingFace Transformers Nemotron model
to the MAX framework using both the Python Graph API and Mojo custom ops.

## Architecture Overview

Nemotron-4 is a decoder-only transformer with the following key differences from Llama:

| Feature | Llama | Nemotron-4 |
|---------|-------|------------|
| Normalization | RMSNorm | LayerNorm (no bias) |
| MLP Activation | SiLU (SwiGLU) | ReLU² (Squared ReLU) |
| Rotary Embeddings | Full RoPE | Partial RoPE (default 50% of head_dim) |
| Position Embedding | Full dim | `partial_rotary_factor * head_dim` dims |
| Default vocab_size | 128256 | 256000 |
| Default hidden_size | 4096 | 6144 |

## File Structure

```
nemotron/
├── __init__.py              # Architecture registration
├── config.py                # NemotronConfig (HuggingFace config adapter)
├── model.py                 # NemotronModel (MAX PipelineModel)
├── arch.py                  # Model graph construction (MAX Graph API)
├── layers/
│   ├── __init__.py
│   ├── attention.py         # Nemotron attention with partial RoPE
│   ├── mlp.py               # Nemotron MLP with ReLU² activation
│   ├── norm.py              # LayerNorm (instead of RMSNorm)
│   └── rotary_embedding.py  # Partial rotary embedding
├── weight_adapters.py       # SafeTensors / GGUF weight loading
└── mojo_ops/
    ├── squared_relu.mojo    # Custom ReLU² kernel
    ├── layernorm.mojo       # Custom LayerNorm kernel
    └── partial_rope.mojo    # Partial RoPE kernel
```

## Usage

### As a custom architecture with MAX Serve

```bash
max serve \
  --model nvidia/nemotron-3-8b-base-4k-hf \
  --custom-architectures ./nemotron \
  --trust-remote-code
```

### Extending from Llama3 (recommended for quick integration)

Since Nemotron-4 is architecturally close to Llama, the fastest path is to
extend `Llama3Model` and override the differing components. See `model.py`.

## Key Architectural Details

### ReLU² (Squared ReLU)
```python
def squared_relu(x):
    return torch.relu(x) ** 2
```

### Partial Rotary Embeddings
Only the first `partial_rotary_factor * head_dim` dimensions of Q and K
get rotary embeddings applied. The remaining dimensions pass through unchanged.

### LayerNorm
Standard LayerNorm (without bias) instead of RMSNorm:
```python
LayerNorm(hidden_size, eps=norm_eps, elementwise_affine=True, bias=False)
```
