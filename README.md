
## Learn LLM

A from-scratch implementation of the **Qwen3** large language model in pure PyTorch — no Hugging Face `transformers` dependency required for model inference. Built for learning and experimentation.

<p align="center">
  <img src="images/qnwe3.png" />
</p>

###  Features

- **Pure PyTorch Qwen3 implementation** — Transformer blocks, Grouped-Query Attention (GQA), RoPE, RMSNorm, SwiGLU FFN, all hand-written
- **Streaming text generation** — token-by-token output with configurable sampling (temperature / top-k / top-p)
- **Automatic weight loading** — downloads official Qwen3-0.6B weights from ModelScope and maps them into the custom model
- **Chat template support** — built-in Qwen3 chat formatting with optional thinking mode
- **Multi-device support** — automatically selects CUDA, Apple MPS, or CPU

###  Project Structure

```
easy-llm/
├── model/
│   └── qwen3/
│       ├── config.py              # Qwen3-0.6B model hyperparameters
│       ├── modeling_qwen3.py      # Full model: Transformer, GQA, RoPE, RMSNorm, SwiGLU
│       └── tokenizer.py           # Tokenizer with chat template support
├── inference/
│   ├── qwen3_infer.py             # Main inference entry point
│   └── sample.py                  # Sampling strategies (temperature, top-k, top-p)
├── model_repo/                    # Auto-downloaded model weights (git-ignored)
├── requirements.txt
└── README.md
```

###  Quick Start

#### 1. Create & activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

#### 2. Install dependencies

```bash
pip install -r requirements.txt
```

#### 3. Run inference

```bash
python inference/qwen3_infer.py
```

The script will automatically download the **Qwen3-0.6B** weights from ModelScope on the first run, then prompt you for input and stream the generated response.

###  Sampling Parameters

The default sampling configuration mirrors the official Qwen3-0.6B `generation_config.json`:

| Parameter     | Default | Description                                      |
|---------------|---------|--------------------------------------------------|
| `temperature` | 0.6     | Controls randomness (0 = greedy, higher = more random) |
| `top_k`       | 20      | Keeps only the top-k most likely tokens          |
| `top_p`       | 0.95    | Nucleus sampling probability threshold           |

### 🏗️ Model Architecture

| Component                    | Detail                              |
|------------------------------|-------------------------------------|
| **Embedding dim**            | 1024                                |
| **Layers**                   | 28                                  |
| **Attention heads**          | 16 (query) / 8 (KV groups, GQA)    |
| **Head dim**                 | 128                                 |
| **FFN hidden dim**           | 3072 (SwiGLU)                       |
| **Positional encoding**      | RoPE (θ = 1,000,000)               |
| **Normalization**            | RMSNorm with QK-norm                |
| **Vocab size**               | 151,936                             |
| **Context length**           | 40,960                              |
| **Precision**                | bfloat16                            |

###  Dependencies

- `torch` ≥ 2.2.2
- `tokenizers`
- `numpy` ≥ 1.26
- `safetensors`
- `modelscope`

See `requirements.txt` for full details.