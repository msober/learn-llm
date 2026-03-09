import sys
import os

# Ensure the project root is on sys.path so both `model.*` and `sample` imports work
# regardless of which directory the script is invoked from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import time
from modelscope import snapshot_download
from model.qwen3.config import QWEN3_0_6B_CONFIG
from model.qwen3.tokenizer import Qwen3Tokenizer
from model.qwen3.modeling_qwen3 import Qwen3Model
from safetensors.torch import load_file
from sample import SamplingParams, sample



def load_weights_into_qwen(model, model_config, pretrained_weights):
    def assign(target_param, source_weight, tensor_name="unknown"):
        if target_param.shape != source_weight.shape:
            raise ValueError(f"{tensor_name}: {target_param.shape} != {source_weight.shape}")
        
        with torch.no_grad():
            if isinstance(source_weight, torch.Tensor):
                target_param.copy_(source_weight)
            else:
                target_param.copy_(torch.as_tensor(source_weight, dtype=target_param.dtype, device=target_param.device))
    
        return target_param 

    model.token_embedding.weight = assign(model.token_embedding.weight, pretrained_weights["model.embed_tokens.weight"], "model.embed_tokens.weight")

    for layer_idx in range(model_config["n_layers"]):
        transformer_block = model.transformer_blocks[layer_idx]
        attention = transformer_block.attention

        # Q, K, V projections
        attention.query_projection.weight = assign(
            attention.query_projection.weight,
            pretrained_weights[f"model.layers.{layer_idx}.self_attn.q_proj.weight"],
            f"model.layers.{layer_idx}.self_attn.q_proj.weight"
        )
        attention.key_projection.weight = assign(
            attention.key_projection.weight,
            pretrained_weights[f"model.layers.{layer_idx}.self_attn.k_proj.weight"],
            f"model.layers.{layer_idx}.self_attn.k_proj.weight"
        )
        attention.value_projection.weight = assign(
            attention.value_projection.weight,
            pretrained_weights[f"model.layers.{layer_idx}.self_attn.v_proj.weight"],
            f"model.layers.{layer_idx}.self_attn.v_proj.weight"
        )

        # Output projection
        attention.output_projection.weight = assign(
            attention.output_projection.weight,
            pretrained_weights[f"model.layers.{layer_idx}.self_attn.o_proj.weight"],
            f"model.layers.{layer_idx}.self_attn.o_proj.weight"
        )

        # QK norms
        if hasattr(attention, "query_norm") and attention.query_norm is not None:
            attention.query_norm.scale = assign(
                attention.query_norm.scale,
                pretrained_weights[f"model.layers.{layer_idx}.self_attn.q_norm.weight"],
                f"model.layers.{layer_idx}.self_attn.q_norm.weight"
            )
        if hasattr(attention, "key_norm") and attention.key_norm is not None:
            attention.key_norm.scale = assign(
                attention.key_norm.scale,
                pretrained_weights[f"model.layers.{layer_idx}.self_attn.k_norm.weight"],
                f"model.layers.{layer_idx}.self_attn.k_norm.weight"
            )

        # Attention layernorm
        transformer_block.attention_norm.scale = assign(
            transformer_block.attention_norm.scale,
            pretrained_weights[f"model.layers.{layer_idx}.input_layernorm.weight"],
            f"model.layers.{layer_idx}.input_layernorm.weight"
        )

        # Feedforward weights
        transformer_block.feed_forward.gate_projection.weight = assign(
            transformer_block.feed_forward.gate_projection.weight,
            pretrained_weights[f"model.layers.{layer_idx}.mlp.gate_proj.weight"],
            f"model.layers.{layer_idx}.mlp.gate_proj.weight"
        )
        transformer_block.feed_forward.up_projection.weight = assign(
            transformer_block.feed_forward.up_projection.weight,
            pretrained_weights[f"model.layers.{layer_idx}.mlp.up_proj.weight"],
            f"model.layers.{layer_idx}.mlp.up_proj.weight"
        )
        transformer_block.feed_forward.down_projection.weight = assign(
            transformer_block.feed_forward.down_projection.weight,
            pretrained_weights[f"model.layers.{layer_idx}.mlp.down_proj.weight"],
            f"model.layers.{layer_idx}.mlp.down_proj.weight"
        )
        transformer_block.feed_forward_norm.scale = assign(
            transformer_block.feed_forward_norm.scale,
            pretrained_weights[f"model.layers.{layer_idx}.post_attention_layernorm.weight"],
            f"model.layers.{layer_idx}.post_attention_layernorm.weight"
        )

    # Final normalization and output head
    model.final_norm.scale = assign(model.final_norm.scale, pretrained_weights["model.norm.weight"], "model.norm.weight")

    if "lm_head.weight" in pretrained_weights:
        model.output_projection.weight = assign(model.output_projection.weight, pretrained_weights["lm_head.weight"], "lm_head.weight")
    else:
        model.output_projection.weight = model.token_embedding.weight
        print("Model uses weight tying.")


def generate_text_basic_stream(model, token_ids, max_new_tokens, eos_token_id=None, sampling_params=None):
    """Generate tokens one at a time using KV cache for efficient autoregressive decoding.

    The first forward pass (prefill) processes the full prompt and populates the KV cache.
    Subsequent forward passes (decode) process only the newly generated token, reusing
    cached keys and values from all previous positions.

    Args:
        model: The language model (must support start_position and clear_kv_cache).
        token_ids: Input token ids tensor of shape (1, seq_len), single sequence only.
        max_new_tokens: Maximum number of tokens to generate.
        eos_token_id: End-of-sequence token id for early stopping.
        sampling_params: SamplingParams instance controlling temperature/top-k/top-p.
                         If None, falls back to greedy decoding (argmax).
    """
    model.eval()
    model.clear_kv_cache()

    with torch.no_grad():
        # Prefill: process the entire prompt at once (start_position=0)
        outputs = model(token_ids, start_position=0)
        last_token_logits = outputs[0, -1:]
        next_token_id = sample(last_token_logits, sampling_params)

        if eos_token_id is not None and next_token_id.item() == eos_token_id:
            return

        yield next_token_id.item()

        current_position = token_ids.shape[1]

        # Decode: generate one token at a time, reusing KV cache
        for _ in range(max_new_tokens - 1):
            outputs = model(next_token_id, start_position=current_position)
            last_token_logits = outputs[0, -1:]
            next_token_id = sample(last_token_logits, sampling_params)

            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break

            yield next_token_id.item()
            current_position += 1


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("hello qwen3 inference!")

if __name__ == "__main__":
    # --------------------------------------------------
    #  从modelscope下载模型
    # --------------------------------------------------
    model_repo_id = "Qwen/Qwen3-0.6B"
    model_save_dir = f"model_repo/{model_repo_id}"
    model_local_dir = snapshot_download(model_id=model_repo_id, local_dir=model_save_dir)

    # --------------------------------------------------
    #  初始化自己实现的模型
    # --------------------------------------------------
    model = Qwen3Model(QWEN3_0_6B_CONFIG)
    # MPS does not fully support bfloat16; convert to float16 on Apple Silicon
    if device.type == "mps":
        model = model.to(torch.float16)
    model.to(device)

    # --------------------------------------------------
    #  加载 tokenizer和官方权重
    # --------------------------------------------------
    # load original qwen weight
    weights_filepath = os.path.join(model_local_dir, "model.safetensors")
    pretrained_weights = load_file(weights_filepath)
    # load weight into our qwen 
    load_weights_into_qwen(model, QWEN3_0_6B_CONFIG, pretrained_weights)
    del pretrained_weights
    # load tokenizer
    tokenizer_file_path = f"{model_save_dir}/tokenizer.json"
    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=tokenizer_file_path,
        repo_id=model_repo_id,  
        apply_chat_template=True,
        add_generation_prompt=True,
        add_thinking=False
    )

    # --------------------------------------------------
    #  自回归生成
    # --------------------------------------------------
    prompt = input("Please input your prompt: ")
    input_token_ids = tokenizer.encode(prompt)
    input_token_ids_tensor = torch.tensor(input_token_ids, device=device).unsqueeze(0)

    # sampling params (matching Qwen3-0.6B generation_config.json defaults)
    sampling_params = SamplingParams(temperature=0.6, top_k=20, top_p=0.95)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_time = time.perf_counter()
    generated_tokens = 0

    for token_id in generate_text_basic_stream(
        model=model,
        token_ids=input_token_ids_tensor,
        max_new_tokens=500,
        eos_token_id=tokenizer.eos_token_id,
        sampling_params=sampling_params
    ):
        generated_tokens += 1
        print(tokenizer.decode([token_id]), end="", flush=True)

    elapsed_seconds = time.perf_counter() - start_time
    generation_speed = generated_tokens / elapsed_seconds if elapsed_seconds > 0 else 0.0
    print(f"\n\nGeneration speed: {generation_speed:.2f} tokens/sec")

    if torch.cuda.is_available():
        def calc_gpu_gb(bytes_count):
            return f"{bytes_count / 1024 / 1024 / 1024:.2f} GB"

        print(f"GPU memory used: {calc_gpu_gb(torch.cuda.max_memory_allocated())}")

