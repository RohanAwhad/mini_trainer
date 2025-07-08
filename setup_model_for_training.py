import math
import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

from utils import log_rank_0, patch_target_module

# New simple HF-only activation-checkpointing + FSDP2 wrapper
# This mirrors TorchTitan: checkpoint each block, then shard each block and the full model.
def create_tensor_parallel_plan(model) -> dict:
    """Create tensor parallel plan for transformer decoder layers."""
    from torch.distributed.tensor.parallel import (
        ColwiseParallel, RowwiseParallel, SequenceParallel, PrepareModuleInput
    )
    from torch.distributed.tensor import Shard, Replicate
    
    return {
        # LayerNorm layers with sequence parallelism
        "input_layernorm": SequenceParallel(),
        "post_attention_layernorm": SequenceParallel(),
        
        # Attention module preparation 
        "self_attn": PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        ),
        
        # Attention projections
        "self_attn.q_proj": ColwiseParallel(),
        "self_attn.k_proj": ColwiseParallel(),
        "self_attn.v_proj": ColwiseParallel(),
        "self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
        
        # Attention norms (sequence parallel)
        "self_attn.q_norm": SequenceParallel(),
        "self_attn.k_norm": SequenceParallel(),
        
        # MLP module preparation
        "mlp": PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        ),
        
        # MLP projections
        "mlp.gate_proj": ColwiseParallel(),
        "mlp.up_proj": ColwiseParallel(),
        "mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
    }


def parallelize_full_model(model, tp_mesh):
    """Parallelize the entire model including embeddings and output layers."""
    from torch.distributed.tensor.parallel import (
        ColwiseParallel, RowwiseParallel, SequenceParallel, parallelize_module
    )
    from torch.distributed.tensor import Shard, Replicate
    
    # 1. Parallelize decoder layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        layer_tp_plan = create_tensor_parallel_plan(model)
        
        for block in layers:
            parallelize_module(module=block, device_mesh=tp_mesh, parallelize_plan=layer_tp_plan)
    
    # 2. Parallelize model-level components
    model_tp_plan = {}
    
    # Embedding layer - row-wise parallel with sequence output
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        model_tp_plan["model.embed_tokens"] = RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(1),
        )
    
    # Final layer norm - sequence parallel
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        model_tp_plan["model.norm"] = SequenceParallel()
    
    # Output projection - column-wise parallel with DTensor output for loss parallel
    if hasattr(model, "lm_head"):
        model_tp_plan["lm_head"] = ColwiseParallel(
            input_layouts=Shard(1),
            use_local_output=False,  # Keep DTensor for loss parallel
        )
    
    # Apply model-level parallelization
    if model_tp_plan:
        parallelize_module(module=model, device_mesh=tp_mesh, parallelize_plan=model_tp_plan)
    
    return model

def wrap_fsdp2(model: torch.nn.Module, fsdp_mesh, tp_mesh) -> torch.nn.Module:
    # Move model to GPU and disable HuggingFace cache
    model = model.to(torch.device("cuda"))
    if hasattr(model, 'config'):
        try:
            model.config.use_cache = False
        except Exception:
            pass
    
    # Find transformer layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        raise ValueError("Cannot find transformer block container on model")
    
    # Activation checkpoint each block
    for idx, block in enumerate(layers):
        layers[idx] = ptd_checkpoint_wrapper(block, preserve_rng_state=False)

    # Apply tensor parallelism to full model
    model = parallelize_full_model(model, tp_mesh)

    # Mixed-precision policy
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, 
        reduce_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16
    )

    # FSDP2 wrap each block
    for idx, block in enumerate(layers):
        reshard = idx < len(layers) - 1
        fully_shard(block, mesh=fsdp_mesh, mp_policy=mp_policy, reshard_after_forward=reshard)

    # FSDP2 wrap full model
    fully_shard(model, mesh=fsdp_mesh, mp_policy=mp_policy, reshard_after_forward=True)
    return model

def align_model_and_tokenizer(model, tokenizer):
    """
    Aligns the model's vocabulary and special tokens with the tokenizer.
    """
    if len(tokenizer) > model.config.vocab_size:
        print(
            f"WARNING: tokenizer has {len(tokenizer)} tokens but model has {model.config.vocab_size} vocab size"
        )
        model.resize_token_embeddings(
            int(8 * math.ceil(len(tokenizer) / 8.0))
        )  # make the vocab size multiple of 8 for sharding the embedding layer.

    # Fix any discrepancy between model and tokenizer
    special_tokens = {
        'pad': ('pad_token_id', 'Fixing model pad token id'),
        'bos': ('bos_token_id', 'Fixing model bos token id'),
        'eos': ('eos_token_id', 'Fixing model eos token id')
    }

    for token_type, (token_attr, message) in special_tokens.items():
        model_token = getattr(model.config, token_attr)
        tokenizer_token = getattr(tokenizer, token_attr)
        
        if (model_token is not None and tokenizer_token is not None 
            and model_token != tokenizer_token):
            log_rank_0(
                "\033[38;5;226m"
                f"WARNING: There is a mismatch between {token_type} token id of "
                f"model({model_token}) and tokenizer({tokenizer_token}). "
                f"{message} to be same as tokenizer's {token_type} token id"
                "\033[0m"
            )
            setattr(model.config, token_attr, tokenizer_token)

    return model

def setup_model(model=None, **kwargs):
    base_model_args = {
        "pretrained_model_name_or_path": kwargs['model_name_or_path'],
        "torch_dtype": torch.bfloat16,
    }
    base_model_args["attn_implementation"] = "flash_attention_2"

    if kwargs['use_liger_kernels']:
        '''need to patch the loss function to not reduce, so we can reduce across all GPUs'''
        if kwargs['tp_size'] >= 1: raise ValueError("Liger kernels are not supported with tensor parallelism")
        from none_reduction_losses import liger_fixed_fused_linear_cross_entropy_none_reduction
        patch_target_module("liger_kernel.transformers.model.loss_utils.fixed_fused_linear_cross_entropy", 
                            liger_fixed_fused_linear_cross_entropy_none_reduction)
        from liger_kernel.transformers import AutoLigerKernelForCausalLM
        model = AutoLigerKernelForCausalLM.from_pretrained(**base_model_args)
    else:
        from none_reduction_losses import hf_fixed_cross_entropy_none_reduction
        patch_target_module("transformers.loss.loss_utils.fixed_cross_entropy", 
                            hf_fixed_cross_entropy_none_reduction)
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(**base_model_args)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(kwargs['model_name_or_path'])
    model = align_model_and_tokenizer(model, tokenizer)

    if model.__class__.__name__ not in [
        "MistralForCausalLM",
        "GPTDolomiteForCausalLM", 
        "LlamaForCausalLM",
        "Starcoder2ForCausalLM",
        "GemmaForCausalLM",
        "MixtralForCausalLM",
        "GraniteForCausalLM",
        "Qwen2ForCausalLM",
        "Qwen3ForCausalLM",
    ]:
        log_rank_0(
            f"\033[38;2;255;255;0mWarning: Model class name: {model.__class__.__name__} is not in the list of supported models.\033[0m",
            to_print=True,
        )

    # NOTE: Don't enable HuggingFace gradient checkpointing with FSDP2
    # It causes conflicts. TorchTitan applies PyTorch's checkpoint wrapper
    # BEFORE FSDP2 wrapping if needed.
    # model.gradient_checkpointing_enable()
    # torch.compile(model)
    return model

def setup_training_components(model, fsdp_mesh, tp_mesh, **kwargs):
    from transformers import get_scheduler
    
    # Using FSDP2 wrapper
    log_rank_0("Using FSDP2 wrapper")
    model = wrap_fsdp2(model, fsdp_mesh, tp_mesh)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=kwargs['learning_rate'],
        betas=(0.9, 0.95),
        weight_decay=0.0,
        # TODO: fix this to be dynamic, or handle this gracefully so that user doesn't have to worry about it, and still get the optimized version.
        foreach=False if tp_mesh.size(0) > 1 else None,
        fused=False if tp_mesh.size(0) > 1 else None,
    )
    lr_scheduler = get_scheduler(
        name=kwargs['lr_scheduler'],
        optimizer=optimizer,
        num_warmup_steps=kwargs['num_warmup_steps'],
    )
    lr_scheduler.split_batches = True
    lr_scheduler.step() #the scheduler starts at 0 and there's no learning.
    return model, optimizer, lr_scheduler

