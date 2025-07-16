# Tensor Parallelism Ramp
# Ref: https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html

# Let's start with why we need Tensor Parallelism, when we have DeepSpeed ZeRO and FSDP.
# Both of these methods are for Data Parallelism. They shard model weights across GPUs, and
# for different stages offload optimizer, gradient and parameters to CPU ram.
# This in order to free up GPU ram, that is not being actively used, and only use it when required.
# The problem is when the model size gets too large, mainly by increasing the weight matrix size,
# which in turn increases the memory required to store intermediate activations for backward pass.
# This is exactly where tensor parallelism comes into play.

# Lets perform single forward and backward pass of Qwen/Qwen3-1.7B using both tensor parallelism and data paralleism
# to see how to use them independently and in conjunction

# To run this script, we need to use the following command:
# torchrun --nnodes=1 --nproc_per_node=8 tensor_parallelism_getting_started.py
# 8 GPUs are required for this script.
# And we need to setup_distributed() function to initialize the distributed process group.
import torch
import os
from torch.distributed import init_process_group

def setup_distributed() -> int:
  init_process_group(backend="nccl")
  local_rank = int(os.environ["LOCAL_RANK"])
  torch.cuda.set_device(local_rank)
  return local_rank

local_rank = setup_distributed()
device = torch.device(f"cuda:{local_rank}")

# We will be using NousResearch/hermes function calling dataset. So let's start by creating dataset
# You can check it out here: https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1

# For starters lets download and transform it to expected format, i.e.:
# {"messages": [{"role": "user","content": "<input 1>"},{"role": "assistant","content": "<output 1>"}]}
# {"messages": [{"role": "user","content": "<input 2>"},{"role": "assistant","content": "<output 2>"}]}
import json
from datasets import load_dataset

# Only rank 0 will download the dataset and process it.
if local_rank == 0:
    ds = load_dataset("NousResearch/hermes-function-calling-v1", "func_calling_singleturn")
    new_ds = []
    for convo in ds['train']['conversations']:
        tmp = []
        for turn in convo:
            role = 'user' if turn['from'] in ['system', 'human'] else 'assistant'
            if len(tmp) and tmp[-1]['role'] == role: tmp[-1]['content'] += '\n\n' + turn['value']
            else: tmp.append({'role': role, 'content': turn['value']})
        new_ds.append({'messages': tmp})

    processed_data_fn = 'play.jsonl'
    with open(processed_data_fn, 'w') as f:
        for convo in new_ds: f.write(json.dumps(convo) + '\n')
# After this you should have a JSONl file that you can checkout by `less play.jsonl'

# Once, we get the dataset into the required format, we can use process_data.py to tokenize
# the dataset. And make it ready to be trained on. 
# we need scripts from root directory to be in the path.
import sys; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import process_data
model_name = "Qwen/Qwen3-1.7B"
tokenized_data_fn = 'tokenized_qwen3_1.7B_play.jsonl'
# Only rank 0 will tokenize the dataset.
if local_rank == 0:
    process_data.process_data(
        input_jsonl = processed_data_fn,
        output_jsonl= tokenized_data_fn,
        model_name_or_path=model_name,
        max_sample_num_tokens=128000,  # max sample length. samples longer than this will be removed 
        string_for_printing_masks="<|mAsK|>",  # post-tokenization samples are written to stdout.
    )
# this will tokenize, filter out samples longer than 128000 tokens, and add labels to the dataset
# It will also print 2 random samples
# original messages: original message from the input jsonl,
"""
original messages:
[{'role': 'user', 'content': "You are a function calling AI model.
You are provided with function signatures within <tools> </tools> XML tags.
You may call one or more functions to assist with the user query.
Don't make assumptions about what values to plug into functions.
<tools> [...]
"""
# input_ids decoded: will print the decoded input_ids. You can see the chat template being applied,
"""
input_ids decoded:
<|im_start|>user
You are a function calling AI model. You are provided with function
signatures within <tools> </tools> XML tags. You may call one or
more functions to assist with the user query. Don't make assumptions
about what values to plug into functions.
<tools> [...]
"""
# labels: will print the labels, and the masked tokens will be replaced with <|mAsK|>
#   we only backprop on the assistant tokens.
"""
labels:
<|mAsK|><|mAsK|><|mAsK|><|mAsK|><|mAsK|><|mAsK|><|mAsK|><|mAsK|><|mAsK|><|mAsK|><|mAsK|><|mAsK|><|mAsK|><|mAsK|><|mAsK|>
[...]
<think>

</think>

<tool_call>
{'arguments': {'reviews_json_path': '/data/reviews/financial_services_reviews.json', 'categories': ['mortgages', 'personal loans', 'credit cards', 'investment accounts'], 'output_format': 'json_schema'}, 'name': 'classify_financial_reviews'}
</tool_call>
<tool_call>
{'arguments': {'categories': ['mortgages', 'personal loans', 'credit cards', 'investment accounts']}, 'name': 'generate_financial_review_schema'}
</tool_call>
<|im_end|>
"""
# we want all processes to pause here and wait for all processes to be ready
torch.distributed.barrier()
# this will ensure that the data is tokenized and ready to be used by all processes

# ===============================================
# Now before we start with full finetuning, let's start
# with a single forward and backward pass to verify that the setup is working.

# Starting with Data Parallelism using FSDP2
import sampler
dataset = sampler.JsonlDataset(tokenized_data_fn)
def to_input_example(sample):
    return {
        'input_ids': sample['input_ids'].to(device).unsqueeze(0),
        'labels': sample['labels'].to(device).unsqueeze(0),
        'position_ids': torch.arange(len(sample['labels'])).to(device).unsqueeze(0),
    }

# Setup the model
import setup_model_for_training
model = setup_model_for_training.setup_model(model_name_or_path=model_name, use_liger_kernels=False).to(device)
# we will create a deepcopy of the model, so that we can start from a clean slate, when we do TP and DP + TP.
import copy
dp_model = copy.deepcopy(model)
fsdp_model = setup_model_for_training.wrap_fsdp2(dp_model)
optimizer = torch.optim.AdamW(
    fsdp_model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.95),
    weight_decay=0.0,
)
if local_rank == 0 or local_rank == 1: example = to_input_example(dataset[0])
elif local_rank == 2 or local_rank == 3: example = to_input_example(dataset[1])
elif local_rank == 4 or local_rank == 5: example = to_input_example(dataset[2])
elif local_rank == 6 or local_rank == 7: example = to_input_example(dataset[3])

output = fsdp_model(**example)
loss = output.loss.float().sum()
# for debugging, we will perform a all_reduce on the loss, and print the loss sum
with torch.no_grad():
    reduced_loss = loss.clone()
    torch.distributed.all_reduce(reduced_loss, op=torch.distributed.ReduceOp.SUM)
    print(f'\033[38;5;208m DP - rank {local_rank} | loss sum {reduced_loss.item() / 2}\033[0m') # divide by 2 because we are running an example twice.
    # >>> DP - rank 0 | loss sum 260.198394775390 ... and same for other ranks
loss.backward(); optimizer.step(); fsdp_model.zero_grad()
# delete the model from memory and clear the cache
del fsdp_model, optimizer; torch.cuda.empty_cache()

# ===============================================
# let's now try this with tensor parallelism
# we will start by defining the device mesh
# A device mesh is a n-dimensional grid of devices.
# In our case, we will have 8 GPUs, and we will use 1 dimension to represent the tensor parallelism.
from torch.distributed.device_mesh import init_device_mesh
device_mesh = init_device_mesh("cuda", (8,), mesh_dim_names=["tp"])
tp_mesh = device_mesh['tp']
if local_rank == 0: print(f'tp_mesh: {tp_mesh}')
# >>> tp_mesh: DeviceMesh('cuda', [0, 1, 2, 3, 4, 5, 6, 7], mesh_dim_names=('tp',)

# now we will have to create a plan for how to shard weight matrices across tensor parallel groups
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
tp_block_plan = {
    # mlp layers
    "mlp.gate_proj": ColwiseParallel(),
    "mlp.up_proj": ColwiseParallel(),
    "mlp.down_proj": RowwiseParallel(),
}
# like the name suggests, ColwiseParallel will shard the weight matrix across columns, and
# RowwiseParallel will shard the weight matrix across rows.

# before applying the plan, we will make a deepcopy of the model
# so that we can start from a clean slate, when we do DP + TP.
# now we can apply this plan to the individual transformer blocks in the model
from torch.distributed.tensor.parallel import parallelize_module
tp_model = copy.deepcopy(model)
for layer_id, transformer_block in enumerate(tp_model.model.layers):
    parallelize_module(module=transformer_block, device_mesh=tp_mesh, parallelize_plan=tp_block_plan)
    # inorder to shard the model, while keeping the api intact, this function will change the data type
    # from torch.Tensor to torch.DTensor.

# now we can run a forward pass on those 4 examples, and then perform an all_reduce on the loss
batch_examples = [to_input_example(dataset[i]) for i in range(4)]
optimizer = torch.optim.AdamW(tp_model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.0, foreach=False, fused=False)
# foreach and fused are methods to speed up the training. by default, the most optimized method is used.
# this method doesn't work when you have mix of torch.Tensor and torch.DTensor. so we will set them to False.
total_loss = 0.0
for x in batch_examples: total_loss += tp_model(**x).loss.float().sum()
with torch.no_grad():
    reduced_loss = total_loss.clone()
    torch.distributed.all_reduce(reduced_loss, op=torch.distributed.ReduceOp.SUM)
    print(f'\033[38;5;208m TP - rank {local_rank} | loss sum {reduced_loss.item() / 8}\033[0m') # divide by 8 because we are using 8 GPUs.
    # >>> TP - rank 0 | loss sum 262.038604736328 ... and same for other ranks
    # you may notice difference in loss sum between DP and TP. If you set the model dtype to
    # float32, the loss sum will be the same. It is because of the fp32 -> fp16/bf16 precision loss.
# now we can run a backward pass and update the model parameters
total_loss.backward(); optimizer.step(); tp_model.zero_grad()
# delete the model from memory and clear the cache
del tp_model, optimizer; torch.cuda.empty_cache()

# ===============================================
# now lets do a forward + backward pass using FSDP + TP.
# you can create a device mesh with 4 data parallel groups, and 2 tensor parallel groups
world_mesh = init_device_mesh("cuda", (4, 2), mesh_dim_names=["fsdp", "tp"])
fsdp_mesh = world_mesh['fsdp']
tp_mesh = world_mesh['tp']
if local_rank == 0: print(f'fsdp + tp mesh: {world_mesh}')
# >>> fsdp + tp mesh: DeviceMesh('cuda', [[0, 1], [2, 3], [4, 5], [6, 7]], mesh_dim_names=('fsdp', 'tp')
# here, the inner list represents tensor parallel groups, and the outer list represents data parallel groups.
# we will feed in same example to processes within same data parallel group, i.e. [0,1] or [2,3] or [4,5] or [6,7]

fsdp_tp_model = copy.deepcopy(model)
for transformer_block in fsdp_tp_model.model.layers:
    parallelize_module(module=transformer_block, device_mesh=tp_mesh, parallelize_plan=tp_block_plan)

# setup fsdp2 model
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper as ptd_checkpoint_wrapper

fsdp_tp_model.config.use_cache = False  # disable cache for HF model
for idx, block in enumerate(fsdp_tp_model.model.layers):
    fsdp_tp_model.model.layers[idx] = ptd_checkpoint_wrapper(block, preserve_rng_state=False)  # activation checkpoint each block
mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, output_dtype=torch.bfloat16)
# FSDP2 wrap each block
for idx, block in enumerate(fsdp_tp_model.model.layers):
    reshard = idx < len(fsdp_tp_model.model.layers) - 1
    fully_shard(block, mesh=fsdp_mesh, mp_policy=mp_policy, reshard_after_forward=reshard)
fully_shard(fsdp_tp_model, mesh=fsdp_mesh, mp_policy=mp_policy, reshard_after_forward=True)  # wrap the full model

optimizer = torch.optim.AdamW(fsdp_tp_model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.0, foreach=False, fused=False)
output = fsdp_tp_model(**example)
loss = output.loss.float().sum()
with torch.no_grad():
    reduced_loss = loss.clone()
    torch.distributed.all_reduce(reduced_loss, op=torch.distributed.ReduceOp.SUM)
    print(f'\033[38;5;208m FSDP + TP - rank {local_rank} | loss sum {reduced_loss.item() / 2}\033[0m') # divide by 2 because we are adding loss twice per example, due to TP.
    # >>> FSDP + TP - rank 0 | loss sum 262.120819091796 ... and same for other ranks
loss.backward(); optimizer.step(); fsdp_tp_model.zero_grad()
# delete the model from memory and clear the cache
del fsdp_tp_model, optimizer; torch.cuda.empty_cache()
torch.distributed.destroy_process_group()
