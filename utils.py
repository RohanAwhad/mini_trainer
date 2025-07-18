from datetime import timedelta
import importlib
import inspect
import logging
import os
from typing import Any

import torch
from torch.distributed import is_initialized, get_rank
from rich.logging import RichHandler

def get_caller(num_frames=1):
    frame = inspect.currentframe().f_back
    for _ in range(num_frames - 1):
        frame = frame.f_back
    file_name = frame.f_code.co_filename
    line_number = frame.f_lineno
    return f"In {file_name}, line {line_number}"

def log_rank_0(msg, include_caller=False, rank=None, to_print=True):
    if rank is None:
        rank = get_rank() if is_initialized() else 0
    if rank <= 0:
        if include_caller:
            msg = f"{get_caller(num_frames=2)}: {msg}"
        if to_print:
            print(msg)
        else:
            logging.info(msg)

def setup_logger(level="DEBUG"):
    logging.basicConfig(
        level=level, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
    )

def patch_target_module(
    to_patch: str,
    replace_with: Any,
):
    to_patch = to_patch.split(".")
    assert len(to_patch) > 1, "must have an object to patch"

    to_patch, obj_name_to_patch = to_patch[:-1], to_patch[-1]
    to_patch = ".".join(to_patch)
    source = importlib.import_module(to_patch)
    setattr(source, obj_name_to_patch, replace_with)

def init_distributed_environment(tp_size: int = 1):
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group("nccl", timeout=timedelta(minutes=180))
    tensor = torch.ByteTensor([False]).cuda()
    torch.distributed.all_reduce(tensor)
    torch.distributed.barrier()

    # Create device mesh for distributed training
    from torch.distributed.device_mesh import init_device_mesh
    import torch.distributed as dist
    world_size = dist.get_world_size()
    assert tp_size >= 1 and tp_size <= 8, f"expected tp_size to be between [1, 8], but got '{tp_size}'"
    assert world_size % tp_size == 0, f"expected world_size to be divisible by tensor parallel size, but got '{world_size} % {tp_size} == {world_size % tp_size}'"

    fsdp_size = world_size // tp_size
    world_mesh = init_device_mesh("cuda", (fsdp_size, tp_size), mesh_dim_names=("fsdp", "tp"))
    return world_mesh['fsdp'], world_mesh['tp']
