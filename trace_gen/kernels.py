"""Kernel -> all-bank command-stream lowering.

Layout convention (Sec. IV-B "Memory-Hierarchy-Aligned Parallelism"):
shared matrices are column-striped across the banks of a channel, so one
all-bank RDMAC sweeps one 32 B slice from every bank concurrently; the K and
V caches are striped in orthogonal dimensions so decode-appended entries
preserve load balance.

Every kernel emits (per channel):
  ACT_AB <row>; RDMAC_AB <row> <col> x (row_bytes/32B); PRE_AB; ...
optionally preceded by a MODE command when the scheduler selected a
connectivity switch (Sec. IV-C "Lightweight Reconfiguration").
"""
import math

from proteus_sim.dram.commands import (Command, ACT_AB, RDMAC_AB, WR_AB,
                                       PRE_AB, MODE)


def layout_rows_per_bank(operand_bytes, mem_cfg, channels=None):
    """Rows each bank must stream for `operand_bytes` striped over a device
    (or over `channels` channels of it)."""
    channels = channels or (mem_cfg["packages_per_device"]
                            * mem_cfg["channels_per_package"])
    banks = channels * mem_cfg["dies_per_channel"] \
        * mem_cfg["bankgroups_per_die"] * mem_cfg["banks_per_bankgroup"]
    per_bank = operand_bytes / banks
    return max(1, math.ceil(per_bank / mem_cfg["row_bytes"]))


def _stream_rows(rows, bursts_per_row, base_row=0):
    cmds = []
    for r in range(rows):
        cmds.append(Command(ACT_AB, row=base_row + r))
        for c in range(bursts_per_row):
            cmds.append(Command(RDMAC_AB, row=base_row + r, col=c))
        cmds.append(Command(PRE_AB))
    return cmds


def gemv_trace(rows_per_bank, mem_cfg, set_mode=True):
    """Reuse-free GEMV: one pass over the striped matrix, direct mode."""
    bursts = mem_cfg["row_bytes"] // 32
    cmds = [Command(MODE, arg="direct")] if set_mode else []
    cmds += _stream_rows(rows_per_bank, bursts)
    return cmds


def skinny_gemm_trace(rows_per_bank, n_vectors, mem_cfg, mode="broadcast",
                      fanout=4, set_mode=True):
    """Shared-operand skinny-GEMM with `n_vectors` concurrent input vectors.

    direct    : no inter-PE reuse -> the matrix is re-streamed per vector.
    broadcast : each burst feeds `fanout` PEs -> ceil(n/fanout) passes
                (Sec. IV-C "Broadcasting Mode").
    """
    bursts = mem_cfg["row_bytes"] // 32
    passes = n_vectors if mode == "direct" else math.ceil(n_vectors / fanout)
    cmds = [Command(MODE, arg=mode)] if set_mode else []
    for p in range(passes):
        cmds += _stream_rows(rows_per_bank, bursts)
    return cmds


def attention_trace(ctx_tokens, kv_bytes_per_token_layer, group_size,
                    mem_cfg, mode="broadcast", fanout=4, channels_per_head=8,
                    set_mode=True, kv_append=False):
    """Decode attention of one head-group over its resident KV slice.

    The KV cache of `ctx_tokens` tokens is striped over the banks of
    ``channels_per_head`` channels (Fig. 7). GQA/MLA supplies
    `group_size` queries sharing the same KV operand:
      direct    -> the KV slice is re-streamed once per query,
      broadcast -> ceil(group_size/fanout) passes.
    """
    kv_bytes = ctx_tokens * kv_bytes_per_token_layer
    rows = layout_rows_per_bank(kv_bytes, mem_cfg, channels=channels_per_head)
    passes = group_size if mode == "direct" \
        else math.ceil(group_size / fanout)
    bursts = mem_cfg["row_bytes"] // 32
    cmds = [Command(MODE, arg=mode)] if set_mode else []
    for p in range(passes):
        cmds += _stream_rows(rows, bursts)
    if kv_append:
        # Decode appends this step's K/V entries in place at the stripe tail
        # (Sec. IV-A "Execution Flow"): the per-bank share of one token's KV
        # is below one burst, so a single all-bank write burst set suffices.
        cmds.append(Command(WR_AB, row=rows, col=0))
        cmds.append(Command(PRE_AB))
    return cmds
