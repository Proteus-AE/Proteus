"""CXL 3.0 scale-out fabric model (Sec. IV-B "Scale-out Fabric").

Message-level model of the single-tier switched topology: peer-to-peer
CXL.mem writes into double-buffered staging regions followed by doorbell
notifications, entirely bypassing the host. Transfer latency decomposes into
port round-trip, retimer, switch traversal, and memory-controller/DRAM
components; link bandwidth is shared among concurrently active devices and
oversubscribed transfers serialize at the switch.

Two collective patterns appear in a decode iteration:

*within a group*  each transformer block ends with tensor-parallel reductions
                  (attention output projection, FFN output). A ring collective
                  over ``N`` devices moves ``2 (N-1)/N`` times the activation
                  tensor per device and per collective. For Llama-3.1-70B with
                  d = 8192 and b = 32 over eight devices this is 147 MB per
                  device and decode iteration, 1.15 ms of payload over one x16
                  port -- the same order as the 1.60 ms the device spends
                  streaming its 17.5 GB weight shard at the sustained
                  all-bank rate, and it replaces the 140 GB that replication
                  or pipelining would fetch at equal batch.

*across groups*   a single ``b x d_model`` activation transfer per pipeline
                  boundary: 512 KB, about 4 us, negligible against
                  millisecond-scale token latency.
"""
from dataclasses import dataclass


@dataclass
class Transfer:
    src: int
    dst: int
    bytes: float
    start_ns: float
    end_ns: float


class CxlFabric:
    def __init__(self, ic_cfg):
        # B/s per direction on one x16 port
        self.link_bw = ic_cfg.get("link_bytes_per_s_per_dir",
                                  ic_cfg.get("link_gbps_per_dir"))
        self.lat_ns = ic_cfg["end_to_end_latency_ns"]
        self.doorbell_ns = ic_cfg.get("doorbell_ns", 0.0)

    # ---- point-to-point ------------------------------------------------ #
    def transfer_ns(self, nbytes, active_pairs=1):
        """Latency of one activation transfer; concurrent transfers through
        the switch share link bandwidth when oversubscribed."""
        eff_bw = self.link_bw / max(1, active_pairs)
        return self.lat_ns + self.doorbell_ns + nbytes / eff_bw * 1e9

    # ---- tensor-parallel collectives ----------------------------------- #
    @staticmethod
    def tp_allreduce_bytes(act_bytes, n_devices, collectives_per_layer=2):
        """Bytes each device sends per layer for the block's AllReduces.

        A bandwidth-optimal ring AllReduce over ``N`` devices transfers
        ``2 (N-1)/N`` times the tensor per participant (reduce-scatter plus
        all-gather)."""
        if n_devices <= 1:
            return 0.0
        return collectives_per_layer * 2.0 * (n_devices - 1) / n_devices \
            * act_bytes

    def tp_allreduce_ns(self, act_bytes, n_devices, collectives_per_layer=2):
        """Wall-clock cost of one block's AllReduces on one x16 port.

        The ring is chunk-pipelined, so the payload of a collective streams at
        link rate, but every chunk still traverses the ``2 (N-1)`` steps of
        the reduce-scatter and all-gather phases and pays the port-to-port
        latency at each hop:

            T = 2 (N-1) * lat  +  2 (N-1)/N * S / B.

        For Llama-3.1-70B (d = 8192, b = 32, N = 8, 80 blocks, two collectives
        per block) this is 1.15 ms of payload plus 0.37 ms of hop latency per
        device and decode iteration. The same expression is applied to every
        evaluated system from its own link configuration."""
        if n_devices <= 1:
            return 0.0
        vol = self.tp_allreduce_bytes(act_bytes, n_devices,
                                      collectives_per_layer)
        hops = 2.0 * (n_devices - 1) * collectives_per_layer
        return vol / self.link_bw * 1e9 + hops * self.lat_ns

    # ---- pipeline timeline --------------------------------------------- #
    def iteration_timeline(self, n_stages, t_stage_ns, act_bytes):
        """Steady-state pipeline timeline of one decode iteration: group i
        computes [i*T, (i+1)*T), then writes activations to group i+1 and
        rings its doorbell. All stage pairs transfer disjointly in time, so
        no switch contention arises in steady state."""
        events = []
        for i in range(n_stages):
            c0 = i * t_stage_ns
            c1 = c0 + t_stage_ns
            events.append(("compute", i, c0, c1))
            if i + 1 < n_stages:
                tr = self.transfer_ns(act_bytes)
                events.append(("transfer", i, c1, c1 + tr))
                events.append(("doorbell", i + 1, c1 + tr, c1 + tr))
        return events

    @staticmethod
    def render(events, width=64):
        """ASCII Gantt of an iteration timeline."""
        t_end = max(e[3] for e in events)
        lines = []
        stages = sorted({e[1] for e in events})
        for s in stages:
            row = [" "] * width
            for kind, st, a, b in events:
                if st != s:
                    continue
                i0 = int(a / t_end * (width - 1))
                i1 = max(i0 + 1, int(b / t_end * (width - 1)))
                ch = {"compute": "#", "transfer": ">", "doorbell": "!"}[kind]
                for i in range(i0, min(i1, width)):
                    if row[i] == " " or ch == "!":
                        row[i] = ch
            lines.append(f"  grp{s:<2}|{''.join(row)}|")
        lines.append(f"        0{' ' * (width - 12)}{t_end/1e6:9.3f} ms")
        lines.append("        # compute   > activation transfer   ! doorbell")
        return "\n".join(lines)
