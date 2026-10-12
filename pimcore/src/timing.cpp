#include "pimcore/timing.hpp"

#include <stdexcept>

namespace pimcore {

// The JEDEC parameters, the organization and the per-command energies all
// come from configs/memory/<name>.yaml; the fallbacks below are the ones the
// Python layer applies to the same file (proteus_sim/dram/timing.py,
// proteus_sim/memory.py), so the two readers agree on an incomplete file as
// well as on a complete one.

TimingParams TimingParams::from_config(const ConfigNode& mem) {
  TimingParams t;
  t.tRCD = mem.get_double("tRCD_ns");
  t.tRP = mem.get_double("tRP_ns");
  t.tRAS = mem.get_double("tRAS_ns");
  t.tRC = mem.get_double("tRC_ns");
  t.tCCD_L = mem.get_double("tCCD_L_ns");
  t.tCCD_S = mem.get_double("tCCD_S_ns");
  // The near-bank read path leaves the bank over a bank-private connection
  // and never enters the bank-group I/O, so its column cycle is set by the
  // bank core alone; a substrate whose datapath is fixed one-to-one has no
  // such path and declares tCCD_PIM_ns = tCCD_L_ns.
  t.tCCD_PIM = mem.get_double("tCCD_PIM_ns", t.tCCD_L / 2.0);
  t.tRTP = mem.get_double("tRTP_ns");
  t.tWR = mem.get_double("tWR_ns");
  t.tWTR = mem.get_double("tWTR_ns", 0.0);
  t.tRRD_L = mem.get_double("tRRD_L_ns");
  t.tRRD_S = mem.get_double("tRRD_S_ns");
  t.tFAW = mem.get_double("tFAW_ns");
  t.tREFI = mem.get_double("tREFI_ns");
  t.tRFCab = mem.get_double("tRFCab_ns");
  t.ca_cmd_ns = mem.get_double("ca_cmd_ns", 1.0);
  t.row_bytes = static_cast<int>(mem.get_int("row_bytes"));
  double burst_length = mem.get_double("burst_length");
  t.burst_bytes =
      static_cast<int>(mem.get_double("io_width") * burst_length / 8.0);
  t.burst_ns = burst_length / mem.get_double("data_rate_mtps") * 1e3;
  t.allbank_rcd = t.tRCD;
  t.allbank_ras = t.tRAS;
  t.allbank_rp = t.tRP;
  if (mem.has("allbank")) {
    const ConfigNode& ab = mem.at("allbank");
    t.allbank_rcd = ab.get_double("act_rcd_ns", t.tRCD);
    t.allbank_ras = ab.get_double("ras_ns", t.tRAS);
    t.allbank_rp = ab.get_double("rp_ns", t.tRP);
    t.allbank_faw_exempt = ab.get_string("faw_exempt", "true") != "false";
  }
  t.validate();
  return t;
}

void TimingParams::validate() const {
  auto require = [](bool ok, const char* what) {
    if (!ok) throw std::runtime_error(std::string("timing violation: ") + what);
  };
  require(tRC >= tRAS + tRP - 1e-9, "tRC >= tRAS + tRP");
  require(tRAS >= tRCD, "tRAS >= tRCD");
  require(tCCD_L >= tCCD_S, "tCCD_L >= tCCD_S");
  require(tCCD_L >= tCCD_PIM, "tCCD_L >= tCCD_PIM");
  require(tRRD_L >= tRRD_S, "tRRD_L >= tRRD_S");
  require(tFAW >= tRRD_S, "tFAW covers at least one tRRD_S");
  require(burst_ns > 0 && ca_cmd_ns > 0, "positive bus cadences");
  require(burst_bytes > 0, "positive burst payload");
  require(row_bytes % burst_bytes == 0, "row is a whole number of bursts");
}

Geometry Geometry::from_config(const ConfigNode& mem) {
  Geometry g;
  g.channels = static_cast<int>(mem.get_int("packages_per_device") *
                                mem.get_int("channels_per_package"));
  g.dies_per_channel = static_cast<int>(mem.get_int("dies_per_channel"));
  g.bankgroups_per_die = static_cast<int>(mem.get_int("bankgroups_per_die"));
  g.banks_per_bankgroup = static_cast<int>(mem.get_int("banks_per_bankgroup"));
  g.pe_lanes = static_cast<int>(mem.get_int("pe_lanes"));
  g.pe_freq_ghz = mem.get_double("pe_freq_ghz");
  g.pe_fifo_depth = static_cast<int>(mem.get_int("pe_operand_fifo"));
  g.broadcast_fanout =
      static_cast<int>(mem.get_int("broadcast_fanout", g.banks_per_bankgroup));
  return g;
}

EnergyTable EnergyTable::from_config(const ConfigNode& mem) {
  const ConfigNode& c = mem.at("command_energy_pj");
  EnergyTable e;
  e.act_pre_pj = c.get_double("act_pre");
  e.rd_burst_array_pj = c.get_double("rd_burst_array");
  e.wr_burst_array_pj = c.get_double("wr_burst_array", e.rd_burst_array_pj);
  e.rd_burst_io_pj = c.get_double("rd_burst_io");
  e.bg_broadcast_pj = c.get_double("bg_broadcast", 0.0);
  e.mac_op_pj = c.get_double("mac_op");
  e.mode_switch_pj = c.get_double("mode_switch");
  e.refresh_ab_pj = c.get_double("refresh_ab");
  return e;
}

}  // namespace pimcore
