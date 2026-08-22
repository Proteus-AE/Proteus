// Command-trace serialization (text format shared with the Python layer):
//   MODE broadcast
//   ACT_AB   <row>
//   RDMAC_AB <row> <col>
//   WR_AB    <row> <col>
//   PRE_AB
//   ACT      <row> <col> <bank>       # host (xPU) single-bank traffic
//   RD       <row> <col> <bank>
//   PRE      <row> <col> <bank>
//   BARRIER
//
// This is the whole command set: a write is an all-bank operation and
// refresh is scheduler-driven, so neither a single-bank WR nor an explicit
// refresh command appears in a trace, and a line carrying one is rejected.
#pragma once

#include <string>
#include <vector>

#include "pimcore/types.hpp"

namespace pimcore {

std::vector<Command> read_trace(const std::string& path);
void write_trace(const std::string& path, const std::vector<Command>& cmds);

}  // namespace pimcore
