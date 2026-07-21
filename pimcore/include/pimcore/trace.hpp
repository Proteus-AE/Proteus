// Command-trace serialization (text format shared with the Python layer):
//   MODE broadcast
//   ACT_AB   <row>
//   RDMAC_AB <row> <col>
//   WR_AB    <row> <col>
//   PRE_AB
//   RD       <row> <col> <bank>
//   BARRIER
#pragma once

#include <string>
#include <vector>

#include "pimcore/types.hpp"

namespace pimcore {

std::vector<Command> read_trace(const std::string& path);
void write_trace(const std::string& path, const std::vector<Command>& cmds);

}  // namespace pimcore
