#include "pimcore/types.hpp"

#include <stdexcept>

namespace pimcore {

const char* to_string(CommandKind k) {
  switch (k) {
    case CommandKind::ACT_AB:   return "ACT_AB";
    case CommandKind::RDMAC_AB: return "RDMAC_AB";
    case CommandKind::WR_AB:    return "WR_AB";
    case CommandKind::PRE_AB:   return "PRE_AB";
    case CommandKind::ACT:      return "ACT";
    case CommandKind::RD:       return "RD";
    case CommandKind::PRE:      return "PRE";
    case CommandKind::MODE:     return "MODE";
    case CommandKind::BARRIER:  return "BARRIER";
    default:                    return "INVALID";
  }
}

CommandKind command_from_string(const std::string& s) {
  if (s == "ACT_AB")   return CommandKind::ACT_AB;
  if (s == "RDMAC_AB") return CommandKind::RDMAC_AB;
  if (s == "WR_AB")    return CommandKind::WR_AB;
  if (s == "PRE_AB")   return CommandKind::PRE_AB;
  if (s == "ACT")      return CommandKind::ACT;
  if (s == "RD")       return CommandKind::RD;
  if (s == "PRE")      return CommandKind::PRE;
  if (s == "MODE")     return CommandKind::MODE;
  if (s == "BARRIER")  return CommandKind::BARRIER;
  return CommandKind::INVALID;
}

const char* to_string(ConnectivityMode m) {
  return m == ConnectivityMode::DIRECT ? "direct" : "broadcast";
}

ArbitrationPolicy arbitration_from_string(const std::string& s) {
  if (s == "pim-priority")  return ArbitrationPolicy::PIM_PRIORITY;
  if (s == "host-priority") return ArbitrationPolicy::HOST_PRIORITY;
  if (s == "interleave")    return ArbitrationPolicy::INTERLEAVE;
  throw std::runtime_error("unknown arbitration policy: " + s);
}

std::string Command::format() const {
  std::string out = to_string(kind);
  switch (kind) {
    case CommandKind::ACT_AB:
      out += " " + std::to_string(row);
      break;
    case CommandKind::RDMAC_AB:
    case CommandKind::WR_AB:
      out += " " + std::to_string(row) + " " + std::to_string(col);
      break;
    case CommandKind::ACT:
    case CommandKind::RD:
    case CommandKind::PRE:
      out += " " + std::to_string(row) + " " + std::to_string(col) + " " +
             std::to_string(bank);
      break;
    case CommandKind::MODE:
      out += std::string(" ") + to_string(mode);
      break;
    default:
      break;
  }
  return out;
}

}  // namespace pimcore
