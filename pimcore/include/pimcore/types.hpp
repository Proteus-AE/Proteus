// PimCore: standalone command-level near-bank PIM memory simulator.
// Basic types shared across the simulator.
#pragma once

#include <cstdint>
#include <string>

namespace pimcore {

using ns_t = double;          // simulation time, nanoseconds
using addr_t = uint64_t;

// ------------------------------------------------------------------ //
// DRAM command set. All-bank commands (_AB) are issued once on the
// command/address bus and executed by every bank of the channel subject to
// each bank's local timing state; single-bank commands model host traffic.
// ------------------------------------------------------------------ //
enum class CommandKind : uint8_t {
  ACT_AB,        // all-bank row activate
  RDMAC_AB,      // all-bank column read feeding the near-bank MAC path
  WR_AB,         // all-bank column write (in-place KV append)
  PRE_AB,        // all-bank precharge
  ACT,           // single-bank activate (host path)
  RD,            // single-bank read (host path)
  PRE,           // single-bank precharge
  MODE,          // connectivity mode-register update
  BARRIER,       // stream synchronization marker
  INVALID
};

const char* to_string(CommandKind k);
CommandKind command_from_string(const std::string& s);

// Bank-to-PE connectivity of the reconfigurable datapath.
enum class ConnectivityMode : uint8_t {
  DIRECT,        // one-to-one: each burst feeds the bank-local PE only
  BROADCAST      // one-to-many: each burst feeds every PE of the bank group
};

const char* to_string(ConnectivityMode m);

// Host/PIM arbitration policy for the shared channel (unified memory path).
enum class ArbitrationPolicy : uint8_t {
  PIM_PRIORITY,  // host requests only fill scheduling gaps
  HOST_PRIORITY, // host requests preempt the next all-bank slot
  INTERLEAVE     // round-robin slot sharing when both are backlogged
};

ArbitrationPolicy arbitration_from_string(const std::string& s);

// A single command as it appears in a trace or a generated stream.
struct Command {
  CommandKind kind = CommandKind::INVALID;
  int row = 0;
  int col = 0;
  int bank = -1;                       // target bank for single-bank kinds
  ConnectivityMode mode = ConnectivityMode::DIRECT;  // for MODE
  std::string format() const;
};

}  // namespace pimcore
