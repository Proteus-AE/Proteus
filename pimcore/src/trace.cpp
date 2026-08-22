#include "pimcore/trace.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace pimcore {

std::vector<Command> read_trace(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open trace: " + path);
  std::vector<Command> out;
  std::string line;
  size_t lineno = 0;
  while (std::getline(f, line)) {
    ++lineno;
    std::istringstream is(line);
    std::string tok;
    if (!(is >> tok) || tok[0] == '#') continue;
    Command c;
    c.kind = command_from_string(tok);
    switch (c.kind) {
      case CommandKind::MODE: {
        std::string m;
        is >> m;
        c.mode = (m == "broadcast") ? ConnectivityMode::BROADCAST
                                    : ConnectivityMode::DIRECT;
        break;
      }
      case CommandKind::ACT_AB:
        is >> c.row;
        break;
      case CommandKind::RDMAC_AB:
      case CommandKind::WR_AB:
        is >> c.row >> c.col;
        break;
      case CommandKind::ACT:
      case CommandKind::RD:
      case CommandKind::PRE:
        is >> c.row >> c.col >> c.bank;
        break;
      case CommandKind::PRE_AB:
      case CommandKind::BARRIER:
        break;
      default:
        throw std::runtime_error("bad trace line " + std::to_string(lineno) +
                                 ": " + line);
    }
    out.push_back(c);
  }
  return out;
}

void write_trace(const std::string& path, const std::vector<Command>& cmds) {
  std::ofstream f(path);
  if (!f) throw std::runtime_error("cannot write trace: " + path);
  for (const Command& c : cmds) f << c.format() << "\n";
}

}  // namespace pimcore
