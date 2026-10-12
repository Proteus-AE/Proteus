// Physical address mapping for host-path traffic.
//
// Decodes flat physical addresses into (channel, die, bank group, bank,
// row, column) coordinates under a configurable interleaving order. The
// default order channel -> bank group -> bank -> column -> row spreads
// consecutive cache lines across channels and bank groups, the layout the
// unified xPU/PIM address space uses for activation and weight regions.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "pimcore/timing.hpp"
#include "pimcore/types.hpp"

namespace pimcore {

struct Coordinates {
  int channel = 0;
  int die = 0;
  int bankgroup = 0;   // within die
  int bank = 0;        // within bank group
  int row = 0;
  int col = 0;         // burst index within the row

  int flat_bank(const Geometry& g) const {
    return (die * g.bankgroups_per_die + bankgroup) * g.banks_per_bankgroup +
           bank;
  }
};

class AddressMapper {
 public:
  AddressMapper(const Geometry& geom, const TimingParams& timing,
                const std::string& order = "ch-bg-bank-col-row");

  Coordinates decode(addr_t addr) const;
  addr_t encode(const Coordinates& c) const;

  int bursts_per_row() const { return bursts_per_row_; }
  const std::string& order() const { return order_; }

 private:
  struct Field {
    char id;        // 'C' channel, 'D' die, 'G' bg, 'B' bank, 'R' row, 'O' col
    int size;       // number of values
  };
  std::vector<Field> fields_;   // least-significant first
  Geometry geom_;
  int bursts_per_row_;
  int rows_;
  std::string order_;
};

}  // namespace pimcore
