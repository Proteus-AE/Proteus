#include "pimcore/address.hpp"

#include <sstream>
#include <stdexcept>

namespace pimcore {

AddressMapper::AddressMapper(const Geometry& geom, const TimingParams& timing,
                             const std::string& order)
    : geom_(geom), order_(order) {
  bursts_per_row_ = timing.row_bytes / timing.burst_bytes;
  rows_ = 1 << 16;   // modeled row space per bank (capacity-agnostic timing)

  // Parse "a-b-c-d-e" into interleaving fields; the FIRST token owns the
  // lowest address bits, so consecutive addresses interleave across it.
  std::vector<std::string> toks;
  std::stringstream ss(order);
  std::string t;
  while (std::getline(ss, t, '-')) toks.push_back(t);
  if (toks.size() != 5)
    throw std::runtime_error("address order must have 5 fields: " + order);

  auto field_of = [&](const std::string& name) -> Field {
    if (name == "ch")   return {'C', geom_.channels};
    if (name == "die")  return {'D', geom_.dies_per_channel};
    if (name == "bg")   return {'G', geom_.bankgroups_per_die};
    if (name == "bank") return {'B', geom_.banks_per_bankgroup};
    if (name == "row")  return {'R', rows_};
    if (name == "col")  return {'O', bursts_per_row_};
    throw std::runtime_error("unknown address field: " + name);
  };
  // The mapper folds die into the bank-group field when 'die' is omitted
  // from a 5-field order that includes both bg and bank.
  bool has_die = false;
  for (const auto& tok : toks) has_die |= (tok == "die");
  for (const auto& tok : toks) fields_.push_back(field_of(tok));
  if (!has_die) {
    // widen the bank-group field to cover dies x bank groups
    for (auto& f : fields_)
      if (f.id == 'G') f.size = geom_.dies_per_channel * geom_.bankgroups_per_die;
  }
}

Coordinates AddressMapper::decode(addr_t addr) const {
  Coordinates c;
  addr_t rem = addr;
  int bg_all = 0;
  for (const Field& f : fields_) {
    int v = static_cast<int>(rem % f.size);
    rem /= f.size;
    switch (f.id) {
      case 'C': c.channel = v; break;
      case 'D': c.die = v; break;
      case 'G': bg_all = v; break;
      case 'B': c.bank = v; break;
      case 'R': c.row = v; break;
      case 'O': c.col = v; break;
    }
  }
  // split the (possibly widened) bank-group field
  c.die = bg_all / geom_.bankgroups_per_die;
  c.bankgroup = bg_all % geom_.bankgroups_per_die;
  return c;
}

addr_t AddressMapper::encode(const Coordinates& c) const {
  addr_t addr = 0;
  addr_t scale = 1;
  int bg_all = c.die * geom_.bankgroups_per_die + c.bankgroup;
  for (const Field& f : fields_) {
    int v = 0;
    switch (f.id) {
      case 'C': v = c.channel; break;
      case 'D': v = c.die; break;
      case 'G': v = bg_all; break;
      case 'B': v = c.bank; break;
      case 'R': v = c.row; break;
      case 'O': v = c.col; break;
    }
    addr += scale * static_cast<addr_t>(v % f.size);
    scale *= f.size;
  }
  return addr;
}

}  // namespace pimcore
