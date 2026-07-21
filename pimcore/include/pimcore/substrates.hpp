// Built-in memory-substrate presets.
//
// One translation unit per device family (src/substrates/*.cpp) holds the
// full timing / geometry / energy tables for that substrate, following the
// respective datasheet class:
//
//   lpddr5x_pim.cpp   LPDDR5X-8533 with near-bank PEs (the Proteus device)
//   hbm_pim.cpp       HBM2E-class stacked DRAM with bank-adjacent FPUs
//   gddr6_aim.cpp     GDDR6-class graphics DRAM with per-bank MAC units
//
// The tables are the defaults behind TimingParams/Geometry/EnergyTable
// ::defaults_for(); any field can still be overridden from a config file.
#pragma once

#include "pimcore/timing.hpp"

namespace pimcore {
namespace substrates {

TimingParams lpddr5x_pim_timing();
Geometry     lpddr5x_pim_geometry();
EnergyTable  lpddr5x_pim_energy();

TimingParams hbm_pim_timing();
Geometry     hbm_pim_geometry();
EnergyTable  hbm_pim_energy();

TimingParams gddr6_aim_timing();
Geometry     gddr6_aim_geometry();
EnergyTable  gddr6_aim_energy();

}  // namespace substrates
}  // namespace pimcore
