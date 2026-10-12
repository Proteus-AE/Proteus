// Per-bank DRAM state machine and per-die activation-window bookkeeping.
#pragma once

#include <deque>

#include "pimcore/timing.hpp"
#include "pimcore/types.hpp"

namespace pimcore {

enum class BankPhase : uint8_t { IDLE, ACTIVE };

class Bank {
 public:
  Bank(int die, int bg, int index)
      : die_(die), bg_(bg), index_(index) {}

  int die() const { return die_; }
  int bank_group() const { return bg_; }
  int index() const { return index_; }

  BankPhase phase() const { return phase_; }
  int open_row() const { return open_row_; }
  bool row_open(int row) const {
    return phase_ == BankPhase::ACTIVE && open_row_ == row;
  }

  // Earliest-issue constraint accessors (absolute ns).
  ns_t next_act() const { return next_act_; }
  ns_t next_col() const { return next_col_; }
  ns_t next_pre() const { return next_pre_; }

  // State transitions. Each returns the effective issue time after applying
  // this bank's local constraints (callers add bus/die/refresh constraints).
  ns_t apply_act(ns_t t, int row, const TimingParams& p,
                 bool all_bank = false);
  // `cadence` is the bank's own column cycle: the near-bank read path runs
  // from the bank's I/O sense amplifiers straight into its co-located PE and
  // never touches the shared bus, so its cycle is set by the connectivity
  // mode rather than by tCCD_L.
  ns_t apply_read(ns_t t, const TimingParams& p, ns_t cadence = 0.0);
  // An all-bank column write is executed by the bank at the cadence of the
  // mode in force, exactly as its reads are; `cadence` of zero selects the
  // shared-bus column cycle a single-bank host write would take.
  ns_t apply_write(ns_t t, const TimingParams& p, ns_t cadence = 0.0);
  void apply_pre(ns_t t, const TimingParams& p,
                 bool all_bank = false);

  // statistics
  uint64_t acts = 0;
  uint64_t reads = 0;
  uint64_t writes = 0;
  uint64_t hits = 0;

 private:
  int die_;
  int bg_;
  int index_;
  BankPhase phase_ = BankPhase::IDLE;
  int open_row_ = -1;
  ns_t act_time_ = -1e18;
  ns_t next_act_ = 0.0;
  ns_t next_col_ = 0.0;
  ns_t next_pre_ = 0.0;
  ns_t last_write_ = -1e18;
};

// Per-die constraints: tRRD spacing, the rolling tFAW window, and the
// all-bank refresh schedule (aligned across dies for lockstep execution).
class DieState {
 public:
  explicit DieState(const TimingParams& p) : p_(p) {
    next_refresh_ = p.tREFI;
  }

  // Earliest time an ACT may issue on this die at or after `t`.
  ns_t constrain_act(ns_t t);
  void record_act(ns_t t);

  // Advance `t` past any refresh window; returns adjusted time and
  // accumulates blocked time into `stall`.
  ns_t constrain_refresh(ns_t t, ns_t* stall, uint64_t* events);

 private:
  const TimingParams& p_;
  std::deque<ns_t> act_window_;
  ns_t last_act_ = -1e18;
  ns_t next_refresh_;
};

}  // namespace pimcore
