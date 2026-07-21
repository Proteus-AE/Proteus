#include "pimcore/bank.hpp"

#include <algorithm>

namespace pimcore {

ns_t Bank::apply_act(ns_t t, int row, const TimingParams& p) {
  t = std::max(t, next_act_);
  phase_ = BankPhase::ACTIVE;
  open_row_ = row;
  act_time_ = t;
  ++acts;
  next_act_ = t + p.tRC;
  next_col_ = std::max(next_col_, t + p.tRCD);
  next_pre_ = std::max(next_pre_, t + p.tRAS);
  return t;
}

ns_t Bank::apply_read(ns_t t, const TimingParams& p) {
  t = std::max(t, next_col_);
  // write-to-read turnaround on the same bank
  t = std::max(t, last_write_ + p.tWTR);
  ++reads;
  next_col_ = std::max(next_col_, t + p.tCCD_L);
  next_pre_ = std::max(next_pre_, t + p.tRTP);
  return t;
}

ns_t Bank::apply_write(ns_t t, const TimingParams& p) {
  t = std::max(t, next_col_);
  ++writes;
  last_write_ = t;
  next_col_ = std::max(next_col_, t + p.tCCD_L);
  next_pre_ = std::max(next_pre_, t + p.tWR);
  return t;
}

void Bank::apply_pre(ns_t t, const TimingParams& p) {
  phase_ = BankPhase::IDLE;
  open_row_ = -1;
  next_act_ = std::max(next_act_, std::max(t, next_pre_) + p.tRP);
}

ns_t DieState::constrain_act(ns_t t) {
  t = std::max(t, last_act_ + p_.tRRD_S);
  while (!act_window_.empty() && act_window_.front() <= t - p_.tFAW)
    act_window_.pop_front();
  if (act_window_.size() >= 4) {
    t = std::max(t, act_window_.front() + p_.tFAW);
    while (!act_window_.empty() && act_window_.front() <= t - p_.tFAW)
      act_window_.pop_front();
  }
  return t;
}

void DieState::record_act(ns_t t) {
  last_act_ = t;
  act_window_.push_back(t);
}

ns_t DieState::constrain_refresh(ns_t t, ns_t* stall, uint64_t* events) {
  while (t >= next_refresh_) {
    ns_t start = next_refresh_;
    if (t < start + p_.tRFCab) {
      if (stall) *stall += (start + p_.tRFCab) - t;
      t = start + p_.tRFCab;
    }
    next_refresh_ += p_.tREFI;
    if (events) ++(*events);
  }
  return t;
}

}  // namespace pimcore
