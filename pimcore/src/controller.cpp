#include "pimcore/controller.hpp"

namespace pimcore {

HostController::HostController(const TimingParams& timing,
                               const Geometry& geom,
                               ArbitrationPolicy policy, size_t queue_depth)
    : timing_(timing), geom_(geom), policy_(policy), depth_(queue_depth),
      mapper_(geom, timing) {}

void HostController::push(const HostRequest& req) {
  if (queue_.size() >= depth_) {
    ++dropped;
    return;
  }
  queue_.push_back(req);
  ++issued;
}

HostRequest HostController::pop(int idx) {
  HostRequest r = queue_[idx];
  queue_.erase(queue_.begin() + idx);
  return r;
}

}  // namespace pimcore
