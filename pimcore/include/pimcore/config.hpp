// Minimal dependency-free YAML-subset configuration loader.
//
// Supports the subset used by the PimCore configuration files:
//   * nested mappings by two-space indentation
//   * scalar values: integers, floats, booleans, strings
//   * comments (#) and blank lines
// This keeps the simulator free of external libraries while remaining
// compatible with the repository-wide YAML configuration style.
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace pimcore {

class ConfigNode {
 public:
  ConfigNode() = default;
  explicit ConfigNode(std::string scalar) : scalar_(std::move(scalar)) {}

  bool is_scalar() const { return children_.empty(); }
  bool has(const std::string& key) const;

  const ConfigNode& at(const std::string& key) const;
  ConfigNode& child(const std::string& key);

  // Scalar accessors with optional defaults.
  std::string as_string() const { return scalar_; }
  double as_double() const;
  int64_t as_int() const;
  bool as_bool() const;

  double get_double(const std::string& key) const;
  double get_double(const std::string& key, double fallback) const;
  int64_t get_int(const std::string& key) const;
  int64_t get_int(const std::string& key, int64_t fallback) const;
  std::string get_string(const std::string& key) const;
  std::string get_string(const std::string& key,
                         const std::string& fallback) const;

  const std::map<std::string, ConfigNode>& children() const {
    return children_;
  }

  static ConfigNode load_file(const std::string& path);
  static ConfigNode parse(const std::string& text);

 private:
  std::string scalar_;
  std::map<std::string, ConfigNode> children_;
};

}  // namespace pimcore
