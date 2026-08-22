#include "pimcore/config.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace pimcore {

namespace {

std::string strip(const std::string& s) {
  size_t a = s.find_first_not_of(" \t\r\n");
  if (a == std::string::npos) return "";
  size_t b = s.find_last_not_of(" \t\r\n");
  return s.substr(a, b - a + 1);
}

// Remove a trailing comment that is not inside quotes.
std::string strip_comment(const std::string& s) {
  bool quoted = false;
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '"') quoted = !quoted;
    if (s[i] == '#' && !quoted) return s.substr(0, i);
  }
  return s;
}

struct Line {
  int indent;
  std::string key;
  std::string value;   // empty when the line opens a nested mapping
};

}  // namespace

bool ConfigNode::has(const std::string& key) const {
  return children_.find(key) != children_.end();
}

const ConfigNode& ConfigNode::at(const std::string& key) const {
  auto it = children_.find(key);
  if (it == children_.end())
    throw std::runtime_error("config key not found: " + key);
  return it->second;
}

ConfigNode& ConfigNode::child(const std::string& key) {
  return children_[key];
}

double ConfigNode::as_double() const {
  try {
    return std::stod(scalar_);
  } catch (...) {
    throw std::runtime_error("config value is not numeric: '" + scalar_ + "'");
  }
}

int64_t ConfigNode::as_int() const {
  // Accept scientific notation for large integers (e.g. 64.0e+9).
  return static_cast<int64_t>(as_double());
}

bool ConfigNode::as_bool() const {
  std::string v = scalar_;
  std::transform(v.begin(), v.end(), v.begin(), ::tolower);
  return v == "true" || v == "yes" || v == "on" || v == "1";
}

double ConfigNode::get_double(const std::string& key) const {
  return at(key).as_double();
}

double ConfigNode::get_double(const std::string& key, double fallback) const {
  return has(key) ? at(key).as_double() : fallback;
}

int64_t ConfigNode::get_int(const std::string& key) const {
  return at(key).as_int();
}

int64_t ConfigNode::get_int(const std::string& key, int64_t fallback) const {
  return has(key) ? at(key).as_int() : fallback;
}

std::string ConfigNode::get_string(const std::string& key) const {
  return at(key).as_string();
}

std::string ConfigNode::get_string(const std::string& key,
                                   const std::string& fallback) const {
  return has(key) ? at(key).as_string() : fallback;
}

ConfigNode ConfigNode::load_file(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open config file: " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  return parse(ss.str());
}

ConfigNode ConfigNode::parse(const std::string& text) {
  std::vector<Line> lines;
  std::stringstream ss(text);
  std::string raw;
  while (std::getline(ss, raw)) {
    std::string body = strip_comment(raw);
    if (strip(body).empty()) continue;
    int indent = 0;
    while (indent < static_cast<int>(body.size()) && body[indent] == ' ')
      ++indent;
    std::string content = strip(body);
    size_t colon = content.find(':');
    if (colon == std::string::npos)
      throw std::runtime_error("malformed config line: " + raw);
    Line ln;
    ln.indent = indent;
    ln.key = strip(content.substr(0, colon));
    ln.value = strip(content.substr(colon + 1));
    if (!ln.value.empty() && ln.value.front() == '[' &&
        ln.value.back() == ']')
      ln.value = strip(ln.value.substr(1, ln.value.size() - 2));
    // Strip surrounding quotes from string scalars.
    if (ln.value.size() >= 2 && ln.value.front() == '"' &&
        ln.value.back() == '"')
      ln.value = ln.value.substr(1, ln.value.size() - 2);
    lines.push_back(ln);
  }

  ConfigNode root;
  // Stack of (indent, node) for nested mapping construction.
  std::vector<std::pair<int, ConfigNode*>> stack{{-1, &root}};
  for (const Line& ln : lines) {
    while (stack.size() > 1 && ln.indent <= stack.back().first)
      stack.pop_back();
    ConfigNode* parent = stack.back().second;
    ConfigNode& node = parent->child(ln.key);
    if (ln.value.empty()) {
      stack.emplace_back(ln.indent, &node);
    } else if (ln.value.front() == '{' && ln.value.back() == '}') {
      // inline flow mapping: {a: 1, b: 2}
      std::string body = ln.value.substr(1, ln.value.size() - 2);
      std::stringstream fs(body);
      std::string item;
      while (std::getline(fs, item, ',')) {
        size_t ic = item.find(':');
        if (ic == std::string::npos)
          throw std::runtime_error("malformed flow mapping: " + ln.value);
        node.child(strip(item.substr(0, ic))) =
            ConfigNode(strip(item.substr(ic + 1)));
      }
    } else {
      node = ConfigNode(ln.value);
    }
  }
  return root;
}

std::string config_path(const std::string& config_root,
                        const std::string& kind, const std::string& name,
                        const std::string& option) {
  if (name.empty())
    throw std::runtime_error(option + " needs a name; expected one of the " +
                             kind + " configurations in " + config_root + "/" +
                             kind + "/");
  const std::string path = config_root + "/" + kind + "/" + name + ".yaml";
  std::ifstream probe(path);
  if (!probe)
    throw std::runtime_error("no " + kind + " configuration named '" + name +
                             "' (" + option + "); expected " + path);
  return path;
}

ConfigNode load_config(const std::string& config_root, const std::string& kind,
                       const std::string& name, const std::string& option) {
  return ConfigNode::load_file(config_path(config_root, kind, name, option));
}

}  // namespace pimcore
