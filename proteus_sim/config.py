"""Configuration loading: model / system / memory YAML files.

Every quantity the simulator consumes comes from ``configs/``. A missing or
malformed entry is reported against the file it came from rather than as a
bare ``KeyError`` from deep inside the timing engine.
"""
import os
import yaml

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_DIR = os.path.join(_ROOT, "configs")


class ConfigError(RuntimeError):
    """A configuration file is missing, unreadable, or missing a key."""


class Cfg(dict):
    """Configuration mapping that reports its origin on a missing key."""

    def __init__(self, data, origin=""):
        super().__init__(data)
        self.origin = origin
        for k, v in list(self.items()):
            if isinstance(v, dict) and not isinstance(v, Cfg):
                super().__setitem__(k, Cfg(v, origin))

    def __missing__(self, key):
        where = f" of {self.origin}" if self.origin else ""
        raise ConfigError(f"missing configuration key '{key}'{where}")

    def get(self, key, default=None):
        if key not in self:
            return default
        return self[key]


def _load(path):
    if not os.path.exists(path):
        raise ConfigError(f"configuration file not found: {path}")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError(f"{path} does not contain a YAML mapping")
    return Cfg(data, os.path.relpath(path, _ROOT))


def load_model(name):
    return _load(os.path.join(CONFIG_DIR, "models", f"{name}.yaml"))


def load_system(name):
    return _load(os.path.join(CONFIG_DIR, "systems", f"{name}.yaml"))


def load_memory(name):
    return _load(os.path.join(CONFIG_DIR, "memory", f"{name}.yaml"))


_COMMON = None


def load_common():
    """Cross-system constants (configs/common.yaml)."""
    global _COMMON
    if _COMMON is None:
        _COMMON = _load(os.path.join(CONFIG_DIR, "common.yaml"))
    return _COMMON


def list_models():
    d = os.path.join(CONFIG_DIR, "models")
    return sorted(f[:-5] for f in os.listdir(d) if f.endswith(".yaml"))


def list_systems():
    d = os.path.join(CONFIG_DIR, "systems")
    return sorted(f[:-5] for f in os.listdir(d) if f.endswith(".yaml"))


def list_memories():
    d = os.path.join(CONFIG_DIR, "memory")
    return sorted(f[:-5] for f in os.listdir(d) if f.endswith(".yaml"))
