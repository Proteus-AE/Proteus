"""Configuration loading: model / system / memory YAML files."""
import os
import yaml

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_DIR = os.path.join(_ROOT, "configs")


class Cfg(dict):
    """Dict with attribute access (read-only convenience wrapper)."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Cfg(v) if isinstance(v, dict) else v


def _load(path):
    with open(path, "r") as f:
        return Cfg(yaml.safe_load(f))


def load_model(name):
    return _load(os.path.join(CONFIG_DIR, "models", f"{name}.yaml"))


def load_system(name):
    return _load(os.path.join(CONFIG_DIR, "systems", f"{name}.yaml"))


def load_memory(name):
    return _load(os.path.join(CONFIG_DIR, "memory", f"{name}.yaml"))


def list_models():
    d = os.path.join(CONFIG_DIR, "models")
    return sorted(f[:-5] for f in os.listdir(d) if f.endswith(".yaml"))


def list_systems():
    d = os.path.join(CONFIG_DIR, "systems")
    return sorted(f[:-5] for f in os.listdir(d) if f.endswith(".yaml"))
