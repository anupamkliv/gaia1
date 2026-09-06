from types import SimpleNamespace
import yaml

def _convert(x):
    if isinstance(x, dict):
        return SimpleNamespace(**{k:_convert(v) for k,v in x.items()})
    if isinstance(x, list):
        return [_convert(v) for v in x]
    return x

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return _convert(yaml.safe_load(f))
