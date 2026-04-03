"""Checkpoint index: map short names to checkpoint directories.

Maintains a JSON index file (default: logdir/ckpt_index.json) that maps
user-defined names to full checkpoint paths.

Usage:
  # Register a checkpoint after training
  python ckpt_index.py add safe_95_s1 ./logdir/dataset_cifar10/.../seed_1

  # Register by config (auto-generates from flags)
  python ckpt_index.py add-from-config safe_95_s1 --sparsifier safe --sp 0.95 --seed 1 ...

  # List all registered checkpoints
  python ckpt_index.py list

  # Get path by name
  python ckpt_index.py get safe_95_s1

  # Remove
  python ckpt_index.py rm safe_95_s1

In eval scripts, use --checkpoint_name instead of --checkpoint_dir:
  python eval_corrupt.py --checkpoint_name safe_95_s1 --corrupt_intensity 3
"""

import os
import json
import sys
from pathlib import Path


DEFAULT_INDEX = os.path.join(os.path.dirname(__file__), 'logdir', 'ckpt_index.json')


def load_index(index_path=DEFAULT_INDEX):
    if os.path.exists(index_path):
        with open(index_path) as f:
            return json.load(f)
    return {}


def save_index(index, index_path=DEFAULT_INDEX):
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2, sort_keys=True)


def resolve_name(name, index_path=DEFAULT_INDEX):
    """Resolve a checkpoint name to (path, config_dict)."""
    index = load_index(index_path)
    if name not in index:
        raise KeyError(f"Checkpoint '{name}' not found. Available: {list(index.keys())}")
    entry = index[name]
    # Support both old format (string path) and new format (dict with path + config)
    if isinstance(entry, str):
        return entry, {}
    path = entry['path']
    config = entry.get('config', {})
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint '{name}' points to '{path}' which does not exist.")
    return path, config


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == 'add':
        if len(sys.argv) != 4:
            print("Usage: python ckpt_index.py add NAME PATH")
            sys.exit(1)
        name, path = sys.argv[2], os.path.abspath(sys.argv[3])
        assert os.path.exists(path), f"Path does not exist: {path}"
        index = load_index()
        index[name] = path
        save_index(index)
        print(f"Added: {name} -> {path}")

    elif cmd == 'list':
        index = load_index()
        if not index:
            print("No checkpoints registered.")
        else:
            for name, path in sorted(index.items()):
                exists = "✓" if os.path.exists(path) else "✗"
                print(f"  {exists} {name:<30s} {path}")

    elif cmd == 'get':
        if len(sys.argv) != 3:
            print("Usage: python ckpt_index.py get NAME")
            sys.exit(1)
        print(resolve_name(sys.argv[2]))

    elif cmd == 'rm':
        if len(sys.argv) != 3:
            print("Usage: python ckpt_index.py rm NAME")
            sys.exit(1)
        index = load_index()
        name = sys.argv[2]
        if name in index:
            del index[name]
            save_index(index)
            print(f"Removed: {name}")
        else:
            print(f"Not found: {name}")

    elif cmd == 'auto':
        # Auto-register all checkpoint dirs found in logdir
        logdir = sys.argv[2] if len(sys.argv) > 2 else 'logdir'
        index = load_index()
        count = 0
        for root, dirs, files in os.walk(logdir):
            ckpts = [f for f in files if f.startswith('checkpoint_')]
            if ckpts:
                # Parse path components to build a short name
                parts = Path(root).relative_to(logdir).parts
                # Parse known key-value path components
                known_keys = ['dataset', 'optimizer', 'model', 'num_epochs', 'lr', 'wd',
                              'momentum', 'label_noise_ratio', 'sparsifier', 'sp', 'sp_scope',
                              'sp_schedule', 'lambda', 'lambda_schedule', 'rho',
                              'dual_update_interval', 'seed']
                kv = {}
                for p in parts:
                    for k in sorted(known_keys, key=len, reverse=True):
                        if p.startswith(k + '_'):
                            kv[k] = p[len(k) + 1:]
                            break
                sparsifier = kv.get('sparsifier', '?')
                sparsity = kv.get('sp', '?')
                seed = kv.get('seed', '?')
                name = f"{sparsifier}_{sparsity}_s{seed}"
                index[name] = os.path.abspath(root)
                count += 1
        save_index(index)
        print(f"Registered {count} checkpoints. Run 'python ckpt_index.py list' to see them.")

    else:
        print(f"Unknown command: {cmd}")
        print("Commands: add, list, get, rm, auto")
        sys.exit(1)


if __name__ == '__main__':
    main()
