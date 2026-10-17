"""Fix the Windows multiprocessing blocker in DDSP-SVC 6.x.

Symptom
-------
`python preprocess.py -c configs/reflow.yaml -j N` dies with

    TypeError: 'NoneType' object is not callable      (in reduction.dump)
    EOFError: Ran out of input                        (in the spawned child)

Cause
-----
`logger/utils.py` defines:

    class DotDict(dict):
        def __getattr__(*args):
            val = dict.get(*args)
            return DotDict(val) if type(val) is dict else val

`__getattr__` returns None for *every* missing attribute, including dunder
probes. Pickle asks for `obj.__getstate__`, gets None, and calls it.

preprocess.py passes this config object to a ProcessPoolExecutor initializer,
and on Windows the start method is spawn, so the config must be picklable -
hence preprocessing cannot run at all on Windows.

Fix
---
Only answer real keys; raise AttributeError for dunder lookups. Missing plain
keys still return None, so upstream behaviour (e.g. `args.model.t_start` being
None) is unchanged.

Idempotent, keeps a `.orig` backup.

  python scripts/patch_train_windows.py
"""
from __future__ import annotations

import os
import re
import shutil

ROOT = r"D:\17551\Documents\ai-cover-ddsp"

# The upstream source has trailing spaces on several lines, so match on
# structure rather than on an exact literal.
PATTERN = re.compile(
    r"class DotDict\(dict\):[ \t]*\n"
    r"[ \t]*def __getattr__\(\*args\):[^\n]*\n"
    r"[ \t]*val = dict\.get\(\*args\)[^\n]*\n"
    r"[ \t]*return DotDict\(val\) if type\(val\) is dict else val[^\n]*\n"
    r"[ \t]*\n"
    r"[ \t]*__setattr__ = dict\.__setitem__[^\n]*\n"
    r"[ \t]*__delattr__ = dict\.__delitem__[^\n]*\n"
)

NEW = '''class DotDict(dict):
    def __getattr__(self, name):
        # Returning None for missing attributes breaks pickling: pickle probes
        # __getstate__, gets None, and tries to call it. That kills every
        # multiprocessing path on Windows (preprocess.py's ProcessPoolExecutor
        # with the spawn start method, DataLoader workers). Only real keys are
        # answered here; dunder probes raise AttributeError as they should.
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)
        val = dict.get(self, name)
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
'''

TARGETS = [
    os.path.join(ROOT, "tools", "DDSP-SVC-6.3", "logger", "utils.py"),
    os.path.join(ROOT, "tools", "DDSP-SVC", "logger", "utils.py"),
]

MARKER = "dunder probes raise AttributeError"


def main() -> int:
    patched_any = False
    for target in TARGETS:
        if not os.path.isfile(target):
            print(f"skip (not found): {target}")
            continue
        with open(target, encoding="utf-8") as fh:
            src = fh.read()
        if MARKER in src:
            print(f"already patched: {target}")
            continue
        new_src, count = PATTERN.subn(NEW, src, count=1)
        if count != 1:
            print(f"pattern not found, refusing to patch: {target}")
            continue
        shutil.copy2(target, target + ".orig")
        with open(target, "w", encoding="utf-8") as fh:
            fh.write(new_src)
        print(f"patched: {target}  (backup: {target}.orig)")
        patched_any = True

    print("\nnothing to do" if not patched_any else "\nverify with the commands in 训练笔记.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
