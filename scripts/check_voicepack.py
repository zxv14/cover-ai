"""Check whether a "voice pack" (speaker model) can actually be used by this tree.

It runs the exact loader that inference uses (`reflow.vocoder.load_model_vocoder`
for 6.x), so a PASS means inference will really start - not just that the file
looks plausible.

  python scripts/check_voicepack.py --model models/sakiko-6.3/sakiko.pt
  python scripts/check_voicepack.py --model models/pallas-6.2/pallas.pt --repo tools/DDSP-SVC

Checks, in order of how often they actually bite:
  1. file type      - is this a DDSP-SVC reflow checkpoint, or an RVC/So-VITS pack?
  2. layout         - config.yaml next to the .pt, branch identity of the repo
  3. references     - encoder_ckpt / vocoder.ckpt resolve, vocoder has the fields
                      the code reads (pc_aug), sampling rate / block size sane
  4. real load      - build the model and load the state dict, strict
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = r"D:\17551\Documents\ai-cover-ddsp"

PASS, FAIL, WARN, INFO = "PASS", "FAIL", "WARN", "INFO "
_results: list[tuple[str, str, str]] = []


def record(status: str, what: str, detail: str = "") -> None:
    _results.append((status, what, detail))
    print(f"  [{status}] {what}" + (f"  -  {detail}" if detail else ""), flush=True)


def _collect_keys(obj, depth: int = 2) -> list[str]:
    """Flatten key names a couple of levels deep.

    RVC checkpoints are often wrapped as {'weight': {...}} rather than being a
    flat tensor dict, so looking only at top-level keys misses them.
    """
    out: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            out.append(str(key))
            if depth > 0 and isinstance(value, dict):
                out.extend(_collect_keys(value, depth - 1))
    return out


def detect_family(ckpt) -> tuple[str, str]:
    """Guess which SVC family a checkpoint belongs to from its structure."""
    keys = _collect_keys(ckpt, depth=2)
    joined = " ".join(keys).lower()
    wrapped = isinstance(ckpt, dict) and isinstance(ckpt.get("model"), dict)

    ddsp_markers = ("unit2control", "aux_decoder", "reflow", "velocity_fn", "cfm")
    rvc_markers = ("enc_p", "emb_g", "posterior_encoder", "flow.flows", "dec.flow")

    if wrapped and any(m in joined for m in ddsp_markers):
        return "ddsp-reflow", "top-level 'model' dict containing reflow/unit2control weights"
    if any(m in joined for m in rvc_markers):
        return "rvc", f"RVC-style keys found ({', '.join(m for m in rvc_markers if m in joined)})"
    if wrapped:
        return "ddsp", "'model' dict but no reflow keys (older 3.x/5.x-era DDSP model?)"
    if isinstance(ckpt, dict) and keys:
        return "unknown", f"dict checkpoint, first keys: {keys[:6]}"
    if hasattr(ckpt, "keys"):
        return "bare-state-dict", "a bare state dict, not a DDSP training checkpoint"
    return "unknown", f"unexpected object: {type(ckpt).__name__}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to the speaker .pt")
    ap.add_argument("--repo", default=os.path.join(ROOT, "tools", "DDSP-SVC-6.3"),
                    help="DDSP-SVC checkout that will run inference")
    args = ap.parse_args()

    model = os.path.abspath(args.model)
    repo = os.path.abspath(args.repo)
    print(f"model: {model}\nrepo : {repo}\n")

    # ---------- 1. branch identity ----------
    print("1) branch / repo")
    if not os.path.isdir(repo):
        record(FAIL, "repo directory", f"{repo} does not exist")
        return finish()
    has_62 = os.path.isfile(os.path.join(repo, "reflow", "lynxnet.py"))
    has_63 = os.path.isfile(os.path.join(repo, "reflow", "lynxnet2.py"))
    branch = "6.2" if has_62 else ("6.3/6.5" if has_63 else "unknown")
    record(INFO if has_62 or has_63 else FAIL, "backbone present",
           f"{branch} (lynxnet.py={has_62}, lynxnet2.py={has_63})")
    if has_62 and "DDSP-SVC-6.3" in repo:
        record(WARN, "repo naming", "this path looks like the 6.3 tree but has 6.2 backbone files")

    # ---------- 2. checkpoint family ----------
    print("\n2) checkpoint")
    if not os.path.isfile(model):
        record(FAIL, "model file", f"{model} does not exist")
        return finish()
    record(PASS, "model file", f"{os.path.getsize(model) / 1e6:.1f} MB")

    import torch  # imported late so the errors above stay readable

    try:
        ckpt = torch.load(model, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        record(FAIL, "torch.load", f"{type(exc).__name__}: {exc}")
        return finish()

    family, why = detect_family(ckpt)
    if family == "ddsp-reflow":
        record(PASS, "checkpoint family", f"DDSP-SVC reflow ({why})")
    elif family in ("rvc",):
        record(FAIL, "checkpoint family",
               f"{family} - DDSP-SVC cannot load this; use the RVC project instead ({why})")
        return finish()
    else:
        record(WARN, "checkpoint family", f"{family} - {why} (will still try to load it)")

    if isinstance(ckpt, dict):
        extra = {k: v for k, v in ckpt.items() if k != "model"}
        if extra:
            scalars = {k: v for k, v in extra.items() if isinstance(v, (int, float, str))}
            record(INFO, "checkpoint metadata", str(scalars)[:120])

    # ---------- 3. layout ----------
    print("\n3) layout")
    cfg = os.path.join(os.path.dirname(model), "config.yaml")
    if not os.path.isfile(cfg):
        record(FAIL, "config.yaml next to the .pt",
               "required - the loader reads it from os.path.split(model_path)[0]")
        return finish()
    record(PASS, "config.yaml present")

    import yaml

    with open(cfg, encoding="utf-8") as fh:
        conf = yaml.safe_load(fh)

    data = conf.get("data", {})
    encoder = data.get("encoder")
    record(INFO, "encoder", f"{encoder} (out_channels={data.get('encoder_out_channels')}, "
                            f"hop={data.get('encoder_hop_size')})")
    record(INFO, "audio", f"{data.get('sampling_rate')} Hz, block_size={data.get('block_size')}, "
                          f"f0_extractor={data.get('f0_extractor')}")
    if data.get("sampling_rate") != 44100:
        record(WARN, "sampling rate", "community models are usually 44100; other rates can still work")

    # ---------- 4. referenced files, resolved the way inference resolves them ----------
    print("\n4) referenced files (relative to the repo, like main_reflow.py)")
    checked_any = False
    for label, key in (("encoder_ckpt", data.get("encoder_ckpt")), ):
        if not key:
            record(WARN, label, "not set in config")
            continue
        resolved = key if os.path.isabs(key) else os.path.join(repo, key)
        if os.path.isfile(resolved):
            record(PASS, label, f"{key} -> {os.path.getsize(resolved) / 1e6:.1f} MB")
        else:
            record(FAIL, label, f"{key} not found under the repo (fix this path in config.yaml)")
        checked_any = True

    voc = conf.get("vocoder", {})
    vckpt = voc.get("ckpt")
    if not vckpt:
        record(FAIL, "vocoder.ckpt", "not set in config")
    else:
        resolved = vckpt if os.path.isabs(vckpt) else os.path.join(repo, vckpt)
        if os.path.isfile(resolved):
            record(PASS, "vocoder.ckpt", f"{vckpt} -> {os.path.getsize(resolved) / 1e6:.1f} MB")
            vcfg = os.path.join(os.path.dirname(resolved), "config.json")
            if not os.path.isfile(vcfg):
                record(FAIL, "vocoder config.json", f"expected next to {vckpt}")
            else:
                import json

                with open(vcfg, encoding="utf-8") as fh:
                    vj = json.load(fh)
                if "pc_aug" in vj:
                    record(PASS, "vocoder fields", f"pc_aug={vj.get('pc_aug')}")
                else:
                    # nsf_hifigan/env.py's AttrDict returns None for missing keys,
                    # so this only disables the optional register-shift feature.
                    record(INFO, "vocoder fields",
                           "no pc_aug key - harmless (AttrDict returns None, main_reflow "
                           "just prints 'register shift not supported'). Prefer the vocoder "
                           "the model was trained with for timbre fidelity.")
        else:
            record(FAIL, "vocoder.ckpt", f"{vckpt} not found under the repo")

    # ---------- 5. the decisive test: actually load it ----------
    print("\n5) real load (same code path as inference)")
    cwd = os.getcwd()
    # drop any repo modules already imported, so switching --repo within one
    # process does not silently keep using the previous tree's code
    for name in [m for m in list(sys.modules)
                 if m in ("ddsp", "reflow", "slicer", "nsf_hifigan", "encoder")
                 or m.startswith(("ddsp.", "reflow.", "nsf_hifigan.", "encoder."))]:
        sys.modules.pop(name, None)
    sys.path.insert(0, repo)
    os.chdir(repo)
    try:
        import importlib

        import ddsp.vocoder  # noqa: F401  (must import before reflow.vocoder)
        rv = importlib.import_module("reflow.vocoder")
        model_obj, vocoder_obj, loaded = rv.load_model_vocoder(model, device="cpu")
        n_params = sum(p.numel() for p in model_obj.parameters())
        record(PASS, "load_model_vocoder", f"{n_params / 1e6:.1f}M params, "
                                           f"n_spk={loaded.model.n_spk}, type={loaded.model.type}")
        try:
            _ = vocoder_obj.vocoder.h.pc_aug
            record(PASS, "vocoder.h.pc_aug readable", "main_reflow.py's register-shift branch will work")
        except Exception as exc:  # noqa: BLE001
            record(FAIL, "vocoder.h.pc_aug", f"{type(exc).__name__}: {exc} -> main_reflow.py will crash at line ~247")
    except RuntimeError as exc:
        # by far the most common real failure: right family, wrong branch
        record(FAIL, "load_model_vocoder",
               f"state_dict mismatch - this pack does not belong to this branch "
               f"({str(exc)[:150]})")
        record(INFO, "hint", "6.2 packs need tools/DDSP-SVC (lynxnet.py); "
                             "6.3 packs need tools/DDSP-SVC-6.3 (lynxnet2.py)")
    except Exception as exc:  # noqa: BLE001
        record(FAIL, "load_model_vocoder", f"{type(exc).__name__}: {str(exc)[:220]}")
    finally:
        os.chdir(cwd)
        if repo in sys.path:
            sys.path.remove(repo)

    return finish()


def finish() -> int:
    print("\n=== summary ===")
    fails = [r for r in _results if r[0] == FAIL]
    warns = [r for r in _results if r[0] == WARN]
    for status, what, detail in fails:
        print(f"  BLOCKER  {what}: {detail}")
    for status, what, detail in warns:
        print(f"  warning  {what}: {detail}")
    if not fails:
        print("  => this voice pack can be used with this repo. "
              "Run: scripts/run_ddsp.py --model <path> --repo <repo>")
    else:
        print("  => not usable as-is; fix the blockers above.")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
