"""Generate a DDSP-SVC 6.x training config for a new dataset.

Copies the repo's default `configs/reflow.yaml` and rewrites only the fields that
are dataset-specific, so you never hand-edit relative paths.

  python scripts/make_train_config.py --name my-singer \
      --train work/my-dataset/train --val work/my-dataset/val \
      --repo tools/DDSP-SVC-6.3 [--batch-size 16] [--interval-val 500] \
      [--interval-force-save 5000] [--vocoder pretrain/pc_nsf_hifigan/model]

Paths in the config are resolved relative to the repo directory, because both
preprocess.py and train_reflow.py are run with the repo as the working directory.
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = r"D:\17551\Documents\ai-cover-ddsp"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="experiment name; becomes exp/<name>")
    ap.add_argument("--train", required=True, help="dataset train root (contains audio/)")
    ap.add_argument("--val", required=True, help="dataset val root (contains audio/)")
    ap.add_argument("--repo", default=os.path.join(ROOT, "tools", "DDSP-SVC-6.3"))
    ap.add_argument("--base", default="configs/reflow.yaml", help="template config, relative to the repo")
    ap.add_argument("--batch-size", type=int, default=0,
                    help="0 = keep the template value (48); lower it if VRAM is tight "
                         "or if the dataset has fewer clips than the batch size")
    ap.add_argument("--interval-val", type=int, default=0, help="0 = keep template")
    ap.add_argument("--interval-force-save", type=int, default=0, help="0 = keep template")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="training segment length in seconds; must be < shortest clip")
    ap.add_argument("--vocoder", default="", help="override vocoder.ckpt, e.g. pretrain/pc_nsf_hifigan/model")
    args = ap.parse_args()

    repo = os.path.abspath(args.repo)
    template = os.path.join(repo, args.base)
    if not os.path.isfile(template):
        print(f"template not found: {template}")
        return 1

    import yaml

    with open(template, encoding="utf-8") as fh:
        conf = yaml.safe_load(fh)

    n_train = 0
    audio_dir = os.path.join(args.train, "audio")
    if os.path.isdir(audio_dir):
        n_train = len([f for f in os.listdir(audio_dir) if f.lower().endswith(".wav")])

    conf["data"]["train_path"] = os.path.relpath(os.path.abspath(args.train), repo).replace("\\", "/")
    conf["data"]["valid_path"] = os.path.relpath(os.path.abspath(args.val), repo).replace("\\", "/")
    conf["env"]["expdir"] = f"exp/{args.name}"
    if args.batch_size:
        conf["train"]["batch_size"] = args.batch_size
    if args.interval_val:
        conf["train"]["interval_val"] = args.interval_val
    if args.interval_force_save:
        conf["train"]["interval_force_save"] = args.interval_force_save
    if args.duration:
        conf["data"]["duration"] = args.duration
    if args.vocoder:
        conf["vocoder"]["ckpt"] = args.vocoder

    out = os.path.join(repo, "configs", f"train-{args.name}.yaml")
    with open(out, "w", encoding="utf-8") as fh:
        yaml.dump(conf, fh, sort_keys=False, allow_unicode=True)

    bs = conf["train"]["batch_size"]
    print(f"wrote {out}")
    print(f"  train_path : {conf['data']['train_path']}   ({n_train} clips found)")
    print(f"  valid_path : {conf['data']['valid_path']}")
    print(f"  expdir     : {conf['env']['expdir']}")
    print(f"  batch_size : {bs}   duration: {conf['data']['duration']}s   "
          f"vocoder: {conf['vocoder']['ckpt']}")

    problems = []
    if n_train == 0:
        problems.append(f"no .wav clips in {audio_dir}")
    if n_train and bs > n_train:
        problems.append(f"batch_size {bs} exceeds the {n_train} training clips - "
                        f"each epoch becomes a single partial batch; use <= {max(n_train // 2, 1)}")
    if conf["data"]["duration"] and n_train:
        problems.append("") if False else None
    for p in problems:
        if p:
            print(f"  WARNING: {p}")

    print("\nnext:")
    print(f"  cd {repo}")
    print(f"  python preprocess.py -c configs/train-{args.name}.yaml -j 2")
    print(f"  python train_reflow.py -c configs/train-{args.name}.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
