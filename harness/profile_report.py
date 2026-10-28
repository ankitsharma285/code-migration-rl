"""Where the time goes in a GRPO step, from the per-step timings every training run already logs.

  python -m harness.profile_report results/mt_tp_s1 results/st_s1 ...    (a run dir with log.jsonl)

Per run: mean / median seconds per step for each phase, its share of the step, throughput, and peak GPU
memory. Phases (train/train_grpo.py, train/rollout.py):
  generate_s  vLLM sampling for every turn of every episode
  feedback_s  sandbox runs of each reply on the VISIBLE inputs (multi-turn feedback)
  score_s     sandbox runs of the final attempt on the HIDDEN inputs (reward)
  train_s     the GRPO update: old/ref log-probs, forward+backward per micro-batch, optimizer step
  sync_s      copying the new weights into vLLM
Checkpoint evals (every eval_every steps) are not part of a step and are excluded.
peak_mem_gb is torch's peak allocation on the GPU during the step, which includes vLLM's preallocated KV
cache (gpu_memory_utilization of the card) because vLLM runs in the same process.
"""
import argparse
import json
import statistics
from pathlib import Path

PHASES = ["generate_s", "feedback_s", "score_s", "train_s", "sync_s"]


def load(run):
    steps = []
    for line in open(Path(run, "log.jsonl")):
        r = json.loads(line)
        if "train_s" in r and "generate_s" in r:
            steps.append(r)
    cfg = json.loads(Path(run, "config.json").read_text())
    return cfg, steps


def report(run, skip_first):
    cfg, steps = load(run)
    steps = steps[skip_first:]
    if not steps:
        return f"## {run}\n\nno step records\n"
    n_eps = cfg["prompts_per_step"] * cfg["group_size"]
    tot = [sum(r.get(p, 0) for p in PHASES) for r in steps]
    mean_tot = statistics.mean(tot)
    out = [f"## {run}", "",
           f"{len(steps)} steps (first {skip_first} skipped as warm-up); {cfg['prompts_per_step']} prompts x "
           f"{cfg['group_size']} episodes, up to {cfg['turns']} turns; mean step {mean_tot:.0f} s "
           f"(median {statistics.median(tot):.0f} s)", "",
           "| Phase | mean s | median s | share |", "|---|---|---|---|"]
    for p in PHASES:
        v = [r.get(p, 0) for r in steps]
        out.append(f"| {p[:-2]} | {statistics.mean(v):.1f} | {statistics.median(v):.1f} | "
                   f"{100 * statistics.mean(v) / mean_tot:.0f}% |")
    gen_tok = [r["policy_tokens_mean"] * n_eps / r["generate_s"] for r in steps if r.get("generate_s")]
    trn_tok = [r["completion_tokens"] / r["train_s"] for r in steps if r.get("train_s") and r.get("completion_tokens")]
    mem = [r["peak_mem_gb"] for r in steps if "peak_mem_gb" in r]
    out += ["",
            f"- generation throughput: {statistics.median(gen_tok):,.0f} policy tokens/s (median)" if gen_tok else None,
            f"- update throughput: {statistics.median(trn_tok):,.0f} trained tokens/s (median; trained "
            f"episodes only, uniform groups dropped)" if trn_tok else None,
            f"- trained episodes per step: {statistics.mean(r['trained_episodes'] for r in steps):.0f} of {n_eps}",
            f"- peak GPU memory (torch, incl. vLLM KV pool): max {max(mem):.1f} GB, median {statistics.median(mem):.1f} GB"
            if mem else None,
            f"- total step time over the run: {sum(tot) / 3600:.1f} h"]
    prof = Path(run, "profile", "profile.json")
    if prof.exists():
        out += ["", f"torch.profiler trace of one update: {prof.parent} ({json.loads(prof.read_text())})"]
    return "\n".join(l for l in out if l is not None) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--skip-first", type=int, default=1, help="warm-up steps to leave out (compile, allocator)")
    ap.add_argument("--out", help="also write the report to this markdown file")
    a = ap.parse_args()
    text = "\n".join(report(r, a.skip_first) for r in a.runs)
    print(text)
    if a.out:
        Path(a.out).write_text(text)


if __name__ == "__main__":
    main()
