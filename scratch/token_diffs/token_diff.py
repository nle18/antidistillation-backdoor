# -*- coding: utf-8 -*-
"""
compare_next_token_multistep.py

Generate a full completion with a teacher model, then show the baseline vs.
modified next-token distribution at N randomly chosen time steps.

  baseline (lam = 0): softmax(teacher_logits / tau)
  modified (lam > 0): softmax((teacher_logits + sign * ad_term) / tau)
  ad_term = lam / (2 * eps) * (student_logits(+eps*grad) - student_logits(-eps*grad))

--gen_mode decides which distribution the tokens are *sampled* from while
generating (default: modified, i.e. the actual ADS trace). At every chosen step
both distributions are computed on the same prefix, so the comparison is
apples-to-apples either way.

Example:
  python compare_next_token_multistep.py \
      --teacher Qwen/Qwen2.5-1.5B-Instruct \
      --proxy_student Qwen/Qwen2.5-0.5B-Instruct \
      --grad_path grads.pt --lam 1e-3 --eps 1e-2 --num_steps 10
"""

import argparse
import random

import torch
import torch.nn.functional as F
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from transformers import AutoModelForCausalLM, AutoTokenizer

console = Console()

SYSTEM_PROMPT = (
    "You are a math teacher. You will be given a math problem and you will solve it step by step.\n"
    "You will output your final solution like \\boxed{ANSWER}. Be sure to include relevant units within the brackets and fully evaluate arithmetic expressions.\n"
)
CONTEXT = (
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
    "How many clips did Natalia sell altogether in April and May?"
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def load_model(name, dtype, device):
    return AutoModelForCausalLM.from_pretrained(
        name, trust_remote_code=True, attn_implementation="sdpa", torch_dtype=dtype
    ).to(device)


@torch.no_grad()
def perturb_(model, grads, scale):
    """In-place: param += scale * grad (math in fp32, cast back). Returns used grad keys."""
    used = set()
    for name, p in model.named_parameters():
        key = "module." + name  # grads were saved from a DDP/Accelerate-wrapped model
        if key in grads:
            g = grads[key].to(p.device, dtype=torch.float32)
            p.data = (p.data.to(torch.float32) + scale * g).to(p.dtype)
            used.add(key)
    return used


@torch.no_grad()
def fwd(model, ids, cache):
    """One incremental forward pass; returns (last-position logits, updated cache)."""
    out = model(input_ids=ids, past_key_values=cache, use_cache=True)
    return out.logits[0, -1].float(), out.past_key_values


def sample_top_p(probs, top_p):
    sp, si = torch.sort(probs, descending=True)
    cum = sp.cumsum(0)
    sp = torch.where(cum - sp > top_p, torch.zeros_like(sp), sp)  # keep smallest set with mass >= top_p
    sp = sp / sp.sum()
    return si[torch.multinomial(sp, 1)].item()


def tok_str(tokenizer, tid):
    return repr(tokenizer.decode([tid]))


def get_eos_ids(model, tokenizer):
    ids = set()
    g = model.generation_config.eos_token_id
    if g is not None:
        ids.update(g if isinstance(g, (list, tuple)) else [g])
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    return ids


# ----------------------------------------------------------------------------
# Display
# ----------------------------------------------------------------------------
def show_comparison(tokenizer, t_logits, ad_term, args, title):
    sgn = -1.0 if args.sign == "minus" else 1.0
    p_base = F.softmax(t_logits / args.tau, dim=-1)
    p_mod = F.softmax((t_logits + sgn * ad_term) / args.tau, dim=-1)
    k = args.top_k
    pb, ib = torch.topk(p_base, k)
    pm, im = torch.topk(p_mod, k)

    table = Table(title=title, header_style="bold cyan", title_justify="left")
    table.add_column("#", justify="right")
    table.add_column("Baseline (lam=0)", style="green")
    table.add_column("p", justify="right", style="green")
    table.add_column("Modified (lam>0)", style="magenta")
    table.add_column("p", justify="right", style="magenta")
    for r in range(k):
        table.add_row(
            str(r + 1),
            tok_str(tokenizer, ib[r].item()), f"{pb[r].item():.4f}",
            tok_str(tokenizer, im[r].item()), f"{pm[r].item():.4f}",
        )
    console.print(table)

    union = list(dict.fromkeys(ib.tolist() + im.tolist()))
    rank_base = {t: i + 1 for i, t in enumerate(torch.argsort(p_base, descending=True)[:1000].tolist())}
    rank_mod = {t: i + 1 for i, t in enumerate(torch.argsort(p_mod, descending=True)[:1000].tolist())}

    dtab = Table(title="Per-token change (union of both top-k sets)", header_style="bold cyan",
                 title_justify="left")
    dtab.add_column("Token")
    dtab.add_column("p_base", justify="right")
    dtab.add_column("p_mod", justify="right")
    dtab.add_column("Δp", justify="right")
    dtab.add_column("rank base→mod", justify="center")
    dtab.add_column("ad_term", justify="right")
    for t in sorted(union, key=lambda t: -p_base[t].item()):
        d = p_mod[t].item() - p_base[t].item()
        color = "green" if d > 0 else "red" if d < 0 else "white"
        dtab.add_row(
            tok_str(tokenizer, t),
            f"{p_base[t].item():.4f}",
            f"{p_mod[t].item():.4f}",
            f"[{color}]{d:+.4f}[/{color}]",
            f"{rank_base.get(t, '>1000')}→{rank_mod.get(t, '>1000')}",
            f"{ad_term[t].item():+.3e}",
        )
    console.print(dtab)

    kl = F.kl_div(p_mod.clamp_min(1e-12).log(), p_base, reduction="sum").item()  # KL(base || mod)
    tv = 0.5 * (p_base - p_mod).abs().sum().item()
    top1_changed = ib[0].item() != im[0].item()
    console.print(
        f"  KL(base‖mod)={kl:.4e}   TV={tv:.4f}   top-1 changed={top1_changed}   "
        f"|ad_term| mean/max={ad_term.abs().mean().item():.3e}/{ad_term.abs().max().item():.3e}\n"
    )
    return {"kl": kl, "tv": tv, "top1_changed": top1_changed}


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--proxy_student", required=True)
    ap.add_argument("--grad_path", required=True)
    ap.add_argument("--lam", type=float, required=True, help="antidistillation strength (>0)")
    ap.add_argument("--eps", type=float, default=1e-2, help="finite-difference step size")
    ap.add_argument("--tau", type=float, default=0.9, help="sampling temperature")
    ap.add_argument("--top_p", type=float, default=0.95, help="top-p used for *sampling* only")
    ap.add_argument("--tokenizer", type=str, default=None, help="defaults to --teacher")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--num_steps", type=int, default=10, help="# random time steps to display")
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--gen_mode", choices=["modified", "baseline"], default="modified",
                    help="which distribution tokens are sampled from during generation")
    ap.add_argument("--sign", choices=["minus", "plus"], default="minus",
                    help="minus matches your LogprobsModifierBackdoor (scores - ad_term)")
    ap.add_argument("--student_dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    assert args.lam > 0 and args.tau > 0
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    sgn = -1.0 if args.sign == "minus" else 1.0

    # ---- tokenizer / prompt ---------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.teacher, trust_remote_code=True)
    if "llama" in (args.tokenizer or args.teacher).lower():
        tokenizer.pad_token_id = 128004
    else:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": CONTEXT.strip() + "\n"}]
    prompt_ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(prompt_ids):
        prompt_ids = prompt_ids["input_ids"]
    prompt_ids = prompt_ids.to(args.device)
    console.print(f"[dim]Prompt: {prompt_ids.shape[1]} tokens[/dim]")

    # ---- models ---------------------------------------------------------------
    teacher = load_model(args.teacher, torch.bfloat16, args.device)
    teacher.resize_token_embeddings(len(tokenizer))
    eos_ids = get_eos_ids(teacher, tokenizer)

    sdtype = getattr(torch, args.student_dtype)
    s_plus = load_model(args.proxy_student, sdtype, args.device)
    s_minus = load_model(args.proxy_student, sdtype, args.device)
    s_plus.resize_token_embeddings(len(tokenizer))
    s_minus.resize_token_embeddings(len(tokenizer))

    grads = torch.load(args.grad_path, map_location="cpu")
    for m, scale in ((s_plus, +args.eps), (s_minus, -args.eps)):
        used = perturb_(m, grads, scale)
        assert used == set(grads.keys()), f"Unused grads: {set(grads.keys()) ^ used}"
    del grads

    # ---- generate to completion; reservoir-sample N steps ---------------------
    # Reservoir sampling gives a uniform random subset of steps without knowing the
    # final length up front, and only keeps N full-vocab logit vectors in memory.
    caches = [None, None, None]
    cur = prompt_ids
    gen_ids, reservoir = [], []

    for step in range(args.max_new_tokens):
        t, caches[0] = fwd(teacher, cur, caches[0])
        sp, caches[1] = fwd(s_plus, cur, caches[1])
        sm, caches[2] = fwd(s_minus, cur, caches[2])
        V = min(t.shape[0], sp.shape[0], sm.shape[0])
        t, sp, sm = t[:V], sp[:V], sm[:V]

        ad = (args.lam / (2 * args.eps)) * (sp - sm)
        src = t if args.gen_mode == "baseline" else t + sgn * ad
        tok = sample_top_p(F.softmax(src / args.tau, dim=-1), args.top_p)

        item = (step, t.cpu(), ad.cpu(), tok)
        if len(reservoir) < args.num_steps:
            reservoir.append(item)
        else:
            j = random.randint(0, step)
            if j < args.num_steps:
                reservoir[j] = item

        gen_ids.append(tok)
        if tok in eos_ids:
            break
        cur = torch.tensor([[tok]], device=args.device)

    del teacher, s_plus, s_minus, caches
    torch.cuda.empty_cache()

    console.print(Panel(tokenizer.decode(gen_ids, skip_special_tokens=False),
                        title=f"Generated completion ({len(gen_ids)} tokens, gen_mode={args.gen_mode})",
                        border_style="cyan", title_align="left"))

    # ---- show comparisons at the sampled steps --------------------------------
    stats = []
    for step, t, ad, tok in sorted(reservoir, key=lambda x: x[0]):
        prefix = tokenizer.decode(gen_ids[:step])[-70:]
        console.rule(f"[bold]Step {step}/{len(gen_ids) - 1}[/bold]")
        console.print(f"[dim]...{prefix!r}[/dim]  →  sampled: [bold]{tok_str(tokenizer, tok)}[/bold]")
        title = (f"Top-{args.top_k}  |  lam={args.lam:.2e}  eps={args.eps:.2e}  "
                 f"tau={args.tau:g}  sign={args.sign}")
        s = show_comparison(tokenizer, t, ad, args, title)
        s["step"] = step
        stats.append(s)

    # ---- aggregate summary ----------------------------------------------------
    summ = Table(title="Summary across sampled steps", header_style="bold cyan", title_justify="left")
    summ.add_column("Step", justify="right")
    summ.add_column("KL(base‖mod)", justify="right")
    summ.add_column("TV", justify="right")
    summ.add_column("top-1 changed", justify="center")
    for s in stats:
        summ.add_row(str(s["step"]), f"{s['kl']:.3e}", f"{s['tv']:.4f}", "yes" if s["top1_changed"] else "no")
    n = len(stats)
    summ.add_section()
    summ.add_row("mean", f"{sum(s['kl'] for s in stats) / n:.3e}", f"{sum(s['tv'] for s in stats) / n:.4f}",
                 f"{sum(s['top1_changed'] for s in stats)}/{n}")
    console.print(summ)


if __name__ == "__main__":
    main()