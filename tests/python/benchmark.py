#!/usr/bin/env python3
"""
benchmark.py — cppBpe vs rustbpe vs minbpe
===========================================
Run from the build/ directory (where cppBpe.so lives) with minbpe/ copied in:

    PYTHONPATH=. python3 benchmark.py [--corpus input.txt] [--vocab 4096] [--n 50]
"""

import argparse
import sys
import time
import statistics
from pathlib import Path


RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
RED    = "\033[31m"
WHITE  = "\033[97m"

def header(title: str) -> None:
    width = 62
    print()
    print(CYAN + "─" * width + RESET)
    print(CYAN + BOLD + f"  {title}" + RESET)
    print(CYAN + "─" * width + RESET)

def row(label: str, value: str, unit: str = "", winner: bool = False) -> None:
    color = GREEN + BOLD if winner else WHITE
    print(f"  {DIM}{label:<22}{RESET}  {color}{value:>10}{RESET}  {DIM}{unit}{RESET}")

def separator() -> None:
    print(f"  {DIM}{'·' * 40}{RESET}")

def bench(fn, n: int = 1) -> tuple[float, float]:
    """Run fn n times, return (mean_seconds, stdev_seconds)."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.mean(times), (statistics.stdev(times) if n > 1 else 0.0)

def try_import(name: str):
    try:
        return __import__(name)
    except ModuleNotFoundError:
        print(f"{YELLOW}  warning: '{name}' not found — skipping{RESET}")
        return None

def main() -> None:
    ap = argparse.ArgumentParser(description="BPE tokenizer benchmark")
    ap.add_argument("--corpus", default="input.txt",
                    help="Path to plain-text corpus (default: input.txt)")
    ap.add_argument("--vocab",  type=int, default=4096,
                    help="Target vocabulary size (default: 4096)")
    ap.add_argument("--n",      type=int, default=50,
                    help="Encode repetitions for throughput (default: 50)")
    ap.add_argument("--probe",  type=int, default=50_000,
                    help="Characters used for encode benchmark (default: 50_000)")
    args = ap.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        sys.exit(f"Corpus file not found: {corpus_path}")

    text  = corpus_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    print()
    print(BOLD + "  BPE Tokenizer Benchmark" + RESET)
    print(f"  {DIM}corpus : {corpus_path}  ({len(text):,} chars, {len(lines):,} lines){RESET}")
    print(f"  {DIM}vocab  : {args.vocab}   probe: {args.probe:,} chars   n: {args.n}{RESET}")

    cppBpe  = try_import("cppBpe")
    rustbpe = try_import("rustbpe")
    #minbpe  = try_import("minbpe")
    tiktoken = try_import("tiktoken")

    available = {
        "cppBpe":  cppBpe,
        "rustbpe": rustbpe,
        #"minbpe":  minbpe,
    }
    available = {k: v for k, v in available.items() if v is not None}

    if not available:
        sys.exit("No tokenizer libraries found.")

    header("1 · Training speed")

    trained: dict[str, object] = {}
    train_times: dict[str, float] = {}

    if "minbpe" in available:
        tok = available["minbpe"].RegexTokenizer()
        print(f"  {DIM}training minbpe…{RESET}", end="", flush=True)
        mean, _ = bench(lambda: tok.train(text, args.vocab))
        trained["minbpe"] = tok
        train_times["minbpe"] = mean
        print("\r", end="")
        row("minbpe (Python)", f"{mean:.2f}", "s")

    if "rustbpe" in available:
        tok = available["rustbpe"].Tokenizer()
        print(f"  {DIM}training rustbpe…{RESET}", end="", flush=True)
        mean, _ = bench(lambda: tok.train_from_iterator(iter(lines), vocab_size=args.vocab))
        trained["rustbpe"] = tok
        train_times["rustbpe"] = mean
        print("\r", end="")
        row("rustbpe (Rust)", f"{mean:.2f}", "s")

    if "cppBpe" in available:
        tok = available["cppBpe"].Tokenizer()
        print(f"  {DIM}training cppBpe…{RESET}", end="", flush=True)
        mean, _ = bench(lambda: tok.train_from_iterator(iter(lines), vocab_size=args.vocab))
        trained["cppBpe"] = tok
        train_times["cppBpe"] = mean
        print("\r", end="")
        row("cppBpe  (C++)", f"{mean:.2f}", "s")

    fastest_train = min(train_times, key=train_times.get)
    separator()
    for name, t in train_times.items():
        if name != fastest_train:
            speedup = t / train_times[fastest_train]
            row(f"  {fastest_train} speedup vs {name}", f"{speedup:.1f}×", "faster")

    header("2 · Encode throughput  (single string, 50 k chars)")

    probe = text[:args.probe]
    enc_rates: dict[str, float] = {}

    for name, tok in trained.items():
        if name == "minbpe":
            fn = lambda t=tok: t.encode(probe)
        else:
            fn = lambda t=tok: t.encode(probe)

        mean, std = bench(fn, args.n)
        rate = 1.0 / mean
        enc_rates[name] = rate
        row(name, f"{rate:.1f}", "calls/s")

    fastest_enc = max(enc_rates, key=enc_rates.get)
    separator()
    for name, r in enc_rates.items():
        if name != fastest_enc:
            row(f"  {fastest_enc} speedup vs {name}", f"{enc_rates[fastest_enc]/r:.1f}×", "faster")

    header("3 · Batch encode throughput  (100 × 500-char chunks)")

    chunks = [text[i:i+500] for i in range(0, args.probe, 500)]
    batch_rates: dict[str, float] = {}

    for name, tok in trained.items():
        if not hasattr(tok, "batch_encode"):
            row(name, "n/a", "(no batch_encode)")
            continue
        fn = lambda t=tok: t.batch_encode(chunks)
        mean, std = bench(fn, args.n)
        rate = 1.0 / mean
        batch_rates[name] = rate
        row(name, f"{rate:.1f}", "calls/s")

    if batch_rates:
        fastest_batch = max(batch_rates, key=batch_rates.get)
        separator()
        for name, r in batch_rates.items():
            if name != fastest_batch:
                row(f"  {fastest_batch} speedup vs {name}", f"{batch_rates[fastest_batch]/r:.1f}×", "faster")

    header("4 · Decode throughput")

    for name, tok in trained.items():
        ids = tok.encode(probe)
        fn = lambda t=tok, i=ids: t.decode(i)
        mean, _ = bench(fn, args.n)
        rate = 1.0 / mean
        row(name, f"{rate:.1f}", "calls/s")

    header("5 · Correctness — token ID agreement")

    all_ids: dict[str, list] = {}
    for name, tok in trained.items():
        all_ids[name] = tok.encode(probe)
        row(f"{name} token count", str(len(all_ids[name])), "tokens")

    separator()
    names = list(all_ids.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            match = all_ids[a] == all_ids[b]
            color = GREEN if match else RED
            symbol = "✓" if match else "✗"
            print(f"  {DIM}{a} == {b}:{RESET}  {color}{BOLD}{symbol}  {'MATCH' if match else 'MISMATCH'}{RESET}")

    header("6 · Vocabulary overlap")

    vocabs: dict[str, set] = {}
    for name, tok in trained.items():
        ranks = tok.get_mergeable_ranks()
        vocabs[name] = {bytes(b) if isinstance(b, (list, bytes)) else b
                        for b, _ in ranks}
        row(f"{name} vocab size", str(len(vocabs[name])), "tokens")

    separator()
    vnames = list(vocabs.keys())
    for i in range(len(vnames)):
        for j in range(i + 1, len(vnames)):
            a, b = vnames[i], vnames[j]
            inter = len(vocabs[a] & vocabs[b])
            union = len(vocabs[a] | vocabs[b])
            pct   = 100.0 * inter / union if union else 100.0
            color = GREEN if pct > 99 else (YELLOW if pct > 90 else RED)
            print(f"  {DIM}{a} ∩ {b}:{RESET}  {color}{BOLD}{inter}/{union}  ({pct:.1f}% overlap){RESET}")

    header("7 · tiktoken export roundtrip")

    if tiktoken is None:
        print(f"  {YELLOW}skipped — tiktoken not installed{RESET}")
    else:
        for name, tok in trained.items():
            try:
                enc = tiktoken.Encoding(
                    name=f"{name}_bench",
                    pat_str=tok.get_pattern(),
                    mergeable_ranks={
                        (bytes(k) if isinstance(k, list) else k): v
                        for k, v in tok.get_mergeable_ranks()
                    },
                    special_tokens={},
                )
                tik_ids  = enc.encode(probe)
                orig_ids = tok.encode(probe)
                match    = tik_ids == orig_ids
                color    = GREEN if match else RED
                symbol   = "✓" if match else "✗"
                decoded  = enc.decode(tik_ids)
                rt_ok    = decoded == probe
                rt_color = GREEN if rt_ok else RED
                print(f"  {DIM}{name}:{RESET}")
                print(f"    ids match tiktoken:  {color}{BOLD}{symbol}  {'OK' if match else 'FAIL'}{RESET}")
                print(f"    decode roundtrip:    {rt_color}{BOLD}{'✓  OK' if rt_ok else '✗  FAIL'}{RESET}")
            except Exception as e:
                print(f"  {RED}{name}: export failed — {e}{RESET}")

    header("8 · Peak memory during training  (tracemalloc)")

    import tracemalloc

    for name, lib in available.items():
        if name == "minbpe":
            def _train():
                t = lib.RegexTokenizer()
                t.train(text, args.vocab)
        elif name == "rustbpe":
            def _train():
                t = lib.Tokenizer()
                t.train_from_iterator(iter(lines), vocab_size=args.vocab)
        else:
            def _train():
                t = lib.Tokenizer()
                t.train_from_iterator(iter(lines), vocab_size=args.vocab)

        tracemalloc.start()
        _train()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        row(name, f"{peak / 1_048_576:.1f}", "MiB peak")

    header("Summary")

    if train_times:
        best = min(train_times, key=train_times.get)
        print(f"  {GREEN}{BOLD}Fastest training : {best}  ({train_times[best]:.2f}s){RESET}")
    if enc_rates:
        best = max(enc_rates, key=enc_rates.get)
        print(f"  {GREEN}{BOLD}Fastest encode   : {best}  ({enc_rates[best]:.1f} calls/s){RESET}")
    if batch_rates:
        best = max(batch_rates, key=batch_rates.get)
        print(f"  {GREEN}{BOLD}Fastest batch    : {best}  ({batch_rates[best]:.1f} calls/s){RESET}")

    print()
    print(f"  {DIM}Run with --help to see all options.{RESET}")
    print()


if __name__ == "__main__":
    main()