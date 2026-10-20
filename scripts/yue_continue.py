"""YuE2 continues a score that starts with given bars, then sings and plays the whole score.

Run in the MLX port's folder (its `generate` module is imported from the working directory):
  <python> scripts/yue_continue.py --model 8bit --style ... --lyrics-file L --prefix-file P --seed 1 --out song.wav
Writes song.wav and song.abc (prefix + what the model wrote) and prints one JSON line (bars, seconds, stages).
Exit 3: the plan cannot be read by the app (song.abc is kept, no audio is made).  Exit 4: the score fills the
model's window.  A song longer than the window holds is cut at the end and the JSON line says so (cut, wanted_seconds).
Memory: MLX's buffer cache is capped (--cache-gb) and the decoder runs in tiles of --tile latent frames.
"""
import argparse
import json
import sys
import time
from pathlib import Path

TAIL_S, FRAME_S = 4.0, 0.04
MARGIN, MIN_FRAMES = 16, 250       # tokens kept free for the port's own markers; less than 10 s of audio is refused


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--style", required=True)
    ap.add_argument("--lyrics-file", type=Path, required=True)
    ap.add_argument("--prefix-file", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cfg-scale", type=float, default=2.0)
    ap.add_argument("--max-seconds", type=float, default=90.0)
    ap.add_argument("--cache-gb", type=float, default=0.2)
    ap.add_argument("--tile", type=int, default=32)
    ap.add_argument("--kv-bits", type=int, choices=[0, 8], default=0, help="8: the 8-bit KV cache (instmate.yuekv)")
    ap.add_argument("--no-continue", action="store_true", help="the prefix is the whole score: no plan stage")
    ap.add_argument("--plans", type=int, default=1, help="scores to write before the best is kept")
    ap.add_argument("--given-bars", type=int, default=0)
    ap.add_argument("--cut", choices=["", "first"], default="", help="first: sing only up to the end of the first chorus")
    ap.add_argument("--target-bars", type=int, default=0, help="a whole song: the plan nearest this length is kept")
    ap.add_argument("--min-bars", type=int, default=0, help="the plan may not end before the song has this many bars")
    ap.add_argument("--context", type=int, help="the model's token window (default: the port's own)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import mlx.core as mx
    sys.path.insert(0, ".")
    import generate as g
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from instmate import yueimport, yuesong
    if args.kv_bits == 8:
        port_model = sys.modules.get("yue2_model")             # imported by the real port's generate; a test stand-in has none
        if port_model is None:
            print("[kv] the port has no yue2_model: plain KV cache", file=sys.stderr, flush=True)
        else:
            from instmate import yuekv
            yuekv.patch(port_model, g)
    if args.cache_gb > 0:                                     # 0: MLX keeps its buffers (faster to reuse, about 18 GB at the peak)
        mx.set_cache_limit(int(args.cache_gb * 1e9))
    stages, cur = [], ["load", time.perf_counter()]

    def mark(name):
        stages.append({"stage": cur[0], "peak_gb": round(mx.get_peak_memory() / 1e9, 2), "seconds": round(time.perf_counter() - cur[1], 1)})
        mx.clear_cache()
        mx.reset_peak_memory()
        cur[0], cur[1] = name, time.perf_counter()

    def log(msg):
        print(msg, file=sys.stderr, flush=True)
        for tag, name in (("[semantic] prefix", "semantic"), ("[nar] ", "nar"), ("[vae]", "vae")):
            if msg.startswith(tag) and cur[0] != name:
                mark(name)

    lyrics, prefix = args.lyrics_file.read_text(encoding="utf-8"), args.prefix_file.read_text(encoding="utf-8")
    pipe = g.Yue2Pipeline(args.model, log=log)
    pipe.decode = lambda latents: mx.clip(mx.array(pipe.vae.decode_tiled(latents, core=args.tile)), -1, 1)
    tok = pipe.tokenizer
    mark("plan")
    request = g.token_prefix(tok, args.style, lyrics, "full")
    truncated, plans, plan_seed = False, [], None
    if args.no_continue:
        abc = prefix
    else:
        sampling = pipe.abc_sampling
        if args.min_bars:                                     # about 18 tokens a bar: the end token is held back until then
            least = max(0, args.min_bars - yuesong.count_bars(prefix)) * 18
            sampling = g.Sampling(**{**sampling.__dict__, "min_tokens": least, "max_tokens": max(sampling.max_tokens, least + 512)})
        best, source = None, prefix
        rounds = 4 if args.target_bars else 1
        for round_ in range(rounds):
            before = yuesong.count_bars(source)
            prompt = source + "\n% verse\nV: Vocal\n" if round_ else source
            plan_sampling = sampling
            if args.target_bars:
                room = (args.context or g.CONTEXT) - len(request) - len(tok.encode(prompt)) - MARGIN - MIN_FRAMES
                if room <= 0:
                    if best is None:
                        print("the score leaves no planning room in the model's window", file=sys.stderr)
                        sys.exit(4)
                    break
                plan_sampling = g.Sampling(**{**sampling.__dict__, "max_tokens": min(sampling.max_tokens, room),
                                             "min_tokens": min(sampling.min_tokens, room)})
            for k in range(max(1, args.plans)):
                candidate_seed = args.seed + round_ * max(1, args.plans) + k
                more, cut_short = g.generate_tokens(pipe.model, request + tok.encode(prompt), plan_sampling, candidate_seed, "abc",
                                                    on_token=pipe._progress("abc"))
                continuation = tok.decode(more)
                # Do not retain an empty prompted section when the model starts its own.
                base = prompt if continuation.strip() and not continuation.lstrip().startswith("%") else source
                text, repaired = yuesong.repair_generated_rests(base + continuation, keep=source)
                dropped = 0
                try:
                    text, dropped = yuesong.repair_plan(text, keep=source)
                except yuesong.YueSongError:
                    pass
                mark_ = yuesong.plan_score(text, args.given_bars, args.target_bars)
                plans.append({"seed": candidate_seed, "round": round_ + 1, "dropped_sections": dropped, "repaired_bars": repaired,
                              **{x: mark_.get(x) for x in ("ok", "score", "has_chorus", "variety", "bars", "length_fit", "error")}})
                if mark_["ok"] and (not args.target_bars or mark_["bars"] <= yuesong.MAX_BARS):
                    rank = yuesong.plan_rank(mark_, args.target_bars)
                    if best is None or rank > best[0]:
                        best = (rank, text, candidate_seed, cut_short)
            if best is None:
                break
            source = best[1]
            bars = yuesong.count_bars(source)
            if not args.target_bars or bars >= args.target_bars or bars <= before:
                break
        if best is None:
            args.out.with_suffix(".abc").write_text(text, encoding="utf-8")
            print("the plan cannot be used: " + (plans[-1].get("error") or "no valid plan within the 200-bar limit"), file=sys.stderr)
            sys.exit(3)
        _, abc, plan_seed, truncated = best
        args.seed = plan_seed
    if args.cut == "first":
        args.out.with_suffix(".full.abc").write_text(abc, encoding="utf-8")
        abc = yuesong.first_part(abc, args.given_bars)
    args.out.with_suffix(".abc").write_text(abc, encoding="utf-8")
    try:
        sheet, _, info = yueimport.read(abc)
    except yueimport.AbcError as exc:
        print(f"the plan cannot be read: {exc}", file=sys.stderr)
        sys.exit(3)
    num, den = (int(x) for x in sheet.meter.split("/"))
    wanted = round(min(info["bars"] * 60.0 / sheet.tempo * num * 4 / den + TAIL_S, args.max_seconds), 1)
    room = (args.context or g.CONTEXT) - len(request) - len(tok.encode(abc)) - MARGIN     # request + score + audio share one window
    if room < MIN_FRAMES:
        print(f"the score leaves no room for audio in the model's window ({room} frames)", file=sys.stderr)
        sys.exit(4)
    frames = min(int(wanted / FRAME_S), room)
    seconds = round(frames * FRAME_S, 1)
    sampling = g.Sampling(**{**pipe.semantic_sampling.__dict__, "max_tokens": frames, "min_tokens": min(pipe.semantic_sampling.min_tokens, frames)})
    audio, _ = pipe(args.style, lyrics, cot="full", seed=args.seed, abc=abc, cfg_scale=args.cfg_scale, semantic_sampling=sampling)
    mx.eval(audio)
    mark("end")
    g.write_wav(args.out, audio)
    made = round(audio.shape[0] / g.SAMPLE_RATE, 1)             # the model may end before the budget is used
    ended_early, seconds = made < seconds - 0.5, made
    print(json.dumps({"bars": info["bars"], "seconds": seconds, "wanted_seconds": wanted, "cut": frames < int(wanted / FRAME_S),
                      "ended_early": ended_early, "plan_truncated": bool(truncated), "plans": plans, "plan_seed": plan_seed, "stages": stages}))


if __name__ == "__main__":
    main()
