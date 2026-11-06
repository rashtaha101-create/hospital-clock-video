#!/usr/bin/env python3
"""Headless pipeline (runs on GitHub Actions / any CPU box):
voice-over -> word timing (Whisper, CPU) -> backgrounds (cached or generated) -> depth -> render shots -> captions+SFX -> MP4."""
import os, sys, json, time, argparse, importlib, subprocess
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E


def log(*a): print(*a, flush=True)


def get_word_times(S, audio, dur, model_size="small"):
    words = E.tokenize(S.SCRIPT)
    try:
        from faster_whisper import WhisperModel
        wm = WhisperModel(model_size, device="cpu", compute_type="int8")
        segs, _ = wm.transcribe(audio, language="en", word_timestamps=True)
        asr = [(w.word.strip(), w.start, w.end) for s in segs for w in s.words]
        log(f"whisper heard {len(asr)} words, script has {len(words)}")
        return words, E.align_words(words, asr, dur)
    except Exception as e:
        log("whisper unavailable, using pause-based estimate:", e)
        return words, E.estimate_words_from_silence(audio, words, dur)


def ensure_backgrounds(S, adir, mode, quality, seed, regen):
    bdir, ddir = f"{adir}/backgrounds", f"{adir}/depth"
    os.makedirs(bdir, exist_ok=True); os.makedirs(ddir, exist_ok=True)
    todo = [k for k in S.BACKGROUNDS if k in regen or not os.path.exists(f"{bdir}/{k}.png")]
    if todo and mode == "never":
        sys.exit(f"missing backgrounds {todo} and --gen-backgrounds=never. Commit PNGs to {bdir} or allow generation.")
    if todo:
        import torch
        from diffusers import AutoPipelineForText2Image, StableDiffusionXLPipeline
        gpu = torch.cuda.is_available()
        dt = torch.float16 if gpu else torch.float32
        if quality == "turbo" or not gpu:   # CPU: turbo only (SDXL base would take hours)
            pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=dt, variant="fp16" if gpu else None)
            size, kw = (512, 896), dict(num_inference_steps=4, guidance_scale=0.0)
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=dt, variant="fp16", use_safetensors=True)
            size, kw = (768, 1344), dict(num_inference_steps=28, guidance_scale=6.5, negative_prompt=S.NEG)
        pipe = pipe.to("cuda" if gpu else "cpu"); pipe.enable_vae_slicing()
        for k in todo:
            t0 = time.time()
            g = torch.Generator("cuda" if gpu else "cpu").manual_seed(seed + list(S.BACKGROUNDS).index(k) + (1000 if k in regen else 0))
            img = pipe(prompt=S.BACKGROUNDS[k] + ", " + S.STYLE, width=size[0], height=size[1], generator=g, **kw).images[0]
            img.save(f"{bdir}/{k}.png"); log(f"background {k} ({time.time()-t0:.0f}s)")
        del pipe
    need = [k for k in S.BACKGROUNDS if k in regen or not os.path.exists(f"{ddir}/{k}.npy")]
    if need:
        import torch
        from PIL import Image
        from transformers import pipeline
        dp = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=0 if torch.cuda.is_available() else -1)
        for k in need:
            d = np.array(dp(Image.open(f"{bdir}/{k}.png").convert("RGB"))["depth"]).astype(np.float32)
            np.save(f"{ddir}/{k}.npy", (d - d.min()) / (d.max() - d.min() + 1e-6)); log("depth", k)


def load_plan(path):
    p = json.load(open(path)); return p["dur"], [tuple(x) for x in p["wt"]], p["starts"], p["ends"]


def write_frame(path, frame, ext):
    if ext == "jpg": cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else: cv2.imwrite(path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "prep", "frames", "assemble"], default="all",
                    help="prep: timing+assets+plan.json | frames: render frame images (one --shot or all) | assemble: frames->mp4")
    ap.add_argument("--story", default="hospital_clock")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--assets", default="assets/hospital_clock")
    ap.add_argument("--work", default="work", help="plan.json + frames/ live here")
    ap.add_argument("--frames-dir", default="", help="read/write frame images here (default: <work>/frames)")
    ap.add_argument("--shot", default="", help="shot id(s) for --stage frames, comma list; empty = all")
    ap.add_argument("--frame-ext", choices=["jpg", "png"], default="jpg")
    ap.add_argument("--out", default="out/video.mp4")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--gen-backgrounds", choices=["auto", "never"], default="auto")
    ap.add_argument("--quality", choices=["base", "turbo"], default="turbo")
    ap.add_argument("--regen", default="", help="comma list of background names to re-roll")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tail", type=float, default=0.85)
    ap.add_argument("--no-sfx", action="store_true")
    ap.add_argument("--cap-size", type=int, default=78)
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, f"{root}/stories")
    S = importlib.import_module(a.story)
    os.makedirs(a.work, exist_ok=True)
    plan_path = f"{a.work}/plan.json"
    E.W, E.H = int(1080 * a.scale) // 2 * 2, int(1920 * a.scale) // 2 * 2
    words = E.tokenize(S.SCRIPT)

    if a.stage in ("all", "prep"):
        dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", a.audio]))
        words, wt = get_word_times(S, a.audio, dur)
        starts = [0.0 if s["anchor"] == 0 else wt[s["anchor"]][0] for s in S.SHOTS]
        ends = starts[1:] + [dur + a.tail]
        for s, x, y in zip(S.SHOTS, starts, ends): log(f"shot {s['id']:>2} {x:6.2f}-{y:6.2f} {s['bg']:<16} {s['tin']}")
        json.dump(dict(dur=dur, wt=wt, starts=starts, ends=ends, shots=[s["id"] for s in S.SHOTS]), open(plan_path, "w"))
        ensure_backgrounds(S, a.assets, a.gen_backgrounds, a.quality, a.seed, [r for r in a.regen.split(",") if r])
        if a.stage == "prep": return
    dur, wt, starts, ends = load_plan(plan_path)
    fdir = a.frames_dir or f"{a.work}/frames"; os.makedirs(fdir, exist_ok=True)
    want = [int(x) for x in a.shot.split(",") if x]
    bg_cache = {}
    def bg_for(name):
        if name not in bg_cache:
            bg_cache[name] = E.BG(cv2.imread(f"{a.assets}/backgrounds/{name}.png"), np.load(f"{a.assets}/depth/{name}.npy"))
        return bg_cache[name]

    if a.stage in ("all", "frames"):
        for i, s in enumerate(S.SHOTS):
            if want and s["id"] not in want: continue
            tout = S.SHOTS[i + 1]["tin"] if i + 1 < len(S.SHOTS) else "cut"
            ghost = None
            if s["tin"] == "dissolve" and i > 0:   # frame mode: re-render the previous shot's final frame as the ghost
                pv = S.SHOTS[i - 1]; _, pn = E.shot_span(starts[i - 1], ends[i - 1])
                gp = f"{a.work}/ghost_{pv['id']}.png"
                E.render_shot(pv, bg_for(pv["bg"]), wt, starts[i - 1], ends[i - 1], None, tin=pv["tin"], tout=s["tin"],
                              frame_range=(pn - 1, pn), sink=lambda *_: None, ghost_save=gp)
                ghost = cv2.imread(gp)
            t0 = time.time(); cnt = [0]
            def sink(gi, frame):
                write_frame(f"{fdir}/{gi:06d}.{a.frame_ext}", frame, a.frame_ext); cnt[0] += 1
            E.render_shot(s, bg_for(s["bg"]), wt, starts[i], ends[i], None, tin=s["tin"], tout=tout, ghost=ghost, sink=sink)
            log(f"shot {s['id']:>2}: {cnt[0]} frames in {time.time()-t0:.0f}s")

    if a.stage in ("all", "assemble"):
        total = int(round(ends[-1] * E.FPS))
        chunks = E.caption_chunks(words, wt)
        whips = [starts[i] for i, s in enumerate(S.SHOTS) if s["tin"] in ("whip_l", "whip_r", "push")]
        pops = [starts[i] + E.tsec(e["t"], starts[i], wt) for i, s in enumerate(S.SHOTS) for e in s.get("events", []) if e["type"] == "speed"]
        mix = f"{a.work}/mix.wav"
        E.make_sfx(mix, a.audio, [] if a.no_sfx else whips, [] if a.no_sfx else pops)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        n = E.finalize_frames(fdir, total, chunks, mix, a.out, ext=a.frame_ext, cap_size=int(a.cap_size * a.scale))
        log(f"done: {n} frames -> {a.out}")


if __name__ == "__main__":
    main()
