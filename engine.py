# engine.py - 2.5D diorama renderer: depth-parallax camera, depth-of-field, 2D character sprites, VFX, transitions,
# word-by-word captions, whoosh SFX and final export. Pure numpy/OpenCV/Pillow -> runs on a free Colab CPU.
import os, re, math, json, random, difflib, subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import characters as ch

W, H, FPS = 1080, 1920, 30

# ------------------------------------------------------------------ easing
def _c01(t): return 0.0 if t < 0 else 1.0 if t > 1 else t
EASE = {
    "lin": lambda t: t,
    "io": lambda t: t * t * (3 - 2 * t),
    "out": lambda t: 1 - (1 - t) ** 3,
    "in": lambda t: t ** 3,
    "back": lambda t: 1 + 2.70158 * (t - 1) ** 3 + 1.70158 * (t - 1) ** 2,
}
def ease(name, t): return EASE[name](_c01(t))
def lerp(a, b, k): return a + (b - a) * k

# ------------------------------------------------------------------ timing helpers
def tsec(spec, shot_start, wt):
    """spec: float (seconds into shot) | ('w', idx) | ('w', idx, offset_seconds)."""
    if isinstance(spec, (int, float)): return float(spec)
    if spec[0] == "w":
        off = spec[2] if len(spec) > 2 else 0.0
        return wt[spec[1]][0] - shot_start + off
    raise ValueError(spec)

# ------------------------------------------------------------------ camera
def cam_state(cam, u, dur, shot_start, wt):
    t = u / dur if dur > 0 else 0
    k = ease(cam.get("ease", "io"), t)
    z = lerp(*cam.get("z", (1, 1)), k)
    x = lerp(*cam.get("x", (0, 0)), k)
    y = lerp(*cam.get("y", (0, 0)), k)
    r = lerp(*cam.get("rot", (0, 0)), k)
    hh = cam.get("hand", 0.0)
    if hh:
        x += hh * 0.006 * (math.sin(u * 2.3) + 0.6 * math.sin(u * 5.7 + 1.3))
        y += hh * 0.006 * (math.sin(u * 1.9 + 0.7) + 0.6 * math.sin(u * 6.1))
        r += hh * 0.35 * math.sin(u * 1.6 + 2.0)
    for (ps, amp) in cam.get("punches", []):
        pu = tsec(ps, shot_start, wt)
        if u >= pu: z += amp * ease("back", (u - pu) / 0.12)
    f0, f1 = cam.get("focus", (0.5, 0.5))
    ft0 = tsec(cam.get("focus_t", (0, dur))[0], shot_start, wt)
    ft1 = tsec(cam.get("focus_t", (0, dur))[1], shot_start, wt)
    focus = lerp(f0, f1, ease("io", (u - ft0) / max(1e-3, ft1 - ft0)))
    return z, x, y, r, focus

class BG:
    """Background image + depth (0=far, 1=near) rendered through a parallax camera with depth-of-field."""
    def __init__(self, img_bgr, depth, scale=1.25):
        self.CW, self.CH = int(W * scale), int(H * scale)
        self.img = cv2.resize(img_bgr, (self.CW, self.CH), interpolation=cv2.INTER_CUBIC)
        d = depth.astype(np.float32)
        d = (d - d.min()) / max(1e-6, d.max() - d.min())
        d = cv2.resize(d, (self.CW, self.CH), interpolation=cv2.INTER_LINEAR)
        self.depth = cv2.GaussianBlur(d, (0, 0), 4)
        self.b1 = cv2.GaussianBlur(self.img, (0, 0), 7)
        self.b2 = cv2.GaussianBlur(self.img, (0, 0), 20)
        self.uu, self.vv = np.meshgrid(np.arange(W, dtype=np.float32) - W / 2, np.arange(H, dtype=np.float32) - H / 2)

    def maps(self, z, x, y, r, par, d_out=None):
        c, s = math.cos(math.radians(r)), math.sin(math.radians(r))
        ru, rv = self.uu * c - self.vv * s, self.uu * s + self.vv * c
        if d_out is None:
            return self.CW / 2 + x * self.CW + ru / z, self.CH / 2 + y * self.CH + rv / z
        zz = np.clip(z * (1 + par * 0.5 * (d_out - 0.5) * (z - 1)), 0.4, 6)
        return (self.CW / 2 + x * self.CW + ru / zz + (d_out - 0.5) * par * x * self.CW,
                self.CH / 2 + y * self.CH + rv / zz + (d_out - 0.5) * par * y * self.CH)

    def render(self, z, x, y, r, focus, dof, par):
        mx0, my0 = self.maps(z, x, y, r, par)
        d_out = cv2.remap(self.depth, mx0, my0, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mx, my = self.maps(z, x, y, r, par, d_out)
        sharp = cv2.remap(self.img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        if dof <= 0.01: return sharp
        coc = np.clip(np.abs(d_out - focus) * dof * 3.0, 0, 1)
        a = np.clip(1 - 2 * coc, 0, 1)[..., None]
        c3 = np.clip(2 * coc - 1, 0, 1)[..., None]
        b1 = cv2.remap(self.b1, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        b2 = cv2.remap(self.b2, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        out = sharp.astype(np.float32) * a + b1.astype(np.float32) * (1 - a - c3) + b2.astype(np.float32) * c3
        return out.astype(np.uint8)

def project(bg, z, x, y, r, par, xf, yf, d):
    """World point (xf,yf as fraction of the z=1 window) at depth d -> screen px and local zoom."""
    sx = bg.CW / 2 + (xf - 0.5) * W
    sy = bg.CH / 2 + (yf - 0.5) * H
    zz = float(np.clip(z * (1 + par * 0.5 * (d - 0.5) * (z - 1)), 0.4, 6))
    ru = (sx - bg.CW / 2 - x * bg.CW - (d - 0.5) * par * x * bg.CW) * zz
    rv = (sy - bg.CH / 2 - y * bg.CH - (d - 0.5) * par * y * bg.CH) * zz
    c, s = math.cos(math.radians(r)), math.sin(math.radians(r))
    return W / 2 + ru * c + rv * s, H / 2 - ru * s + rv * c, zz

# ------------------------------------------------------------------ sprite compositing
def blit(frame, spr, ax, ay, sx, sy, rot=0.0, blur=0.0, alpha=1.0, shadow=0.0):
    """Composite a BGRA sprite so its bottom-centre anchor lands on (ax, ay). Handles scale, squash, lean, blur."""
    h0, w0 = spr.shape[:2]
    nw, nh = max(1, int(w0 * sx)), max(1, int(h0 * sy))
    if nw > 6000 or nh > 6000 or nw < 2 or nh < 2: return
    s = cv2.resize(spr, (nw, nh), interpolation=cv2.INTER_AREA if (sx < 1 or sy < 1) else cv2.INTER_LINEAR)
    pad = int(blur * 3) + 2
    if abs(rot) > 0.05: pad = max(pad, int(max(nw, nh) * abs(math.sin(math.radians(rot))) * 1.3) + 3)
    if pad > 2:
        s = cv2.copyMakeBorder(s, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    if abs(rot) > 0.05:
        M = cv2.getRotationMatrix2D((nw / 2 + pad, nh + pad), -rot, 1.0)
        s = cv2.warpAffine(s, M, (s.shape[1], s.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(0, 0, 0, 0))
    if blur > 0.3: s = cv2.GaussianBlur(s, (0, 0), blur)
    ox, oy = int(round(ax - nw / 2 - pad)), int(round(ay - nh - pad))
    sh_, sw_ = s.shape[:2]
    x0, y0, x1, y1 = max(0, ox), max(0, oy), min(W, ox + sw_), min(H, oy + sh_)
    if x1 <= x0 or y1 <= y0: return
    if shadow > 0:
        rx, ry = int(nw * 0.30), int(nw * 0.055) + 3
        sx0, sy0 = max(0, int(ax) - rx * 2), max(0, int(ay) - ry * 3)
        sx1, sy1 = min(W, int(ax) + rx * 2), min(H, int(ay) + ry * 3)
        if sx1 > sx0 and sy1 > sy0:
            mask = np.zeros((sy1 - sy0, sx1 - sx0), np.float32)
            cv2.ellipse(mask, (int(ax) - sx0, int(ay) - sy0 - 4), (rx, ry), 0, 0, 360, 1.0, -1)
            mask = cv2.GaussianBlur(mask, (0, 0), 9)[..., None]
            frame[sy0:sy1, sx0:sx1] = (frame[sy0:sy1, sx0:sx1].astype(np.float32) * (1 - shadow * mask)).astype(np.uint8)
    crop = s[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    a = (crop[..., 3:4].astype(np.float32) / 255.0) * alpha
    reg = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = np.clip(reg * (1 - a) + crop[..., :3].astype(np.float32) * alpha, 0, 255).astype(np.uint8)

_prem_cache = {}
def premult_cached(spr, flip=False):
    key = (id(spr), flip)
    if key not in _prem_cache:
        p = spr[:, ::-1] if flip else spr
        p = np.ascontiguousarray(p).copy()
        p[..., :3] = (p[..., :3].astype(np.float32) * (p[..., 3:4] / 255.0)).astype(np.uint8)
        _prem_cache[key] = (spr, p)   # keep spr alive so id() stays unique
        if len(_prem_cache) > 200: _prem_cache.pop(next(iter(_prem_cache)))
    return _prem_cache[key][1]

_prop_cache = {}
def get_prop(name, S=2):
    if (name, S) not in _prop_cache: _prop_cache[(name, S)] = ch.draw_prop(name, S)
    return _prop_cache[(name, S)]

# ------------------------------------------------------------------ character animation
def char_state(cd, u, shot_start, wt):
    def tl(key, default):
        items = cd.get(key) or [(0.0, default)]
        cur, tchg = items[0][1], -9.0
        for (ts, val) in items:
            t = tsec(ts, shot_start, wt)
            if u >= t: cur, tchg = val, t
        return cur, tchg
    expr, t_expr = tl("expr", "neutral")
    pose, t_pose = tl("pose", "down")
    path = cd.get("path")
    xf, yf = cd["x"], cd["y"]
    moving = False
    if path:
        t0 = tsec(path[0][0], shot_start, wt); t1 = tsec(path[1][0], shot_start, wt)
        k = ease(cd.get("path_ease", "io"), (u - t0) / max(1e-3, t1 - t0))
        xf = lerp(path[0][1], path[1][1], k); yf = lerp(path[0][2], path[1][2], k)
        moving = 0 < k < 1
    bob = 0.004 * math.sin(u * 5.0 + cd.get("phase", 0)) if not (moving and cd.get("walk")) else 0.010 * abs(math.sin(u * 9.5))
    legs = int(u * 8) % 4 if (moving and cd.get("walk")) else 0
    pop_dt = u - max(t_expr, t_pose)
    sq = 0.0
    if 0 <= pop_dt < 0.28 and max(t_expr, t_pose) > 0.02:
        sq = 0.09 * math.sin(pop_dt / 0.28 * math.pi) * math.exp(-pop_dt * 4)
    wig = 14 * math.sin(u * 14) if pose == "wave" or cd.get("wiggle") else 0
    wig = int(round(wig / 7.0) * 7)
    lean = cd.get("lean", 0.0) * math.sin(u * 1.4 + cd.get("phase", 0))
    return dict(expr=expr, pose=pose, xf=xf, yf=yf + bob, legs=legs, sx=1 - sq * 0.8, sy=1 + sq, wig=wig, lean=lean)

# ------------------------------------------------------------------ VFX
def speed_lines(frame, seed, strength=0.55):
    rnd = random.Random(seed)
    ov = np.zeros((H, W), np.float32)
    cx, cy = W / 2, H * 0.42
    diag = math.hypot(W, H)
    for _ in range(70):
        a = rnd.uniform(0, 2 * math.pi)
        r0 = rnd.uniform(0.28, 0.5) * diag; r1 = r0 + rnd.uniform(0.25, 0.55) * diag
        wdt = rnd.uniform(0.004, 0.013)
        p0 = (cx + math.cos(a) * r0, cy + math.sin(a) * r0)
        pa = (cx + math.cos(a - wdt) * r1, cy + math.sin(a - wdt) * r1)
        pb = (cx + math.cos(a + wdt) * r1, cy + math.sin(a + wdt) * r1)
        cv2.fillConvexPoly(ov, np.array([p0, pa, pb], np.int32), 1.0, cv2.LINE_AA)
    ov = cv2.GaussianBlur(ov, (0, 0), 1.2)[..., None] * strength
    frame[:] = np.clip(frame.astype(np.float32) * (1 - ov) + 255 * ov, 0, 255).astype(np.uint8)

def pulse_ring(frame, cx, cy, p, color=(60, 190, 255)):
    if not 0 <= p <= 1: return
    r = int(30 + 230 * ease("out", p)); a = (1 - p) ** 1.5
    ov = np.zeros((H, W), np.float32)
    cv2.circle(ov, (int(cx), int(cy)), r, 1.0, int(14 * (1 - p) + 4), cv2.LINE_AA)
    cv2.circle(ov, (int(cx), int(cy)), int(r * 0.55), 0.6, -1, cv2.LINE_AA)
    ov = cv2.GaussianBlur(ov, (0, 0), 8)[..., None] * a
    col = np.array(color, np.float32)
    frame[:] = np.clip(frame.astype(np.float32) * (1 - 0.7 * ov) + col * 1.2 * ov, 0, 255).astype(np.uint8)

def sparkles(frame, cx, cy, spread, u, n=9, seed=3):
    rnd = random.Random(seed)
    ov = np.zeros((H, W), np.float32)
    for i in range(n):
        px, py = cx + rnd.uniform(-spread, spread), cy + rnd.uniform(-spread, spread)
        ph, sp = rnd.uniform(0, 6.28), rnd.uniform(5, 9)
        s = max(0, math.sin(u * sp + ph)) ** 2 * rnd.uniform(18, 46)
        if s < 2: continue
        pts = np.array([[px, py - s], [px + s * .18, py - s * .18], [px + s, py], [px + s * .18, py + s * .18],
                        [px, py + s], [px - s * .18, py + s * .18], [px - s, py], [px - s * .18, py - s * .18]], np.int32)
        cv2.fillConvexPoly(ov, pts, 1.0, cv2.LINE_AA)
    ov = cv2.GaussianBlur(ov, (0, 0), 1.5)[..., None]
    frame[:] = np.clip(frame.astype(np.float32) + ov * 255 * 0.9, 0, 255).astype(np.uint8)

def dust(frame, u, seed=7, n=46, strength=0.35):
    rnd = random.Random(seed)
    ov = np.zeros((H, W), np.float32)
    for _ in range(n):
        px = (rnd.uniform(0, W) + u * rnd.uniform(-14, 18)) % W
        py = (rnd.uniform(0, H) + u * rnd.uniform(-30, -6)) % H
        cv2.circle(ov, (int(px), int(py)), rnd.randint(2, 6), rnd.uniform(0.3, 1.0), -1, cv2.LINE_AA)
    ov = cv2.GaussianBlur(ov, (0, 0), 2.0)[..., None] * strength
    frame[:] = np.clip(frame.astype(np.float32) + ov * 255, 0, 255).astype(np.uint8)

_vig = {}
def vignette(frame, amt):
    if amt <= 0: return
    if "m" not in _vig:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        _vig["m"] = (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)[..., None] / 2.0
    frame[:] = np.clip(frame.astype(np.float32) * (1 - amt * _vig["m"] ** 1.3), 0, 255).astype(np.uint8)

def bloom(frame, amt):
    if amt <= 0: return
    sm = cv2.resize(frame, (W // 4, H // 4), interpolation=cv2.INTER_AREA).astype(np.float32)
    br = np.clip(sm - 170, 0, 255) * 3.0
    br = cv2.GaussianBlur(br, (0, 0), 9)
    br = cv2.resize(br, (W, H), interpolation=cv2.INTER_LINEAR)
    frame[:] = np.clip(frame.astype(np.float32) + br * amt, 0, 255).astype(np.uint8)

# ------------------------------------------------------------------ transitions
def motion_blur_h(img, k):
    k = int(k) | 1
    return img if k <= 1 else cv2.blur(img, (k, 1), borderType=cv2.BORDER_REPLICATE)

def radial_blur(img, s):
    acc = img.astype(np.float32)
    n = 6
    for i in range(1, n):
        sc = 1 + s * i / (n - 1)
        M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, sc)
        acc += cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    return (acc / n).astype(np.uint8)

def apply_trans(frame, kind, kn, side):
    """kn in 0..1 (strength). side='out' (leaving) or 'in' (arriving)."""
    if kn <= 0: return frame
    if kind in ("whip_l", "whip_r"):
        d = -1 if kind == "whip_l" else 1
        if side == "in": d = -d
        shift = d * kn * W * 0.22 * (1 if side == "out" else -1) * -1
        M = np.float32([[1, 0, shift], [0, 1, 0]])
        f = cv2.warpAffine(frame, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
        f = motion_blur_h(f, 1 + kn * 150)
        return np.clip(f.astype(np.float32) * (1 + 0.12 * kn), 0, 255).astype(np.uint8)
    if kind == "push":
        s = 0.22 * kn
        M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, 1 + (0.30 * kn if side == "out" else 0.30 * kn))
        f = cv2.warpAffine(frame, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
        return radial_blur(f, s)
    return frame

# ------------------------------------------------------------------ shot renderer
def shot_span(t_start, t_end):
    f0, f1 = int(round(t_start * FPS)), int(round(t_end * FPS))
    return f0, max(1, f1 - f0)


def render_shot(shot, bg, wt, t_start, t_end, out_path, tin="cut", tout="cut", ghost=None, ghost_save=None, n_trans=6,
                progress=None, frame_range=None, sink=None):
    """sink(global_frame_index, frame_bgr) receives frames instead of an mp4 (frame-by-frame mode)."""
    f0, f1 = int(round(t_start * FPS)), int(round(t_end * FPS))
    n = max(1, f1 - f0)
    dur = n / FPS
    cam = shot.get("cam", {})
    fx = shot.get("fx", {})
    par, dof = cam.get("par", 0.7), cam.get("dof", 0.4)
    proc = None if sink else subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
         "-pix_fmt", "yuv420p", out_path], stdin=subprocess.PIPE)
    ev = shot.get("events", [])
    last = None
    for i in (range(*frame_range) if frame_range else range(n)):
        u = i / FPS
        z, x, y, r, focus = cam_state(cam, u, dur, t_start, wt)
        frame = bg.render(z, x, y, r, focus, dof, par)
        sprites = []
        for cd in shot.get("chars", []):
            st = char_state(cd, u, t_start, wt)
            sprites.append((cd.get("depth", 0.5), cd, st))
        for (dep, cd, st) in sorted(sprites, key=lambda s: s[0]):
            sx_, sy_, zz = project(bg, z, x, y, r, par, st["xf"], st["yf"], dep)
            bust = cd.get("bust", False)
            if cd["kind"] == "prop":
                spr = get_prop(cd["name"]); base_h = cd["h"] * H
            else:
                S = 4 if bust else 2
                spr = ch.draw_character(cd["kind"], st["expr"], st["pose"], st["legs"], S, bust, st["wig"])
                base_h = cd["h"] * H
            scale = base_h * zz / spr.shape[0]
            spr = premult_cached(spr, cd.get("flip", False))
            coc = float(np.clip(abs(dep - focus) * dof * 3.0, 0, 1))
            blur = coc * 16 * zz
            blit(frame, spr, sx_, sy_, scale * st["sx"], scale * st["sy"], rot=st["lean"] + r, blur=blur,
                 shadow=0.0 if (bust or cd["kind"] in ("prop", "patient_bed")) else 0.30)
        # --- timed VFX
        for e in ev:
            te = tsec(e["t"], t_start, wt)
            if e["type"] == "speed" and 0 <= u - te < 0.36:
                speed_lines(frame, seed=int((u - te) * FPS // 2) + 11, strength=e.get("strength", 0.55))
            if e["type"] == "pulse":
                px, py, _ = project(bg, z, x, y, r, par, e["pos"][0], e["pos"][1], e.get("depth", 0.5))
                pulse_ring(frame, px, py, (u - te) / 0.5, e.get("color", (60, 190, 255)))
            if e["type"] == "flash" and 0 <= u - te < 0.18:
                a = 0.55 * (1 - (u - te) / 0.18)
                frame[:] = np.clip(frame.astype(np.float32) * (1 - a) + 255 * a, 0, 255).astype(np.uint8)
        if fx.get("sparkle"):
            sp = fx["sparkle"]
            sparkles(frame, sp["pos"][0] * W, sp["pos"][1] * H, sp.get("spread", 260), u, seed=sp.get("seed", 3))
        if fx.get("dust"): dust(frame, u, strength=fx.get("dust_amt", 0.35))
        g = fx.get("grade")
        if g:
            frame = np.clip(frame.astype(np.float32) * np.array(g, np.float32)[None, None, :], 0, 255).astype(np.uint8)
        bloom(frame, fx.get("bloom", 0.0))
        vignette(frame, fx.get("vignette", 0.30))
        last = frame
        # --- transitions
        if tin in ("whip_l", "whip_r", "push") and i < n_trans:
            frame = apply_trans(frame, tin, ((n_trans - i) / n_trans) ** 1.4, "in")
        if tout in ("whip_l", "whip_r", "push") and i >= n - n_trans:
            frame = apply_trans(frame, tout, ((i - (n - n_trans) + 1) / n_trans) ** 1.4, "out")
        if tin == "dissolve" and ghost is not None and i < 9:
            a = 1 - ease("io", (i + 1) / 9)
            frame = cv2.addWeighted(frame, 1 - a, ghost, a, 0)
        if sink: sink(f0 + i, frame)
        else: proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        if progress and i % 30 == 0: progress(i, n)
    if proc: proc.stdin.close(); proc.wait()
    if ghost_save is not None and last is not None: cv2.imwrite(ghost_save, last)
    return n

# ------------------------------------------------------------------ script <-> audio alignment
def tokenize(script):
    return [w for w in script.replace("\n", " ").split(" ") if w.strip()]

def _norm(w): return re.sub(r"[^a-z0-9]", "", w.lower())

def align_words(script_words, asr_words, total_dur):
    """asr_words: list of (text,start,end). Returns [(start,end)] for each script word (monotonic)."""
    sn, an = [_norm(w) for w in script_words], [_norm(w[0]) for w in asr_words]
    n = len(sn)
    res = [None] * n
    sm = difflib.SequenceMatcher(None, sn, an, autojunk=False)
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            res[blk.a + k] = (asr_words[blk.b + k][1], asr_words[blk.b + k][2])
    known = [i for i in range(n) if res[i] is not None]
    if not known:
        t0 = asr_words[0][1] if asr_words else 0.0; t1 = asr_words[-1][2] if asr_words else total_dur
        return [(t0 + (t1 - t0) * i / n, t0 + (t1 - t0) * (i + 1) / n) for i in range(n)]
    for i in range(n):
        if res[i] is None:
            lo = max([k for k in known if k < i], default=None); hi = min([k for k in known if k > i], default=None)
            a = res[lo][1] if lo is not None else 0.0
            b = res[hi][0] if hi is not None else total_dur
            cnt = (hi if hi is not None else n) - (lo + 1 if lo is not None else 0)
            pos = i - (lo + 1 if lo is not None else 0)
            res[i] = (a + (b - a) * pos / cnt, a + (b - a) * (pos + 1) / cnt)
    out, prev = [], 0.0
    for (s, e) in res:
        s = max(s, prev); e = max(e, s + 0.05); out.append((s, e)); prev = s
    return out

def estimate_words_from_silence(audio_path, script_words, total_dur):
    """Fallback when Whisper is unavailable: snap the detected pauses onto punctuation boundaries (DP), then spread
    the words inside each stretch by letter count. Good to roughly +-0.15s on a clean TTS voice-over."""
    out = subprocess.run(["ffmpeg", "-hide_banner", "-i", audio_path, "-af", "silencedetect=n=-38dB:d=0.18", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    starts = [float(m) for m in re.findall(r"silence_start: ([0-9.]+)", out)]
    ends = [float(m) for m in re.findall(r"silence_end: ([0-9.]+)", out)]
    pauses = list(zip(starts, ends[:len(starts)]))
    if len(ends) < len(starts): pauses[-1] = (starts[-1], total_dur)
    lead = pauses[0][1] if pauses and pauses[0][0] < 0.05 else 0.0
    pauses = [p for p in pauses if p[0] > 0.05 and p[1] < total_dur - 0.05]
    weights = [len(_norm(w)) + 1.2 for w in script_words]
    cum = np.cumsum([0] + weights)
    voiced_total = total_dur - lead - sum(e - s for s, e in pauses)
    cand = [i for i, w in enumerate(script_words[:-1]) if w[-1] in ",.:;!?"]
    def est_t(i):  # estimated end time of word i assuming uniform letter speed + pauses seen so far
        t = lead + cum[i + 1] / cum[-1] * voiced_total
        return t + sum(e - s for s, e in pauses if s < t)
    P, C = len(pauses), len(cand)
    if P == 0 or C == 0:
        bounds = []
    else:
        INF = 1e9
        dp = np.full((P + 1, C + 1), INF); dp[0, :] = 0; back = {}
        for p in range(1, P + 1):
            for cidx in range(1, C + 1):
                cost = abs(est_t(cand[cidx - 1]) - pauses[p - 1][0]) - (1.2 if script_words[cand[cidx - 1]][-1] in '.:!?' else 0.0)
                best, arg = dp[p, cidx - 1], ("skip",)
                if dp[p - 1, cidx - 1] + cost < best: best, arg = dp[p - 1, cidx - 1] + cost, ("take",)
                dp[p, cidx] = best; back[(p, cidx)] = arg
        p, cidx, picked = P, C, {}
        while p > 0 and cidx > 0:
            if back[(p, cidx)][0] == "take": picked[p - 1] = cand[cidx - 1]; p -= 1; cidx -= 1
            else: cidx -= 1
        bounds = [picked[k] for k in sorted(picked)]
    # segments: (first_word, last_word, t_start, t_end)
    segs, w0, t0 = [], 0, lead
    for k, wi in enumerate(bounds):
        segs.append((w0, wi, t0, pauses[k][0])); w0, t0 = wi + 1, pauses[k][1]
    segs.append((w0, len(script_words) - 1, t0, total_dur - 0.05))
    res = [None] * len(script_words)
    for (wa, wb, ta, tb) in segs:
        ws = weights[wa:wb + 1]; tot = sum(ws); acc = 0.0
        for j, w in enumerate(ws):
            res[wa + j] = (ta + acc / tot * (tb - ta), ta + (acc + w) / tot * (tb - ta)); acc += w
    return res

# ------------------------------------------------------------------ captions
def caption_chunks(script_words, wt):
    chunks, i = [], 0
    while i < len(script_words):
        j = i + 1
        w = script_words[i]
        short = len(_norm(w)) <= 3
        punct_end = w[-1] in ",.:;!?"
        if j < len(script_words) and not punct_end and (short or (wt[j][1] - wt[i][0]) < 0.45) and len(w) + len(script_words[j]) < 17:
            j += 1
        chunks.append((" ".join(script_words[i:j]), wt[i][0], wt[j - 1][1], i))
        i = j
    out = []
    for k, (txt, s, e, wi) in enumerate(chunks):
        end = chunks[k + 1][1] if k + 1 < len(chunks) else e + 0.6
        out.append((txt, s, min(end, e + 0.65)))
    return out

_font_cache = {}
def _font(size, path=None):
    key = (size, path)
    if key not in _font_cache:
        cands = [path] if path else []
        cands += ["/content/fonts/Montserrat-ExtraBold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
        for c in cands:
            if c and os.path.exists(c):
                _font_cache[key] = ImageFont.truetype(c, size); break
        else: _font_cache[key] = ImageFont.load_default()
    return _font_cache[key]

_cap_cache = {}
def caption_img(text, size=78, font_path=None, ypos=0.70):
    key = (text, size, font_path)
    if key in _cap_cache: return _cap_cache[key]
    f = _font(size, font_path)
    tmp = Image.new("RGBA", (W, 300), (0, 0, 0, 0)); d = ImageDraw.Draw(tmp)
    sw = max(6, size // 9)
    bbox = d.textbbox((0, 0), text, font=f, stroke_width=sw)
    while bbox[2] - bbox[0] > W * 0.88 and size > 30:
        size -= 4; f = _font(size, font_path); sw = max(6, size // 9); bbox = d.textbbox((0, 0), text, font=f, stroke_width=sw)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    im = Image.new("RGBA", (tw + 2 * sw + 8, th + 2 * sw + 8), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.text((sw + 4 - bbox[0], sw + 4 - bbox[1]), text, font=f, fill=(255, 255, 255, 255), stroke_width=sw, stroke_fill=(10, 10, 14, 255))
    a = np.array(im); res = np.ascontiguousarray(a[..., [2, 1, 0, 3]])
    _cap_cache[key] = res
    return res

def draw_caption(frame, chunk, t, size=78, font_path=None, ypos=0.70):
    txt, s, e = chunk
    img = caption_img(txt, size, font_path)
    k = ease("back", (t - s) / 0.13)
    sc = 0.82 + 0.18 * k
    h, w = img.shape[:2]
    im2 = cv2.resize(img, (max(1, int(w * sc)), max(1, int(h * sc))), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR)
    h2, w2 = im2.shape[:2]
    x0, y0 = int(W / 2 - w2 / 2), int(H * ypos - h2 / 2)
    a = im2[..., 3:4].astype(np.float32) / 255.0
    reg = frame[y0:y0 + h2, x0:x0 + w2].astype(np.float32)
    frame[y0:y0 + h2, x0:x0 + w2] = np.clip(reg * (1 - a) + im2[..., :3] * a, 0, 255).astype(np.uint8)

# ------------------------------------------------------------------ SFX
def make_sfx(path_out, audio_path, whoosh_times, pop_times, sfx_db=-20, sr=44100):
    from scipy.signal import butter, lfilter
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", audio_path, "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
                         capture_output=True).stdout
    voice = np.frombuffer(raw, np.float32).copy()
    n = len(voice) + int(sr * 1.5)
    mix = np.zeros(n, np.float32); mix[:len(voice)] = voice
    rng = np.random.default_rng(1)
    g = 10 ** (sfx_db / 20)
    for t in whoosh_times:
        L = int(sr * 0.32); noise = rng.standard_normal(L)
        b, a = butter(2, [350 / (sr / 2), 4200 / (sr / 2)], btype="band"); w = lfilter(b, a, noise)
        env = np.sin(np.linspace(0, np.pi, L)) ** 2
        w = (w / (np.abs(w).max() + 1e-6) * env * g).astype(np.float32)
        s = max(0, int((t - 0.13) * sr)); mix[s:s + L] += w[: max(0, min(L, n - s))]
    for t in pop_times:
        L = int(sr * 0.14); tt = np.arange(L) / sr
        w = (np.sin(2 * np.pi * (240 * np.exp(-tt * 14)) * tt * 1.0) * np.exp(-tt * 30) * g * 1.6).astype(np.float32)
        s = int(t * sr); mix[s:s + L] += w[: max(0, min(L, n - s))]
    mix = np.clip(mix, -1, 1)
    import wave
    with wave.open(path_out, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes((mix * 32767).astype(np.int16).tobytes())

# ------------------------------------------------------------------ final export
def finalize(shot_files, chunks, audio_wav, out_path, cap_size=78, font_path=None, crf=18, tail_fade=0.0):
    list_txt = out_path + ".concat.txt"
    with open(list_txt, "w") as f:
        for s in shot_files: f.write(f"file '{os.path.abspath(s)}'\n")
    cat = out_path + ".cat.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", list_txt, "-c", "copy", cat], check=True)
    cap = cv2.VideoCapture(cat)
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                             "-r", str(FPS), "-i", "-", "-i", audio_wav, "-map", "0:v", "-map", "1:a", "-c:v", "libx264",
                             "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                             "-shortest", "-movflags", "+faststart", out_path], stdin=subprocess.PIPE)
    i = 0; ci = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        t = i / FPS
        while ci + 1 < len(chunks) and chunks[ci + 1][1] <= t: ci += 1
        if chunks and chunks[ci][1] <= t < chunks[ci][2]:
            draw_caption(frame, chunks[ci], t, cap_size, font_path)
        proc.stdin.write(np.ascontiguousarray(frame).tobytes()); i += 1
    proc.stdin.close(); proc.wait(); cap.release()
    for p in (list_txt, cat):
        if os.path.exists(p): os.remove(p)
    return i


def finalize_frames(frames_dir, total, chunks, audio_wav, out_path, ext="jpg", cap_size=78, font_path=None, crf=18):
    """Frame-by-frame assembly: read every frame image in order, draw captions, encode with the voice-over."""
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                             "-r", str(FPS), "-i", "-", "-i", audio_wav, "-map", "0:v", "-map", "1:a", "-c:v", "libx264",
                             "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                             "-shortest", "-movflags", "+faststart", out_path], stdin=subprocess.PIPE)
    ci = 0
    for i in range(total):
        frame = cv2.imread(f"{frames_dir}/{i:06d}.{ext}")
        if frame is None: raise FileNotFoundError(f"missing frame {i:06d}.{ext} in {frames_dir}")
        t = i / FPS
        while ci + 1 < len(chunks) and chunks[ci + 1][1] <= t: ci += 1
        if chunks and chunks[ci][1] <= t < chunks[ci][2]: draw_caption(frame, chunks[ci], t, cap_size, font_path)
        proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    proc.stdin.close(); proc.wait()
    return total
