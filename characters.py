# characters.py - procedural flat 2D characters (round white head, stick limbs, costume) with swappable expressions.
# Everything is drawn in code so characters stay 100% consistent from shot to shot (limited-animation pose swaps).
import math, functools
import numpy as np
from PIL import Image, ImageDraw

DESIGN_W, DESIGN_H, BUST_H = 500, 900, 330
HEAD = (250, 172, 105)  # cx, cy, radius (design units)
SKIN = (255, 252, 246)
INK = (18, 18, 22)

COSTUMES = {
    "doctor":  dict(body=(244, 246, 248), w=170),
    "nurse":   dict(body=(52, 170, 172), w=150),
    "suit":    dict(body=(58, 64, 86), w=180),
    "patient": dict(body=(168, 204, 236), w=160),
}

# eye: round|wide|lid|arc|squint|up ; brow=(inner_dy, outer_dy, raise) ; mouth name
EXPRS = {
    "neutral":  dict(eye="round", brow=(0, 0, 0),    mouth="flat"),
    "deadpan":  dict(eye="lid",   brow=(0, 0, -2),   mouth="flat", lid=0.40, tilt=0),
    "annoyed":  dict(eye="lid",   brow=(9, -4, -4),  mouth="flat", lid=0.45, tilt=10),
    "smug":     dict(eye="lid",   brow=(0, 0, 4),    mouth="smirk", lid=0.40, tilt=0, brow1_raise=10),
    "shock":    dict(eye="wide",  brow=(-6, -6, 14), mouth="o", sweat=True),
    "laugh":    dict(eye="arc",   brow=(-4, -2, 6),  mouth="laugh"),
    "worried":  dict(eye="round", brow=(-10, 6, 4),  mouth="wavy", small_pupil=True),
    "weak":     dict(eye="lid",   brow=(-8, 4, 0),   mouth="frown", lid=0.55, tilt=-8),
    "look_up":  dict(eye="round", brow=(0, 0, 2),    mouth="flat", look=(0, -15)),
    "squint":   dict(eye="squint", brow=(7, -3, -2), mouth="flat", glint=True),
    "grin":     dict(eye="round", brow=(0, 0, 6),    mouth="grin"),
    "think":    dict(eye="round", brow=(4, -4, 2),   mouth="flat", look=(14, -8)),
}

POSES = {  # ((a1,a2) left arm, (a1,a2) right arm) - angle from straight down, outward positive
    "down":   ((10, 6), (10, 6)),
    "point":  ((10, 6), (58, -30)),
    "shrug":  ((40, -105), (40, -105)),
    "up":     ((150, -15), (150, -15)),
    "wave":   ((10, 6), (160, 0)),
    "cross":  ((55, -100), (55, -100)),
    "hold":   ((38, -70), (38, -70)),
    "think":  ((10, 6), (45, -150)),
}


def _pt(o, ang_deg, length, side):
    a = math.radians(ang_deg)
    dx = (-math.sin(a) if side == "L" else math.sin(a)) * length
    return (o[0] + dx, o[1] + math.cos(a) * length)


@functools.lru_cache(maxsize=96)
def draw_character(kind="nurse", expr="neutral", pose="down", legs=0, S=2, bust=False, wig=0):
    """Return a BGRA uint8 array of the character. Design space is 500x900 units, scaled by S."""
    if kind == "patient_bed":
        return _draw_bed(expr, S, pose, wig)
    cos = COSTUMES.get(kind, COSTUMES["patient"])
    cw, ch = int(DESIGN_W * S), int((BUST_H if bust else DESIGN_H) * S)
    big = 2  # supersample
    im = Image.new("RGBA", (cw * big, ch * big), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    k = S * big
    P = lambda x, y: (x * k, y * k)
    LW = max(2, int(7 * k))

    def line(pts, w=7, fill=INK):
        pp = [P(*p) for p in pts]
        d.line(pp, fill=fill, width=int(w * k), joint="curve")
        for p in (pp[0], pp[-1]):
            r = w * k / 2
            d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=fill)

    def ell(cx, cy, rx, ry, fill=None, outline=None, w=5):
        d.ellipse([(cx - rx) * k, (cy - ry) * k, (cx + rx) * k, (cy + ry) * k], fill=fill,
                  outline=outline, width=int(w * k))

    def poly(pts, fill=None, outline=INK, w=7):
        pp = [P(*p) for p in pts]
        if fill:
            d.polygon(pp, fill=fill)
        if outline:
            d.line(pp + [pp[0]], fill=outline, width=int(w * k), joint="curve")

    cx, cy, R = HEAD
    bw = cos["w"]
    # ---------- body ----------
    if kind == "nurse":
        poly([(250 - 66, 272), (250 + 66, 272), (250 + 104, 600), (250 - 104, 600)], fill=cos["body"])
        poly([(250 - 40, 330), (250 + 40, 330), (250 + 62, 600), (250 - 62, 600)], fill=(248, 248, 248), w=5)
        body_bot, leg_x = 600, 34
    else:
        d.rounded_rectangle([(250 - bw / 2) * k, 272 * k, (250 + bw / 2) * k, 575 * k], radius=int(46 * k),
                            fill=cos["body"], outline=INK, width=LW)
        body_bot, leg_x = 575, 36
    if kind == "doctor":
        poly([(250 - 26, 272), (250, 372), (250 + 26, 272)], fill=(200, 215, 235), w=5)
        line([(250 - 36, 276), (250 - 6, 360)], 6); line([(250 + 36, 276), (250 + 6, 360)], 6)
        d.arc([(250 - 52) * k, 270 * k, (250 + 52) * k, 410 * k], 20, 160, fill=INK, width=int(5 * k))
        ell(250, 412, 10, 10, fill=(190, 195, 205), outline=INK, w=4)
    if kind == "suit":
        poly([(250 - 32, 272), (250 + 32, 272), (250, 360)], fill=(250, 250, 250), w=5)
        poly([(250 - 11, 296), (250 + 11, 296), (250 + 16, 420), (250, 440), (250 - 16, 420)],
             fill=(196, 42, 44), w=5)
    if kind == "patient":
        line([(250 - 30, 280), (250 + 30, 330)], 5); line([(250 + 30, 280), (250 - 30, 330)], 5)
    if not bust:
        # ---------- legs ----------
        lo = [(0, 0), (-38, 24), (0, 0), (26, -38)][legs % 4]
        ro = [(0, 0), (26, -38), (0, 0), (-38, 24)][legs % 4]
        for sgn, (dx, dy) in ((-1, lo), (1, ro)):
            x0 = 250 + sgn * leg_x
            foot = (x0 + dx, 838 - max(0, -dy) * 0.0 + (0 if dy >= 0 else dy * 0.35))
            line([(x0, body_bot - 6), foot], 9)
            ell(foot[0] + sgn * 8, foot[1] + 4, 26, 12, fill=INK)
        # ---------- arms ----------
        (la1, la2), (ra1, ra2) = POSES.get(pose, POSES["down"])
        ra2 += wig; la2 += wig * (1 if pose in ("up", "cross") else 0)
        for side, (a1, a2) in (("L", (la1, la2)), ("R", (ra1, ra2))):
            sh = (250 - (bw / 2 - 6), 298) if side == "L" else (250 + (bw / 2 - 6), 298)
            el = _pt(sh, a1, 118, side)
            hd = _pt(el, a1 + a2, 108, side)
            line([sh, el, hd], 9)
            ell(hd[0], hd[1], 12, 12, fill=INK)
    # ---------- head ----------
    ell(cx, cy, R, R, fill=SKIN, outline=INK, w=7)
    if kind == "nurse":
        poly([(cx - 66, cy - R + 24), (cx + 66, cy - R + 24), (cx + 56, cy - R - 36), (cx - 56, cy - R - 36)],
             fill=(250, 250, 250), w=6)
        d.rectangle([(cx - 6) * k, (cy - R - 28) * k, (cx + 6) * k, (cy - R + 14) * k], fill=(200, 38, 44))
        d.rectangle([(cx - 24) * k, (cy - R - 10) * k, (cx + 24) * k, (cy - R + 4) * k], fill=(200, 38, 44))
    if kind == "suit":
        ell(cx, cy - R + 16, 92, 15, fill=(30, 30, 36), outline=INK, w=5)
        d.rounded_rectangle([(cx - 56) * k, (cy - R - 62) * k, (cx + 56) * k, (cy - R + 16) * k],
                            radius=int(10 * k), fill=(30, 30, 36), outline=INK, width=int(5 * k))
        d.rectangle([(cx - 56) * k, (cy - R - 14) * k, (cx + 56) * k, (cy - R + 4) * k], fill=(196, 42, 44))
    if kind == "doctor":
        for sgn in (-1, 1):
            ell(cx + sgn * (R - 6), cy - 22, 17, 24, fill=(200, 200, 206), outline=INK, w=5)
        d.arc([(cx - R + 8) * k, (cy - R + 8) * k, (cx + R - 8) * k, (cy + 40) * k], 205, 335, fill=(90, 90, 100),
              width=int(7 * k))
        ell(cx, cy - R + 24, 24, 24, fill=(214, 218, 226), outline=INK, w=5)
        ell(cx, cy - R + 24, 7, 7, fill=INK)
    if kind == "patient":
        for dx in (-14, 0, 14):
            line([(cx + dx, cy - R + 4), (cx + dx * 1.6, cy - R - 20)], 5)
    _face(d, k, cx, cy, EXPRS.get(expr, EXPRS["neutral"]), kind, line, ell, poly)
    return _finish(im, big)


def _finish(im, big):
    if big > 1:
        im = im.resize((im.width // big, im.height // big), Image.LANCZOS)
    a = np.array(im)  # RGBA
    return np.ascontiguousarray(a[..., [2, 1, 0, 3]])  # -> BGRA


def _face(d, k, cx, cy, e, kind, line, ell, poly):
    ex_off, ey = 42, cy - 4
    rx, ry = 27, 31
    wide = e["eye"] == "wide"
    if wide:
        rx, ry = 31, 38
    look = e.get("look", (0, 0))
    glasses = kind == "doctor"
    for sgn in (-1, 1):
        ex = cx + sgn * ex_off
        if e["eye"] in ("round", "wide", "lid", "up", "squint"):
            ryy = 9 if e["eye"] == "squint" else ry
            ell(ex, ey, rx, ryy, fill=(255, 255, 255), outline=INK, w=5)
            pr = 6 if (wide or e.get("small_pupil")) else 10
            ell(ex + look[0], ey + look[1] * (ryy / ry), pr, pr, fill=INK)
            if e["eye"] == "lid":
                cover = e.get("lid", 0.4)
                ytop = ey - ry - 4
                yline = ey - ry + 2 * ry * cover
                t = e.get("tilt", 0) * sgn * -0.5  # inner side lower for annoyed
                poly([(ex - rx - 6, ytop), (ex + rx + 6, ytop), (ex + rx + 6, yline + (t if sgn == 1 else -t)),
                      (ex - rx - 6, yline + (-t if sgn == 1 else t))], fill=SKIN, outline=None)
                line([(ex - rx - 2, yline + (-t if sgn == 1 else t)), (ex + rx + 2, yline + (t if sgn == 1 else -t))], 7)
        elif e["eye"] == "arc":
            d.arc([(ex - rx) * k, (ey - ry * 0.3) * k, (ex + rx) * k, (ey + ry * 0.9) * k], 200, 340, fill=INK,
                  width=int(8 * k))
        # brows
        inner, outer, rs = e["brow"]
        if sgn == -1 and e.get("brow1_raise"):
            rs2 = rs + e["brow1_raise"]
        else:
            rs2 = rs
        yb = ey - ry - 20 - rs2
        xin, xout = ex - sgn * 20, ex + sgn * 30
        line([(xin, yb + inner), (xout, yb + outer)], 11)
        if glasses:
            ell(ex, ey, rx + 14, ry + 12, outline=INK, w=5)
            if e.get("glint"):
                line([(ex - 14, ey - 18), (ex + 2, ey - 34)], 6, fill=(255, 255, 255))
    if glasses:
        line([(cx - ex_off + rx + 14, ey - 4), (cx + ex_off - rx - 14, ey - 4)], 5)
    my = cy + 62
    m = e["mouth"]
    if m == "flat":
        line([(cx - 28, my), (cx + 28, my)], 7)
    elif m == "frown":
        d.arc([(cx - 30) * k, (my - 2) * k, (cx + 30) * k, (my + 46) * k], 205, 335, fill=INK, width=int(7 * k))
    elif m == "o":
        ell(cx, my + 4, 15, 19, fill=INK)
    elif m == "smirk":
        line([(cx - 26, my + 6), (cx, my + 10), (cx + 30, my - 10)], 7)
    elif m == "wavy":
        line([(cx - 30, my + 2), (cx - 15, my - 6), (cx, my + 4), (cx + 15, my - 6), (cx + 30, my + 2)], 6)
    elif m == "laugh":
        d.pieslice([(cx - 50) * k, (my - 34) * k, (cx + 50) * k, (my + 60) * k], 0, 180, fill=INK)
        d.pieslice([(cx - 30) * k, (my + 8) * k, (cx + 30) * k, (my + 62) * k], 0, 180, fill=(235, 98, 112))
    elif m == "grin":
        d.pieslice([(cx - 46) * k, (my - 30) * k, (cx + 46) * k, (my + 44) * k], 0, 180, fill=(255, 255, 255),
                   outline=INK, width=int(6 * k))
        line([(cx - 46, my + 2), (cx + 46, my + 2)], 5)
        for dx in (-30, -15, 0, 15, 30):
            line([(cx + dx, my + 2), (cx + dx, my + 22)], 4)
    if e.get("sweat"):
        sx, sy = cx + 101, cy - 70
        poly([(sx, sy - 40), (sx - 16, sy), (sx + 16, sy)], fill=(120, 200, 255), w=4)
        ell(sx, sy + 4, 17, 17, fill=(120, 200, 255), outline=INK, w=4)


def _draw_bed(expr, S, pose, wig):
    """Patient lying in bed: head on pillow, blanket, optional waving arm."""
    cw, ch = int(DESIGN_W * S), int(560 * S)
    big = 2
    im = Image.new("RGBA", (cw * big, ch * big), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    k = S * big

    def line(pts, w=7, fill=INK):
        pp = [(x * k, y * k) for x, y in pts]
        d.line(pp, fill=fill, width=int(w * k), joint="curve")
        for p in (pp[0], pp[-1]):
            r = w * k / 2
            d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=fill)

    def ell(cx, cy, rx, ry, fill=None, outline=None, w=5):
        d.ellipse([(cx - rx) * k, (cy - ry) * k, (cx + rx) * k, (cy + ry) * k], fill=fill, outline=outline,
                  width=int(w * k))

    def poly(pts, fill=None, outline=INK, w=7):
        pp = [(x * k, y * k) for x, y in pts]
        if fill:
            d.polygon(pp, fill=fill)
        if outline:
            d.line(pp + [pp[0]], fill=outline, width=int(w * k), joint="curve")

    d.rounded_rectangle([40 * k, 150 * k, 460 * k, 330 * k], radius=int(60 * k), fill=(250, 250, 250), outline=INK,
                        width=int(7 * k))
    d.rounded_rectangle([20 * k, 300 * k, 480 * k, 540 * k], radius=int(46 * k), fill=(92, 164, 160), outline=INK,
                        width=int(7 * k))
    line([(20, 345), (480, 345)], 6)
    if pose == "wave":
        sh = (400, 340)
        el = (440, 250 + wig * 0.6)
        hd = (430 + wig, 170)
        line([sh, el, hd], 9)
        ell(hd[0], hd[1], 13, 13, fill=INK)
    cx, cy, R = 250, 150, 105
    ell(cx, cy, R, R, fill=SKIN, outline=INK, w=7)
    for dx in (-14, 0, 14):
        line([(cx + dx, cy - R + 4), (cx + dx * 1.6, cy - R - 20)], 5)
    _face(d, k, cx, cy, EXPRS.get(expr, EXPRS["weak"]), "patient", line, ell, poly)
    return _finish(im, big)


def draw_prop(name, S=2):
    """Props: 'clock' = brass wall clock (hands 10:10)."""
    size = int(360 * S)
    big = 3
    im = Image.new("RGBA", (size * big, size * big), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    k = size * big / 360.0
    c = 180
    d.ellipse([(c - 170) * k, (c - 170) * k, (c + 170) * k, (c + 170) * k], fill=(214, 164, 52), outline=INK,
              width=int(9 * k))
    d.ellipse([(c - 138) * k, (c - 138) * k, (c + 138) * k, (c + 138) * k], fill=(252, 248, 236), outline=INK,
              width=int(6 * k))
    for i in range(12):
        a = math.radians(i * 30)
        p0 = (c + math.sin(a) * 112, c - math.cos(a) * 112)
        p1 = (c + math.sin(a) * 130, c - math.cos(a) * 130)
        d.line([p0[0] * k, p0[1] * k, p1[0] * k, p1[1] * k], fill=INK, width=int(7 * k))
    for ang, ln, w in ((-60, 74, 12), (60, 108, 8)):  # 10:10
        a = math.radians(ang)
        d.line([c * k, c * k, (c + math.sin(a) * ln) * k, (c - math.cos(a) * ln) * k], fill=INK, width=int(w * k))
    d.ellipse([(c - 10) * k, (c - 10) * k, (c + 10) * k, (c + 10) * k], fill=INK)
    im = im.resize((size, size), Image.LANCZOS)
    a = np.array(im)
    return np.ascontiguousarray(a[..., [2, 1, 0, 3]])


if __name__ == "__main__":
    import cv2
    tiles = []
    for kind, ex, ps in [("doctor", "neutral", "point"), ("doctor", "shock", "down"), ("doctor", "squint", "think"),
                         ("nurse", "neutral", "down"), ("nurse", "laugh", "shrug"), ("suit", "smug", "cross"),
                         ("suit", "laugh", "up"), ("patient", "weak", "down")]:
        s = draw_character(kind, ex, ps, 0, 1)
        bg = np.full((s.shape[0], s.shape[1], 3), (232, 214, 190), np.uint8)
        a = s[..., 3:4] / 255.0
        tiles.append((bg * (1 - a) + s[..., :3] * a).astype(np.uint8))
    sheet = np.hstack(tiles[:4]); sheet2 = np.hstack(tiles[4:])
    cv2.imwrite("/home/claude/proj/chars_test.png", np.vstack([sheet, sheet2]))
