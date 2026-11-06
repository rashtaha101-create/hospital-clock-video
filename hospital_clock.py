# shots.py - the shot list. Every shot is anchored to a WORD INDEX in the script, so when Whisper gives real word
# timings the cuts, punches, expression swaps and pulses all land on the right words automatically.
SCRIPT = ("Long ago, a hospital's big clock stopped, and nobody fixed it. Soon, the head doctor noticed something strange. "
          "Patients were being checked more often than ever. With no clock to watch, the nurses stopped waiting for the hour. "
          "They simply walked the ward, bed by bed. Most people would blame the broken clock and replace it. "
          "But the doctor looked closer. The clock hadn't caused the problem. It had exposed one: "
          "the nurses had been watching the clock, not the patients.")
# word indices: 0 Long | 11 Soon | 18 Patients | 26 With | 38 They | 46 Most | 56 But | 61 The | 67 It | 71 the(nurses) | 78 not

STYLE = ("stylized 3D render, toy-like miniature diorama, vintage hospital, glossy painted materials, soft global "
         "illumination, saturated colors, clean simple shapes, cinematic depth of field, vertical composition, "
         "no people, no characters, no text, no letters")
NEG = "people, person, character, face, text, letters, watermark, logo, blurry, photo, realistic skin"

# ---- backgrounds (7 unique images; shots reuse them with different crops/camera moves) ----------------------------
BACKGROUNDS = {
    "ward_clock_wide": "long old hospital ward seen from the entrance, two rows of iron beds with white sheets, a huge round "
                       "brass wall clock centered on the far wall, warm amber dusty light through tall windows, symmetrical",
    "clock_ecu": "extreme close-up of a giant antique brass wall clock face with roman numerals, frozen hands, dust in a "
                 "shaft of golden light, shallow depth of field, dark warm background",
    "ward_day": "bright cheerful hospital ward in daylight, rows of neat iron beds with teal blankets, big windows, "
                "teal and cream colors, wide empty floor in the foreground, eye level",
    "ward_low_sym": "low angle view down the center aisle of a hospital ward, perfectly symmetrical, rows of beds both sides, "
                    "the big brass clock high on the far wall center, soft light beams, floor in the foreground",
    "ward_topdown": "top-down overhead view of a hospital ward floor, parallel rows of iron beds with teal blankets, "
                    "checkered floor, soft light, diorama",
    "boardroom": "elegant hospital administration hallway with marble floor, red carpet, tall arched windows, wood paneling, "
                 "cool blue morning light with gold accents, empty",
    "spotlight_dark": "dark warm hospital corridor, a single spotlight cone from above on an empty floor, deep brown and amber, "
                      "moody, soft bokeh in the background",
}

def c(kind, x, y, h, **kw):
    d = dict(kind=kind, x=x, y=y, h=h); d.update(kw); return d

W_ = lambda i, off=0.0: ("w", i, off)   # time of word i (+offset)

SHOTS = [
 # 1 HOOK 0.0-1.0  medium-wide, low angle, FAST push-in, then push-through transition into the ECU
 dict(id=1, bg="ward_clock_wide", anchor=0, t=0.00, tin="cut",
      cam=dict(z=(1.0, 1.30), x=(0, 0), y=(0.01, -0.02), ease="out", par=0.9, dof=0.55, focus=(0.35, 0.35), hand=0.4),
      chars=[], fx=dict(dust=True, bloom=0.18, vignette=0.32, grade=(0.95, 1.0, 1.06))),
 # 2 ECU of the stopped clock - push + rack focus (dust sharp -> hands sharp), tiny jolt on "stopped"
 dict(id=2, bg="clock_ecu", anchor=2, t=1.00, tin="push",
      cam=dict(z=(1.0, 1.22), x=(0, 0.01), y=(0, 0), ease="io", par=0.8, dof=0.9, focus=(0.9, 0.35), focus_t=(0.0, ("w", 6)),
               punches=[(W_(6), 0.04)], hand=0.3),
      chars=[], fx=dict(dust=True, dust_amt=0.5, bloom=0.30, vignette=0.38, grade=(0.92, 1.0, 1.1))),
 # 3 staff ignoring the clock - slow lateral dolly, bed rail blurred in the foreground
 dict(id=3, bg="ward_clock_wide", anchor=7, t=3.10, tin="cut",
      cam=dict(z=(1.05, 1.10), x=(-0.05, 0.05), y=(0, 0), ease="lin", par=1.2, dof=0.6, focus=(0.4, 0.4), hand=0.3),
      chars=[c("nurse", 0.28, 0.86, 0.34, depth=0.45, path=((0, 0.20, 0.86), (1.5, 0.40, 0.86)), walk=True),
             c("nurse", 0.74, 0.84, 0.30, depth=0.42, flip=True, path=((0, 0.80, 0.84), (1.5, 0.66, 0.84)), walk=True,
               expr=[(0, "neutral")])],
      fx=dict(dust=True, bloom=0.15, vignette=0.32, grade=(0.95, 1.0, 1.05))),
 # 4 doctor medium, slow push-in ("Soon, the head doctor noticed")
 dict(id=4, bg="ward_day", anchor=11, t=4.60, tin="cut",
      cam=dict(z=(1.0, 1.10), x=(0.02, 0), y=(0, 0), ease="io", par=0.8, dof=0.5, focus=(0.55, 0.55), hand=0.3),
      chars=[c("doctor", 0.46, 0.93, 0.62, depth=0.6, expr=[(0, "neutral"), (W_(15), "think")], pose=[(0, "down")])],
      fx=dict(bloom=0.12, vignette=0.28, grade=(1.04, 1.02, 0.98))),
 # 5 ZOOM PUNCH on "strange": shock face + sweat drop + speed lines
 dict(id=5, bg="ward_day", anchor=16, t=6.35, tin="cut",
      cam=dict(z=(1.18, 1.26), x=(0.0, -0.01), y=(0.015, 0.015), ease="lin", par=0.9, dof=0.7, focus=(0.6, 0.6),
               punches=[(W_(17), 0.20)], hand=0.5),
      chars=[c("doctor", 0.5, 1.02, 0.78, depth=0.6, expr=[(0, "think"), (W_(17), "shock")], pose=[(0, "down"), (W_(17), "up")])],
      events=[dict(type="speed", t=W_(17), strength=0.6)],
      fx=dict(bloom=0.12, vignette=0.30, grade=(1.04, 1.02, 0.98))),
 # 6 patient being checked - medium, push-in on the nurse's check
 dict(id=6, bg="ward_day", anchor=18, t=7.88, tin="cut",
      cam=dict(z=(1.15, 1.30), x=(-0.04, -0.02), y=(0.02, 0.0), ease="io", par=1.0, dof=0.7, focus=(0.55, 0.55), hand=0.3),
      chars=[c("patient_bed", 0.62, 0.78, 0.30, depth=0.55, expr=[(0, "neutral"), (W_(21), "worried")]),
             c("nurse", 0.28, 0.86, 0.46, depth=0.62, expr=[(0, "neutral")], pose=[(0, "down"), (W_(21), "point")])],
      fx=dict(bloom=0.12, vignette=0.28, grade=(1.03, 1.02, 0.99))),
 # 7 REPEATED SHOT: whip to another nurse checking the same patient AGAIN ("more often than ever")
 dict(id=7, bg="ward_day", anchor=22, t=9.20, tin="whip_l",
      cam=dict(z=(1.25, 1.12), x=(0.04, 0.0), y=(0.0, 0.01), ease="io", par=1.0, dof=0.75, focus=(0.5, 0.5), hand=0.5),
      chars=[c("patient_bed", 0.40, 0.80, 0.30, depth=0.55, expr=[(0, "worried"), (W_(25), "weak")]),
             c("nurse", 0.74, 0.88, 0.50, depth=0.65, flip=True, expr=[(0, "neutral")], pose=[(0, "point")])],
      fx=dict(bloom=0.12, vignette=0.30, grade=(1.03, 1.02, 0.99))),
 # 8 nurses under the blank clock, SYMMETRICAL low angle ("With no clock to watch")
 dict(id=8, bg="ward_low_sym", anchor=26, t=10.71, tin="cut",
      cam=dict(z=(1.0, 1.12), x=(0, 0), y=(0.02, -0.01), ease="io", par=0.7, dof=0.5, focus=(0.55, 0.55), hand=0.2),
      chars=[c("nurse", 0.30, 0.90, 0.46, depth=0.55, expr=[(0, "look_up"), (W_(32), "neutral")]),
             c("nurse", 0.70, 0.90, 0.46, depth=0.55, flip=True, expr=[(0, "look_up"), (W_(32), "neutral")])],
      fx=dict(dust=True, bloom=0.2, vignette=0.34, grade=(0.97, 1.0, 1.04))),
 # 9 nurses turn and walk away from the clock ("stopped waiting for the hour")
 dict(id=9, bg="ward_low_sym", anchor=34, t=12.60, tin="cut",
      cam=dict(z=(1.12, 1.0), x=(0.02, -0.02), y=(0, 0), ease="io", par=1.1, dof=0.5, focus=(0.5, 0.5), hand=0.3),
      chars=[c("nurse", 0.30, 0.90, 0.46, depth=0.55, path=((0, 0.30, 0.90), (1.5, 0.12, 0.92)), walk=True),
             c("nurse", 0.70, 0.90, 0.46, depth=0.55, flip=True, path=((0, 0.70, 0.90), (1.5, 0.90, 0.92)), walk=True)],
      fx=dict(dust=True, bloom=0.2, vignette=0.34, grade=(0.97, 1.0, 1.04))),
 # 10 WHIP into a high-angle tracking shot over the ward ("They simply walked the ward,")
 dict(id=10, bg="ward_topdown", anchor=38, t=14.31, tin="whip_r",
      cam=dict(z=(1.0, 1.25), x=(-0.06, 0.06), y=(-0.05, 0.04), rot=(-3, 3), ease="lin", par=1.3, dof=0.35, focus=(0.5, 0.5)),
      chars=[c("nurse", 0.30, 0.55, 0.24, depth=0.5, path=((0, 0.30, 0.52), (1.5, 0.62, 0.62)), walk=True)],
      fx=dict(bloom=0.25, vignette=0.30, grade=(1.0, 1.0, 1.02))),
 # 11 macro "bed by bed" - three glow pulses, one per word, slow top-down push
 dict(id=11, bg="ward_topdown", anchor=43, t=15.87, tin="cut",
      cam=dict(z=(1.3, 1.55), x=(0.05, -0.02), y=(0.04, -0.04), rot=(2, -2), ease="io", par=1.0, dof=0.6, focus=(0.6, 0.6)),
      events=[dict(type="pulse", t=W_(43), pos=(0.32, 0.38)), dict(type="pulse", t=W_(44), pos=(0.66, 0.52)),
              dict(type="pulse", t=W_(45), pos=(0.38, 0.70))],
      chars=[], fx=dict(bloom=0.3, vignette=0.30, grade=(1.0, 1.0, 1.03))),
 # 12 RIDICULE: suits point and laugh at the clock - medium-wide, handheld bob
 dict(id=12, bg="boardroom", anchor=46, t=17.17, tin="cut",
      cam=dict(z=(1.05, 1.15), x=(0.02, -0.02), y=(0, 0), ease="io", par=0.9, dof=0.55, focus=(0.55, 0.55), hand=1.2),
      chars=[c("suit", 0.26, 0.92, 0.50, depth=0.6, expr=[(0, "smug"), (W_(49), "laugh")], pose=[(0, "cross"), (W_(49), "point")]),
             c("suit", 0.62, 0.90, 0.46, depth=0.55, flip=True, expr=[(0, "smug"), (W_(50), "laugh")], pose=[(0, "cross")], wiggle=True),
             c("suit", 0.88, 0.88, 0.40, depth=0.5, flip=True, expr=[(0, "neutral"), (W_(51), "laugh")], pose=[(0, "down")])],
      fx=dict(bloom=0.15, vignette=0.28, grade=(1.05, 1.0, 0.96))),
 # 13 "...and replace it": gleaming new clock rolled in, sparkles + flash
 dict(id=13, bg="boardroom", anchor=53, t=19.00, tin="cut",
      cam=dict(z=(1.1, 1.28), x=(-0.02, 0.01), y=(0.01, -0.01), ease="out", par=0.9, dof=0.6, focus=(0.55, 0.55), hand=0.6),
      chars=[c("prop", 0.5, 0.62, 0.40, name="clock", depth=0.55,
               path=((0, 0.5, 0.80), (0.8, 0.5, 0.62)), path_ease="back"),
             c("suit", 0.20, 0.92, 0.44, depth=0.6, expr=[(0, "smug")], pose=[(0, "cross")])],
      events=[dict(type="flash", t=W_(54))],
      fx=dict(sparkle=dict(pos=(0.5, 0.45), spread=300), bloom=0.4, vignette=0.28, grade=(1.05, 1.02, 0.97))),
 # 14 DISSOLVE to the doctor: push-in to the eyes ("But the doctor looked closer.") + rack focus
 dict(id=14, bg="spotlight_dark", anchor=56, t=20.28, tin="dissolve",
      cam=dict(z=(1.0, 1.35), x=(0, 0), y=(0.03, -0.02), ease="io", par=0.6, dof=0.8, focus=(0.3, 0.7)),
      chars=[c("doctor", 0.5, 1.30, 1.55, depth=0.7, bust=True, expr=[(0, "think"), (W_(59), "squint")])],
      fx=dict(bloom=0.25, vignette=0.45, grade=(0.95, 0.98, 1.05))),
 # 15 ECU of the old clock; slow push, then a long PULL-BACK with rack focus reveals the nurses beneath it
 dict(id=15, bg="clock_ecu", anchor=61, t=22.19, tin="cut",
      cam=dict(z=(1.35, 1.0), x=(0, 0), y=(-0.02, 0.03), ease="io", par=1.1, dof=0.85, focus=(0.9, 0.35), focus_t=(("w", 67), ("w", 71)),
               punches=[]),
      chars=[], fx=dict(dust=True, dust_amt=0.5, bloom=0.3, vignette=0.4, grade=(0.92, 1.0, 1.08))),
 # 16 wide symmetrical: nurses lined up staring at the clock, patients blurred behind
 dict(id=16, bg="ward_low_sym", anchor=71, t=25.86, tin="cut",
      cam=dict(z=(1.0, 1.14), x=(0, 0), y=(0.03, -0.01), ease="io", par=0.8, dof=0.7, focus=(0.6, 0.6), hand=0.15),
      chars=[c("nurse", 0.22, 0.92, 0.46, depth=0.6, expr=[(0, "look_up")]),
             c("nurse", 0.50, 0.94, 0.52, depth=0.62, expr=[(0, "look_up")]),
             c("nurse", 0.78, 0.92, 0.46, depth=0.6, flip=True, expr=[(0, "look_up")]),
             c("patient_bed", 0.50, 0.66, 0.18, depth=0.3, expr=[(0, "weak")])],
      fx=dict(dust=True, bloom=0.22, vignette=0.36, grade=(0.97, 1.0, 1.04))),
 # 17 FINAL HOLD: patient sharp in the foreground, nurses blurred behind; rack focus onto the patient on "patients"
 dict(id=17, bg="ward_low_sym", anchor=78, t=28.03, tin="cut",
      cam=dict(z=(1.35, 1.22), x=(0, 0), y=(0.04, 0.02), ease="out", par=0.9, dof=0.95, focus=(0.55, 0.15), focus_t=(("w", 79), ("w", 80)), hand=0.1),
      chars=[c("nurse", 0.34, 0.84, 0.34, depth=0.62, flip=False, expr=[(0, "look_up")]),
             c("nurse", 0.66, 0.84, 0.34, depth=0.62, flip=True, expr=[(0, "look_up")]),
             c("patient_bed", 0.5, 1.02, 0.60, depth=0.12, expr=[(0, "weak"), (W_(80), "weak")], pose=[(0, "down"), (W_(80), "wave")])],
      fx=dict(bloom=0.2, vignette=0.42, grade=(0.96, 1.0, 1.04))),
]
# transitions OUT of each shot are read from the NEXT shot's "tin"
