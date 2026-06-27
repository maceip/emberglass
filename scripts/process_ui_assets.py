#!/usr/bin/env python3
"""Process retro UI reference sheets in docs/ui/ into clean product derivatives
under docs/ui/processed/. Raw files are NEVER modified. Reproducible: re-run to
regenerate. See docs/ui/processed/manifest.json for the marker->source mapping.

Stages:
  icons  - slice the *I1 Shining Force sheet into a curated, named icon set
           (native + @2x pixel + monochrome silhouette + locked variant).
  frames - crop reusable panel/frame derivatives from *H1/*S1/*J1/*D1/*P1/*C1.
  all    - run everything and (re)write manifest.json.

Usage: python3 scripts/process_ui_assets.py [icons|frames|all]
"""
import json
import os
import sys
from PIL import Image, ImageDraw

UI = "docs/ui"
OUT = "docs/ui/processed"
ICONS = os.path.join(OUT, "icons")
HOME = os.path.join(OUT, "home")
SKILL = os.path.join(OUT, "skill")
BOARD = os.path.join(OUT, "job-board")
SCRATCH = "/tmp/eg_ui_scratch"

SRC = {
    "I1": "Sega CD - Shining Force CD - Miscellaneous - Weapon & Spell Icons.gif",
    "H1": "Game Boy Advance - Pokemon FireRed _ LeafGreen - Menu Elements - PC Interface.png",
    "S1": "PC _ Computer - Heroes of Might and Magic 3 - Miscellaneous - Spellbook.png",
    "J1": "Wii - Fortune Street - Miscellaneous - Menu Boxes.png",
    "D1": "PC _ Computer - Agatha Christie_ Murder on the Orient Express - Inventory - Interface.png",
    "P1": "Game Boy Advance - Flashback Legend (Prototype) - Miscellaneous - Screens (Faces).png",
    "C1": "SNES - Dragon Ball Z_ Legend of the Super Saiyan (JPN) - Miscellaneous - Interface Icons and Text Box.png",
}

# Curated icon mapping: semantic name -> sliced tile index (see scripts montage
# stage; indices verified visually against the *I1 Shining Force sheet).
ICON_MAP = {
    "skill-calendar": 6, "skill-email": 11, "skill-notes": 0, "skill-crm": 3,
    "skill-calls": 20, "skill-code": 9, "skill-travel": 14, "skill-shopping": 1,
    "skill-media": 10,
    "act-event": 6, "act-slot": 5, "act-reminder": 12, "act-rsvp": 20,
    "act-reply": 11, "act-label": 36, "act-archive": 19, "act-meeting": 20,
    "reward-chest": 2, "forbidden": 22,
}

MANIFEST = {"_note": "Processed derivatives of retro UI references. Raw files in docs/ui/ are provenance and untouched.",
            "icons": {}, "frames": {}}


def ensure_dirs():
    for d in (OUT, ICONS, HOME, SKILL, BOARD, SCRATCH):
        os.makedirs(d, exist_ok=True)


def close(a, b, tol=24):
    return all(abs(int(a[i]) - int(b[i])) <= tol for i in range(3))


def runs(mask, gap=1, minlen=6):
    out, n, i = [], len(mask), 0
    while i < n:
        if mask[i]:
            j, holes = i, 0
            while j + 1 < n and (mask[j + 1] or holes < gap):
                holes = 0 if mask[j + 1] else holes + 1
                j += 1
            while j > i and not mask[j]:
                j -= 1
            if j - i + 1 >= minlen:
                out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def key_green(src_path):
    """Load icon sheet, key the green background to transparent, mask legend."""
    im = Image.open(src_path).convert("RGBA")
    rgb = im.convert("RGB")
    W, H = im.size
    bg = rgb.getpixel((0, 0))
    alpha = Image.new("L", (W, H), 0)
    al = alpha.load()
    rp = rgb.load()
    for y in range(H):
        for x in range(W):
            al[x, y] = 0 if close(rp[x, y], bg) else 255
    for y in range(0, 96):
        for x in range(680, W):
            al[x, y] = 0
    for y in range(96, 130):
        for x in range(700, W):
            al[x, y] = 0
    im.putalpha(alpha)
    return im, alpha


def slice_icons(im, alpha):
    W, H = im.size
    ap = alpha.load()
    rowmask = [sum(1 for x in range(0, 680) if ap[x, y] > 0) >= 4 for y in range(H)]
    bands = runs(rowmask, gap=4, minlen=8)
    catalog, idx = [], 0
    for (y0, y1) in bands:
        colmask = [sum(1 for y in range(y0, y1 + 1) if ap[x, y] > 0) >= 3 for x in range(680)]
        for (xs, xe) in runs(colmask, gap=2, minlen=6):
            ys, ye = y1, y0
            for y in range(y0, y1 + 1):
                if any(ap[x, y] > 0 for x in range(xs, xe + 1)):
                    ys, ye = min(ys, y), max(ye, y)
            if ye >= ys and (xe - xs) >= 5 and (ye - ys) >= 5:
                catalog.append((xs, ys, xe + 1, ye + 1))
                idx += 1
    return catalog


def monochrome(icon, rgb=(247, 243, 233)):
    """White-ish silhouette preserving alpha, for CSS tinting / 8-bit look."""
    out = Image.new("RGBA", icon.size, (0, 0, 0, 0))
    sp, op = icon.load(), out.load()
    for y in range(icon.height):
        for x in range(icon.width):
            a = sp[x, y][3]
            if a > 40:
                op[x, y] = (rgb[0], rgb[1], rgb[2], 255)
    return out


def add_lock(icon):
    """Desaturate + darken and stamp a small padlock, for 'locked' skill state."""
    base = Image.new("RGBA", icon.size, (0, 0, 0, 0))
    sp, bp = icon.load(), base.load()
    for y in range(icon.height):
        for x in range(icon.width):
            r, g, b, a = sp[x, y]
            if a > 0:
                l = int(0.3 * r + 0.59 * g + 0.11 * b)
                l = int(l * 0.55 + 20)
                bp[x, y] = (l, l, max(0, l - 6), a)
    d = ImageDraw.Draw(base)
    w, h = icon.size
    pw = max(6, w // 2)
    px, py = w - pw - 1, h - pw - 1
    d.rectangle([px, py + pw // 3, px + pw, py + pw], fill=(20, 20, 24, 255), outline=(247, 220, 120, 255))
    d.arc([px + pw // 5, py - 1, px + pw - pw // 5, py + pw // 2 + 2], 180, 360, fill=(247, 220, 120, 255), width=1)
    return base


def ability_tiles(cat):
    """Wide, square pictogram tiles in the brown ability bands (excludes the
    tall weapon tiles and the narrow FB/DB letter tiles). Sorted top-left."""
    a = [c for c in cat if (c[2] - c[0]) >= 22 and c[1] >= 100]
    a.sort(key=lambda c: (c[1], c[0]))
    return a


def do_montage():
    """Indexed contact sheet of the curated ability-tile list (the SAME list the
    curator indexes into), rendered large for reliable selection."""
    src = os.path.join(UI, SRC["I1"])
    im, alpha = key_green(src)
    ab = ability_tiles(slice_icons(im, alpha))
    cell, cols = 92, 12
    rows = (len(ab) + cols - 1) // cols
    m = Image.new("RGBA", (cols * cell, rows * cell), (24, 30, 32, 255))
    d = ImageDraw.Draw(m)
    for i, (x0, y0, x1, y1) in enumerate(ab):
        cx, cy = (i % cols) * cell, (i // cols) * cell
        ic = im.crop((x0, y0, x1, y1))
        s = min((cell - 26) / ic.width, (cell - 26) / ic.height)
        ic = ic.resize((max(1, int(ic.width * s)), max(1, int(ic.height * s))), Image.NEAREST)
        m.alpha_composite(ic, (cx + (cell - ic.width) // 2, cy + 3))
        d.text((cx + 3, cy + cell - 12), str(i), fill=(190, 225, 215, 255))
    p = os.path.join(SCRATCH, "abilities.png")
    m.save(p)
    print(f"[montage] {len(ab)} ability tiles -> {p}")


def _unused_full_montage(im, cat):
    cell, cols = 64, 16
    rows = (len(cat) + cols - 1) // cols
    m = Image.new("RGBA", (cols * cell, rows * cell), (20, 26, 28, 255))
    d = ImageDraw.Draw(m)
    for i, (x0, y0, x1, y1) in enumerate(cat):
        cx, cy = (i % cols) * cell, (i // cols) * cell
        ic = im.crop((x0, y0, x1, y1))
        s = min((cell - 16) / ic.width, (cell - 16) / ic.height)
        ic = ic.resize((max(1, int(ic.width * s)), max(1, int(ic.height * s))), Image.NEAREST)
        m.alpha_composite(ic, (cx + (cell - ic.width) // 2, cy + 2))
        d.text((cx + 2, cy + cell - 12), str(i), fill=(180, 220, 210, 255))
    p = os.path.join(SCRATCH, "montage.png")
    m.save(p)
    print(f"[montage] {len(cat)} tiles -> {p}")


def do_icons():
    src = os.path.join(UI, SRC["I1"])
    im, alpha = key_green(src)
    ab = ability_tiles(slice_icons(im, alpha))
    print(f"[icons] {len(ab)} ability tiles from {SRC['I1']}")
    for name, idx in ICON_MAP.items():
        if idx >= len(ab):
            print(f"  !! idx {idx} out of range for {name}")
            continue
        x0, y0, x1, y1 = ab[idx]
        ic = im.crop((x0, y0, x1, y1))
        ic.save(os.path.join(ICONS, f"{name}.png"))
        ic.resize((ic.width * 2, ic.height * 2), Image.NEAREST).save(os.path.join(ICONS, f"{name}@2x.png"))
        monochrome(ic).save(os.path.join(ICONS, f"{name}-mono.png"))
        MANIFEST["icons"][name] = {
            "marker": "*I1", "source": SRC["I1"], "src_bbox": [x0, y0, x1, y1],
            "files": [f"icons/{name}.png", f"icons/{name}@2x.png", f"icons/{name}-mono.png"],
        }
    # locked variants for the skill icons only
    for name in [n for n in ICON_MAP if n.startswith("skill-")]:
        p = os.path.join(ICONS, f"{name}.png")
        if os.path.exists(p):
            lock = add_lock(Image.open(p).convert("RGBA"))
            lock.save(os.path.join(ICONS, f"{name}-locked.png"))
            MANIFEST["icons"][name]["files"].append(f"icons/{name}-locked.png")
    print(f"[icons] wrote {len(ICON_MAP)} curated icons (+variants) to {ICONS}")


# ---- frames: reusable panel/meter/callout derivatives (visually measured) ----
# name: (markerkey, (x0,y0,x1,y1), outdir, scale, key_bg_corner)
FRAME_CROPS = {
    # *J1 Fortune Street glossy status bands -> Job Board card bands
    "band-locked":   ("J1", (6, 352, 527, 387), BOARD, 1, True),   # purple
    "band-reliable": ("J1", (6, 392, 527, 428), BOARD, 1, True),   # teal
    "band-learning": ("J1", (6, 431, 527, 465), BOARD, 1, True),   # gold
    "band-review":   ("J1", (6, 469, 527, 503), BOARD, 1, True),   # pink
    "panel-light":   ("J1", (6, 221, 579, 331), BOARD, 1, True),   # detail/plan panel
    # *P1 Flashback -> training console meter (used as the forge track)
    "meter-strip":   ("P1", (0, 1, 362, 47), SKILL, 1, True),
    # *C1 DBZ -> ornate callout frame (border-image at slice=6; real border ~6px,
    # so the interior alphabet is discarded — do NOT slice deeper or text ghosts)
    "callout-frame": ("C1", (2, 2, 208, 60), SKILL, 1, True),
    # NOTE: *H1 (Pokemon PC) intentionally NOT skinned. Its boxes carry a header
    # bar + wallpaper interior, so they don't 9-slice into a uniform border ring;
    # the heroic equipped slot is realized in CSS instead. *H1 stays provenance.
}


def do_frames():
    for name, (mk, box, outdir, scale, keybg) in FRAME_CROPS.items():
        src = os.path.join(UI, SRC[mk])
        im = Image.open(src).convert("RGBA")
        crop = im.crop(box)
        if keybg:
            bg = im.getpixel((1, 1))
            cp = crop.load()
            for y in range(crop.height):
                for x in range(crop.width):
                    px = cp[x, y]
                    if all(abs(int(px[i]) - int(bg[i])) <= 18 for i in range(3)):
                        cp[x, y] = (px[0], px[1], px[2], 0)
        if scale != 1:
            crop = crop.resize((crop.width * scale, crop.height * scale), Image.NEAREST)
        os.makedirs(outdir, exist_ok=True)
        out = os.path.join(outdir, f"{name}.png")
        crop.save(out)
        rel = os.path.relpath(out, OUT)
        MANIFEST["frames"][name] = {"marker": f"*{mk}", "source": SRC[mk],
                                    "src_bbox": list(box), "file": rel}
        print(f"[frames] {name} <- *{mk} {box} -> {rel}")
    # Provenance-only markers (intentionally not skinned with bitmaps):
    MANIFEST["frames"]["_D1_provenance"] = {"marker": "*D1", "source": SRC["D1"],
        "note": "Selected-detail framing reference; not skinned (noir palette excluded per review)."}
    MANIFEST["frames"]["_H1_provenance"] = {"marker": "*H1", "source": SRC["H1"],
        "note": "Home slot reference; not 9-sliced (header bar + wallpaper give a non-uniform ring). Heroic equipped slot realized in CSS."}


def write_manifest():
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(MANIFEST, f, indent=2)
    print(f"[manifest] {OUT}/manifest.json")


def main():
    ensure_dirs()
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage == "montage":
        do_montage()
        return
    if stage in ("icons", "all"):
        do_icons()
    if stage in ("frames", "all"):
        do_frames()
    # merge existing manifest sections we didn't regenerate this run
    mp = os.path.join(OUT, "manifest.json")
    if os.path.exists(mp):
        try:
            old = json.load(open(mp))
            for sec in ("icons", "frames"):
                if not MANIFEST[sec] and old.get(sec):
                    MANIFEST[sec] = old[sec]
        except Exception:
            pass
    write_manifest()


if __name__ == "__main__":
    main()
