"""Cluster-only probe: fetch ambientCG facades and tile their colour maps into a contact sheet. Not a scene."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mitsuba as mi
mi.set_variant("scalar_rgb")
from PIL import Image, ImageDraw
import web_assets as W

ids = sys.argv[1].split(",")
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
tiles = []
for aid in ids:
    s = W.surface(aid)
    bc = (s["bsdf"] if s["type"] == "normalmap" else s)["base_color"]["filename"]
    im = Image.open(bc).convert("RGB").resize((256, 256))
    ImageDraw.Draw(im).text((4, 4), f"{aid} {Image.open(bc).size}", fill=(255, 0, 0))
    tiles.append(im)
cols = 6
sheet = Image.new("RGB", (256 * cols, 256 * ((len(tiles) + cols - 1) // cols)))
for i, t in enumerate(tiles):
    sheet.paste(t, ((i % cols) * 256, (i // cols) * 256))
sheet.save(out / "sheet.png")
print("ok", len(tiles))
