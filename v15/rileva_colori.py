"""Rilevamento automatico dei puntini colorati sul dipinto.
Criteri FISSATI PRIMA di qualsiasi test astronomico (22/11/2026):
  HSV (H in gradi, S e V in 0-1), solo dentro la tela, macchie di area 40-6000 px
  verde:     H 85-170, S>0.35, V>0.30
  giallo:    H 45-70,  S>0.45, V>0.55
  arancione: H 15-44,  S>0.50, V>0.50
  bianco:    S<0.18,   V>0.80
I trattini blu e il contorno magenta sono esclusi dai criteri (H 190-250 e H 300-350)."""
import numpy as np, sys
from PIL import Image
from scipy import ndimage
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, colorsys, csv
F = sys.argv[1]
im = Image.open(F).convert("RGB"); a = np.asarray(im).astype(float) / 255
H0, W0 = a.shape[:2]
x0, x1, y0, y1 = 190, 2650, 130, 3750            # tela, esclusi bordi e muro
r, g, b = a[..., 0], a[..., 1], a[..., 2]
mx, mn = a.max(2), a.min(2); v = mx; s = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
d = np.maximum(mx - mn, 1e-6)
h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
crit = {"verde": (h >= 85) & (h <= 170) & (s > .35) & (v > .30),
        "giallo": (h >= 45) & (h <= 70) & (s > .45) & (v > .55),
        "arancione": (h >= 15) & (h < 45) & (s > .50) & (v > .50),
        "bianco": (s < .18) & (v > .80)}
tela = np.zeros_like(v, bool); tela[y0:y1, x0:x1] = True
col = {"verde": "lime", "giallo": "yellow", "arancione": "orange", "bianco": "white"}
fig, ax = plt.subplots(figsize=(9, 12)); ax.imshow(im); ax.set_axis_off()
rows = []
for nome, m in crit.items():
    lab, n = ndimage.label(m & tela)
    if n == 0: continue
    aree = ndimage.sum(np.ones_like(v), lab, range(1, n + 1))
    cm = ndimage.center_of_mass(np.ones_like(v), lab, range(1, n + 1))
    k = 0
    for ar, (cy, cx) in zip(aree, cm):
        if 40 <= ar <= 6000:
            k += 1
            rows.append({"colore": nome, "id": f"{nome[0].upper()}{k:02d}", "x_px": round(cx), "y_px": round(cy),
                         "x_norm": round(cx / W0, 4), "y_norm": round(1 - cy / H0, 4), "area_px": int(ar)})
            ax.scatter(cx, cy, s=120, facecolors="none", edgecolors=col[nome], lw=1.2)
    print(nome, k)
with open("puntini_rilevati.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
ax.set_title("Puntini rilevati automaticamente (cerchi: verde, giallo, arancione, bianco)")
fig.savefig("puntini_rilevati.png", dpi=110, bbox_inches="tight")
