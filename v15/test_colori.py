#!/usr/bin/env python3
"""Test dei puntini verdi e gialli rilevati automaticamente (criterio fissato prima, rileva_colori.py).
Domanda aperta, come per i trattini blu: miglior costellazione e punto di vista, p su 100 insiemi
casuali uniformi con la stessa ricerca. Si riportano anche le posizioni di Dragone, Lira e Boote
(le costellazioni nominate dalla v14 per i colori)."""
import sys, time, csv
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np, blind_test_v2 as v2
from astropy.table import Table
log = v2.log
out = v2.CARTELLA_SCRIPT / "gaia_sky_out" / "colori"; out.mkdir(parents=True, exist_ok=True)
v2.configura_log(out)
rows = list(csv.DictReader(open(__import__("pathlib").Path(__file__).resolve().parent / "puntini_rilevati.csv")))
tab = v2.scarica_hipparcos(6.0)
ris = []
for colore in ["verde", "giallo"]:
    nodi = np.array([[float(r["x_norm"]), float(r["y_norm"])] for r in rows if r["colore"] == colore])
    n = len(nodi); v2.N_NODI = n
    g = v2.stelle_per_costellazione(tab, 5.0)
    forme, desc, info = v2.costruisci_libreria(g, 600, [25,50,75,100,150,200,300,400,500,541,600,800,1000])
    nomi = np.array([x[0] for x in info]); nn = v2.normalizza(nodi)
    best = {}
    for c in g:
        d, i, _ = v2.miglior_match(nn, forme, desc, 4000, maschera=(nomi == c))
        best[c] = (d, info[i][1])
    cl = sorted(best.items(), key=lambda kv: kv[1][0])
    rng = np.random.default_rng(31); nd = []
    for k in range(100):
        nd.append(v2.miglior_match(v2.normalizza(rng.uniform(0, 1, (n, 2))), forme, desc, 4000)[0])
    nd = np.array(nd); p = (np.sum(nd <= cl[0][1][0]) + 1) / 101
    pos = {c: [x for x, _ in cl].index(c) + 1 for c in ["Draco", "Lyra", "Bootes"] if c in best}
    log.info("%s (%d punti): miglior %s d=%.4f da %.0f al, p=%.3f | posizioni %s su %d",
             colore, n, cl[0][0], cl[0][1][0], cl[0][1][1], p, pos, len(cl))
    ris.append(dict(colore=colore, n=n, miglior=cl[0][0], d=round(cl[0][1][0], 5), da_ly=cl[0][1][1], p=round(p, 4),
                    null_mediana=round(float(np.median(nd)), 5), pos_draco=pos.get("Draco"), pos_lyra=pos.get("Lyra"),
                    pos_bootes=pos.get("Bootes"), costellazioni=len(cl)))
    Table(rows=ris).write(out / "risultati_colori.csv", format="ascii.csv", overwrite=True)
log.info("FATTO")
