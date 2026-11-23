#!/usr/bin/env python3
"""Correzioni dalla revisione di Gwen (22/11/2026), regole fissate PRIMA di eseguire:
 1. modello 'a_coppie' corretto: i punti fuori dal riquadro vengono riflessi dentro;
 2. controllo C completo: p per TUTTE le costellazioni (rilevati15, 400-600 al), classifica per p;
 3. robustezza: 20 repliche dei 15 trattini spostati a caso di +-0.01 (circa mezza larghezza
    di un trattino), p del Dragone per ciascuna.
Regola di decisione dichiarata: segnale solo se p(Dragone) < 0.05, Dragone 1o per p,
e p < 0.05 in almeno 16 repliche su 20."""
import numpy as np, sys, time
import blind_test_v2 as v2, blind_test_v3 as v3
from astropy.table import Table
log = v2.log

def a_coppie_corretto(rng, n, riq, sep):
    p = v3.genera_a_coppie(rng, n, riq, sep)
    for j, (lo, hi) in enumerate(riq):
        p[:, j] = np.where(p[:, j] < lo, 2 * lo - p[:, j], p[:, j])
        p[:, j] = np.where(p[:, j] > hi, 2 * hi - p[:, j], p[:, j])
    return p

def main():
    NULL, TOPK, SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 100, 500, 11
    out = v2.CARTELLA_SCRIPT / "gaia_sky_out" / "blind_test_v3"
    v2.configura_log(out)
    tab = v2.scarica_hipparcos(6.0)
    nodi = v3.RILEVATI15; v2.N_NODI = 15
    g = v2.stelle_per_costellazione(tab, 5.0)
    forme, desc, info = v2.costruisci_libreria(g, 600, [400, 500, 541, 600])
    nomi = np.array([x[0] for x in info])
    _, sep = v3.coppie_strette(nodi)
    riq = ((nodi[:, 0].min(), nodi[:, 0].max()), (nodi[:, 1].min(), nodi[:, 1].max()))
    rng = np.random.default_rng(SEED)
    casuali = [v2.normalizza(a_coppie_corretto(rng, 15, riq, sep)) for _ in range(NULL)]
    righe = []; t0 = time.time()
    null_draco = None
    for k, c in enumerate(sorted(g), 1):
        m = nomi == c
        d0 = v2.miglior_match(v2.normalizza(nodi), forme, desc, TOPK, maschera=m)[0]
        nd = np.array([v2.miglior_match(x, forme, desc, TOPK, maschera=m)[0] for x in casuali])
        if c == "Draco": null_draco = nd
        p = (np.sum(nd <= d0) + 1) / (NULL + 1)
        righe.append(dict(costellazione=c, d=round(d0, 5), p=round(p, 4), null_mediana=round(float(np.median(nd)), 5)))
        log.info("%2d/%d %-20s d=%.4f p=%.3f  (%.0f min)", k, len(g), c, d0, p, (time.time() - t0) / 60)
        Table(rows=sorted(righe, key=lambda r: (r["p"], r["d"]))).write(out / "controllo_C_tutte.csv", format="ascii.csv", overwrite=True)
    cl = sorted(righe, key=lambda r: (r["p"], r["d"]))
    pos = [r["costellazione"] for r in cl].index("Draco") + 1
    nsig = sum(r["p"] < 0.05 for r in cl)
    log.info("CONTROLLO C: Dragone %d su %d per p; costellazioni con p<0.05: %d", pos, len(cl), nsig)
    m = nomi == "Draco"; rj = np.random.default_rng(SEED + 1); rep = []
    for i in range(20):
        nj = nodi + rj.uniform(-0.01, 0.01, nodi.shape)
        d = v2.miglior_match(v2.normalizza(nj), forme, desc, TOPK, maschera=m)[0]
        rep.append(dict(replica=i + 1, d=round(d, 5), p=round((np.sum(null_draco <= d) + 1) / (NULL + 1), 4)))
        log.info("  jitter %2d: d=%.4f p=%.3f", i + 1, d, rep[-1]["p"])
    Table(rows=rep).write(out / "robustezza_jitter.csv", format="ascii.csv", overwrite=True)
    ok = sum(r["p"] < 0.05 for r in rep)
    log.info("ROBUSTEZZA: p<0.05 in %d repliche su 20", ok)
    pd = [r for r in cl if r["costellazione"] == "Draco"][0]["p"]
    log.info("REGOLA DI DECISIONE: p(Dragone)=%.3f (<0.05? %s), primo per p? %s, repliche ok %d/20 (>=16? %s)",
             pd, pd < 0.05, pos == 1, ok, ok >= 16)

if __name__ == "__main__":
    main()
