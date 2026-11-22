#!/usr/bin/env python3
"""
blind_test_v3.py
Verifica di robustezza del test cieco v2 per "Draco at 500 Light-Years" (punto 1 della v15).

Cosa cambia rispetto a blind_test_v2.py (che resta invariato e viene importato):
  1. INSIEMI DI NODI: oltre ai 16 nodi del repo si provano
       - "rilevati15": i 15 trattini blu rilevati sulla foto del dipinto
       - "repo14":     i 16 nodi del repo senza B05 e B12 (nessun trattino in quei punti)
     Il numero di stelle per costellazione segue il numero di nodi (N più brillanti).
  2. DUE MODELLI CASUALI per il p-value:
       - "uniforme":  N punti uniformi nel quadrato (come la v2)
       - "a_coppie":  stessi riquadro, numero di coppie strette e distanze di coppia
                      del dipinto; resto uniforme. È un modello più severo, perché
                      il dipinto ha coppie di trattini vicini che un insieme
                      uniforme raramente ha.
  3. GRIGLIA PIÙ FITTA: più direzioni e più distanze dell'osservatore.

Uso:
    python blind_test_v3.py                              # tutti gli insiemi, tutti i modelli
    python blind_test_v3.py --insiemi rilevati15 --null 200
Output: gaia_sky_out\\blind_test_v3\\ (riepilogo_v3.csv, classifiche, grafici, log)
"""

import argparse
import shutil
import time
from pathlib import Path

import numpy as np

import blind_test_v2 as v2

RILEVATI15 = np.array([
    [0.313, .912], [.515, .821], [.532, .781], [.443, .704], [.487, .624],
    [.171, .569], [.183, .530], [.689, .443], [.709, .405], [.592, .397],
    [.435, .348], [.446, .318], [.319, .268], [.510, .185], [.799, .121]])
REPO16 = v2.NODI_DEFAULT
REPO14 = np.delete(REPO16, [4, 11], axis=0)        # senza B05 e B12
INSIEMI = {"rilevati15": RILEVATI15, "repo16": REPO16, "repo14": REPO14}

log = v2.log


# ================================================================ modello casuale a coppie
def coppie_strette(p: np.ndarray, soglia_rel: float = 0.5):
    """Coppie di vicini reciproci con distanza < soglia_rel x mediana dei SECONDI vicini
    (la mediana dei primi vicini non va: nel dipinto metà dei punti sono in coppia)."""
    d = np.linalg.norm(p[:, None] - p[None], axis=2)
    np.fill_diagonal(d, np.inf)
    nn = d.argmin(axis=1)
    dnn = d.min(axis=1)
    soglia = soglia_rel * np.median(np.sort(d, axis=1)[:, 1])
    coppie = sorted({tuple(sorted((i, int(nn[i])))) for i in range(len(p))
                     if nn[nn[i]] == i and dnn[i] < soglia})
    return coppie, np.array([d[i, j] for i, j in coppie])


def genera_a_coppie(rng, n: int, riquadro, sep: np.ndarray):
    (x0, x1), (y0, y1) = riquadro
    k = len(sep)
    punti = []
    for s in sep:                                   # coppie con le stesse distanze
        c = rng.uniform([x0, y0], [x1, y1])
        t = rng.uniform(0, 2 * np.pi)
        punti += [c, c + s * np.array([np.cos(t), np.sin(t)])]
    punti += list(rng.uniform([x0, y0], [x1, y1], (n - 2 * k, 2)))
    return np.array(punti)


# ================================================================ main
def main():
    p = argparse.ArgumentParser(description="Test cieco v3 (robustezza)")
    p.add_argument("--insiemi", default="rilevati15,repo14,repo16")
    p.add_argument("--modelli", default="uniforme,a_coppie")
    p.add_argument("--mag", type=float, default=6.0)
    p.add_argument("--plx-snr", type=float, default=5.0)
    p.add_argument("--direzioni", type=int, default=600)
    p.add_argument("--distanze", default="25,50,75,100,150,200,300,400,500,541,600,800,1000")
    p.add_argument("--top-k", type=int, default=4000)
    p.add_argument("--null", type=int, default=200)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--hip-cache", default=None, help="file .ecsv Hipparcos già scaricato")
    a = p.parse_args()

    out = v2.CARTELLA_SCRIPT / "gaia_sky_out" / "blind_test_v3"
    out.mkdir(parents=True, exist_ok=True)
    file_log = v2.configura_log(out)
    log.info("Cartella: %s  | log: %s", out.resolve(), file_log.name)

    if a.hip_cache:                                  # mette la cache dove la cerca la v2
        dest = v2.CARTELLA_SCRIPT / "gaia_sky_out" / "blind_test_v2" / f"hipparcos_hp{a.mag:.1f}.ecsv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy(a.hip_cache, dest)
    tab = v2.scarica_hipparcos(a.mag)
    distanze = [float(x) for x in a.distanze.split(",")]
    modelli = a.modelli.split(",")
    from astropy.table import Table
    riepilogo = []

    for nome_ins in a.insiemi.split(","):
        nodi = INSIEMI[nome_ins]
        n = len(nodi)
        v2.N_NODI = n                                # N stelle più brillanti per costellazione
        log.info("=" * 70)
        log.info("INSIEME %s: %d nodi", nome_ins, n)
        nodi_norm = v2.normalizza(nodi)
        coppie, sep = coppie_strette(nodi)
        log.info("  coppie strette nel dipinto: %d  %s  (distanze %s)", len(coppie),
                 coppie, np.round(sep, 3).tolist())

        gruppi = v2.stelle_per_costellazione(tab, a.plx_snr)
        t0 = time.time()
        forme, desc, info = v2.costruisci_libreria(gruppi, a.direzioni, distanze)
        nomi = np.array([x[0] for x in info])
        log.info("  %d forme (%d costellazioni x %d punti di vista) in %.0f s",
                 len(forme), len(gruppi), 1 + a.direzioni * len(distanze), time.time() - t0)

        # dalla Terra
        d_terra = {}
        for i in np.where(np.array([x[1] == 0.0 for x in info]))[0]:
            d_terra[nomi[i]] = v2.procrustes_senza_ordine(nodi_norm, forme[i])[0]
        # miglior punto di vista per costellazione
        ris = {}
        t0 = time.time()
        for c in d_terra:
            d, i, alli = v2.miglior_match(nodi_norm, forme, desc, a.top_k, maschera=(nomi == c))
            ris[c] = dict(d_terra=d_terra[c], d=d, da_ly=info[i][1], ra=info[i][2],
                          dec=info[i][3], alli=alli)
        log.info("  ricerca per costellazione: %.0f s", time.time() - t0)
        clas = sorted(ris.items(), key=lambda kv: kv[1]["d"])
        for pos, (c, r) in enumerate(clas[:8], 1):
            log.info("   %2d. %-18s d=%.4f  da %4.0f al (RA %5.1f Dec %+5.1f)  | dalla Terra d=%.4f",
                     pos, c, r["d"], r["da_ly"], r["ra"], r["dec"], r["d_terra"])
        pos_dra = [c for c, _ in clas].index("Draco") + 1
        log.info("   Dragone: posizione %d su %d, d=%.4f da %.0f al",
                 pos_dra, len(clas), ris["Draco"]["d"], ris["Draco"]["da_ly"])
        Table(rows=[{"costellazione": c, "d_terra": round(r["d_terra"], 5), "d_migliore": round(r["d"], 5),
                     "da_ly": r["da_ly"], "ra_oss": round(r["ra"], 2), "dec_oss": round(r["dec"], 2)}
                    for c, r in clas]).write(out / f"classifica_{nome_ins}.csv",
                                             format="ascii.csv", overwrite=True)
        best_c, best = clas[0]
        v2.grafico_confronto(nodi_norm, best["alli"],
                             f"{nome_ins}: {best_c} da {best['da_ly']:.0f} al, d={best['d']:.3f}",
                             out / f"confronto_{nome_ins}.png")

        # p-value con ciascun modello casuale
        riq = ((nodi[:, 0].min(), nodi[:, 0].max()), (nodi[:, 1].min(), nodi[:, 1].max()))
        for mod in modelli:
            rng = np.random.default_rng(a.seed)
            null_d = []
            t0 = time.time()
            for k in range(a.null):
                if mod == "uniforme":
                    cas = rng.uniform(0, 1, (n, 2))
                else:
                    cas = genera_a_coppie(rng, n, riq, sep)
                null_d.append(v2.miglior_match(v2.normalizza(cas), forme, desc, a.top_k)[0])
                if (k + 1) % 25 == 0:
                    el = time.time() - t0
                    log.info("   [%s/%s] %d/%d (restano ~%.0f min)", nome_ins, mod, k + 1, a.null,
                             el / (k + 1) * (a.null - k - 1) / 60)
            null_d = np.array(null_d)
            pv = (np.sum(null_d <= best["d"]) + 1) / (a.null + 1)
            pv_dra = (np.sum(null_d <= ris["Draco"]["d"]) + 1) / (a.null + 1)
            log.info("   %s / %s: casuali min %.4f mediana %.4f | dipinto %.4f (%s) -> p = %.3f"
                     " | Dragone p = %.3f", nome_ins, mod, null_d.min(), np.median(null_d),
                     best["d"], best_c, pv, pv_dra)
            Table({"d": null_d}).write(out / f"null_{nome_ins}_{mod}.csv",
                                       format="ascii.csv", overwrite=True)
            v2.grafico_null(null_d, best["d"], f"{nome_ins} / {mod}: p = {pv:.3f} ({a.null} prove)",
                            out / f"null_{nome_ins}_{mod}.png")
            riepilogo.append(dict(insieme=nome_ins, n_nodi=n, coppie=len(coppie), modello=mod,
                                  miglior=best_c, d=round(best["d"], 5), da_ly=best["da_ly"],
                                  null_mediana=round(float(np.median(null_d)), 5),
                                  null_min=round(float(null_d.min()), 5), p=round(pv, 4),
                                  draco_pos=pos_dra, draco_d=round(ris["Draco"]["d"], 5),
                                  draco_p=round(pv_dra, 4), prove=a.null))
            Table(rows=riepilogo).write(out / "riepilogo_v3.csv", format="ascii.csv", overwrite=True)

    log.info("=" * 70)
    for r in riepilogo:
        log.info("  %-11s %-9s miglior %-12s d=%.4f  p=%.3f | Dragone pos %d p=%.3f",
                 r["insieme"], r["modello"], r["miglior"], r["d"], r["p"], r["draco_pos"], r["draco_p"])
    log.info("FATTO. File in %s", out.resolve())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.warning("Interrotto.")
    except Exception:
        log.exception("Errore:")
    finally:
        try:
            input("\nPremi Invio per chiudere...")
        except EOFError:
            pass
