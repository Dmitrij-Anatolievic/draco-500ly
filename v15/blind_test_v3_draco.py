#!/usr/bin/env python3
"""
blind_test_v3_draco.py
Test dell'ipotesi DICHIARATA IN ANTICIPO dalla v14: "i nodi somigliano al Dragone".

Differenza dal p-value generale di blind_test_v3.py:
  - lì ogni insieme casuale poteva scegliere la costellazione più adatta fra tutte
    (giusto per la domanda "somiglia a QUALCHE costellazione?");
  - qui ogni insieme casuale viene confrontato SOLO con il Dragone, dagli stessi
    punti di vista. È il confronto corretto per un'ipotesi fissata prima.
Si riporta anche il p ristretto alle distanze 400-600 al (l'affermazione della v14).

Uso: python blind_test_v3_draco.py [--null 200] [--hip-cache file.ecsv]
"""
import argparse
import shutil
import time

import numpy as np

import blind_test_v2 as v2
import blind_test_v3 as v3

log = v2.log


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--insiemi", default="rilevati15,repo14,repo16")
    p.add_argument("--modelli", default="uniforme,a_coppie")
    p.add_argument("--direzioni", type=int, default=600)
    p.add_argument("--distanze", default="25,50,75,100,150,200,300,400,500,541,600,800,1000")
    p.add_argument("--top-k", type=int, default=4000)
    p.add_argument("--null", type=int, default=200)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--hip-cache", default=None)
    a = p.parse_args()

    out = v2.CARTELLA_SCRIPT / "gaia_sky_out" / "blind_test_v3"
    out.mkdir(parents=True, exist_ok=True)
    v2.configura_log(out)
    if a.hip_cache:
        dest = v2.CARTELLA_SCRIPT / "gaia_sky_out" / "blind_test_v2" / "hipparcos_hp6.0.ecsv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy(a.hip_cache, dest)
    tab = v2.scarica_hipparcos(6.0)
    distanze = [float(x) for x in a.distanze.split(",")]
    from astropy.table import Table
    righe = []

    for nome in a.insiemi.split(","):
        nodi = v3.INSIEMI[nome]
        n = len(nodi)
        v2.N_NODI = n
        gruppi = v2.stelle_per_costellazione(tab, 5.0)
        forme, desc, info = v2.costruisci_libreria({"Draco": gruppi["Draco"]}, a.direzioni, distanze)
        dly = np.array([x[1] for x in info])
        vicino500 = (dly >= 400) & (dly <= 600)
        nn = v2.normalizza(nodi)
        d_tutti = v2.miglior_match(nn, forme, desc, a.top_k)
        d_500 = v2.miglior_match(nn, forme, desc, a.top_k, maschera=vicino500)
        log.info("%s: dipinto vs Dragone  d=%.4f (da %.0f al) | solo 400-600 al: d=%.4f (da %.0f al)",
                 nome, d_tutti[0], info[d_tutti[1]][1], d_500[0], info[d_500[1]][1])
        _, sep = v3.coppie_strette(nodi)
        riq = ((nodi[:, 0].min(), nodi[:, 0].max()), (nodi[:, 1].min(), nodi[:, 1].max()))
        for mod in a.modelli.split(","):
            rng = np.random.default_rng(a.seed)
            nt, n5 = [], []
            t0 = time.time()
            for k in range(a.null):
                cas = rng.uniform(0, 1, (n, 2)) if mod == "uniforme" else v3.genera_a_coppie(rng, n, riq, sep)
                cn = v2.normalizza(cas)
                nt.append(v2.miglior_match(cn, forme, desc, a.top_k)[0])
                n5.append(v2.miglior_match(cn, forme, desc, a.top_k, maschera=vicino500)[0])
                if (k + 1) % 50 == 0:
                    log.info("   [%s/%s] %d/%d (%.0f s)", nome, mod, k + 1, a.null, time.time() - t0)
            nt, n5 = np.array(nt), np.array(n5)
            p_t = (np.sum(nt <= d_tutti[0]) + 1) / (a.null + 1)
            p_5 = (np.sum(n5 <= d_500[0]) + 1) / (a.null + 1)
            log.info("   %s / %s: p(Dragone, tutte le distanze) = %.3f  [casuali mediana %.4f]"
                     " | p(Dragone, 400-600 al) = %.3f  [casuali mediana %.4f]",
                     nome, mod, p_t, np.median(nt), p_5, np.median(n5))
            righe.append(dict(insieme=nome, modello=mod, d_draco=round(d_tutti[0], 5),
                              da_ly=info[d_tutti[1]][1], p_draco=round(p_t, 4),
                              null_med=round(float(np.median(nt)), 5),
                              d_draco_400_600=round(d_500[0], 5), p_draco_400_600=round(p_5, 4),
                              null_med_400_600=round(float(np.median(n5)), 5), prove=a.null))
            Table(rows=righe).write(out / "draco_ipotesi_dichiarata.csv", format="ascii.csv", overwrite=True)
    log.info("FATTO.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("Errore:")
    finally:
        try:
            input("\nPremi Invio per chiudere...")
        except EOFError:
            pass
