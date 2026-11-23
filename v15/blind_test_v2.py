#!/usr/bin/env python3
"""
blind_test_v2.py
Test cieco corretto per il paper "Draco at 500 Light-Years".

Domanda: i 16 nodi blu del dipinto somigliano alla forma di una costellazione
vista dalla Terra o da un altro punto della Galassia, più di quanto ci si
aspetterebbe per caso?

Correzioni rispetto a blind_test_88.py:
  1. STELLE VERE: posizioni e parallassi dal catalogo Hipparcos (van Leeuwen 2007,
     VizieR I/311/hip2), non valori scritti a mano.
     Per ogni costellazione (confini IAU) si prendono le 16 stelle più brillanti:
     scelta oggettiva, nessuna selezione manuale.
  2. PROIEZIONE CORRETTA: le stelle sono proiettate sul piano del cielo
     (proiezione gnomonica) dal punto di vista dell'osservatore, senza la
     deformazione di usare RA e Dec come coordinate piane.
  3. ABBINAMENTO OTTIMALE: la distanza di Procrustes viene calcolata cercando il
     miglior abbinamento nodo-stella (algoritmo ungherese + rotazioni ripetute).
     Il risultato NON dipende più dall'ordine in cui sono scritti i punti.
     Sono ammesse rotazioni, scala e immagine speculare.
  4. P-VALUE ONESTO: la stessa identica ricerca (tutte le costellazioni, tutti i
     punti di vista) viene ripetuta su insiemi di 16 punti casuali. Il p-value è
     la frazione di insiemi casuali che ottengono un risultato uguale o migliore.

Uso:
    pip install astroquery astropy scipy numpy matplotlib
    python blind_test_v2.py                    # test completo
    python blind_test_v2.py --null 50          # prova veloce (p-value meno preciso)
    python blind_test_v2.py --nodi blue_nodes_16.csv

Output (in gaia_sky_out\\blind_test_v2, accanto allo script):
    classifica_costellazioni.csv, null_distribuzione.csv,
    confronto_migliore.png, confronto_draco.png, distribuzione_null.png,
    log_blind_test_<data_ora>.txt
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

CARTELLA_SCRIPT = Path(__file__).resolve().parent
ANNI_LUCE_PER_PC = 3.261563777
N_NODI = 16

# I 16 nodi blu del dipinto (da data/blue_nodes_16.csv del repo draco-500ly)
NODI_DEFAULT = np.array([
    [0.316, 0.889], [0.532, 0.779], [0.532, 0.757], [0.445, 0.694],
    [0.664, 0.626], [0.487, 0.618], [0.186, 0.557], [0.199, 0.519],
    [0.681, 0.434], [0.699, 0.399], [0.588, 0.390], [0.070, 0.378],
    [0.446, 0.332], [0.334, 0.268], [0.513, 0.188], [0.778, 0.131],
])

log = logging.getLogger("blindv2")


# ================================================================ utilità
def configura_log(cartella: Path) -> Path:
    file_log = cartella / f"log_blind_test_{datetime.now():%Y%m%d_%H%M%S}.txt"
    formato = logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(file_log, encoding="utf-8")):
        h.setFormatter(formato)
        log.addHandler(h)
    log.setLevel(logging.INFO)
    return file_log


def normalizza(p: np.ndarray) -> np.ndarray:
    """Centra e porta a norma unitaria (come scipy.spatial.procrustes)."""
    q = p - p.mean(axis=0)
    s = np.sqrt((q ** 2).sum())
    return q / s if s > 0 else q


def descrittore(p_norm: np.ndarray) -> np.ndarray:
    """Distanze reciproche ordinate: non dipende da ordine, rotazione, specchio, scala."""
    i, j = np.triu_indices(len(p_norm), 1)
    return np.sort(np.linalg.norm(p_norm[i] - p_norm[j], axis=1))


# ================================================================ Procrustes senza ordine
def procrustes_senza_ordine(A: np.ndarray, B: np.ndarray, n_angoli: int = 12,
                            n_iter: int = 10):
    """
    Distanza di Procrustes tra due insiemi di punti SENZA corrispondenza nota.
    A, B: (n,2) già normalizzati. Cerca abbinamento + rotazione (anche speculare)
    ottimali partendo da più rotazioni iniziali. Restituisce (d, perm, B_allineato)
    con d in [0,1] come scipy.spatial.procrustes.
    """
    from scipy.optimize import linear_sum_assignment

    migliore = (2.0, None, None)
    for specchio in (1.0, -1.0):
        Bs = B * np.array([1.0, specchio])
        for k in range(n_angoli):
            t = 2 * np.pi * k / n_angoli
            R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
            Bc = Bs @ R
            perm_prec = None
            for _ in range(n_iter):
                costo = ((A[:, None, :] - Bc[None, :, :]) ** 2).sum(axis=2)
                _, perm = linear_sum_assignment(costo)
                if perm_prec is not None and np.array_equal(perm, perm_prec):
                    break
                perm_prec = perm
                M = B[perm]
                U, S, Vt = np.linalg.svd(M.T @ A)
                Rt = U @ Vt                     # rotazione/riflessione ottimale
                Bc = B @ Rt
            M = B[perm]
            U, S, Vt = np.linalg.svd(M.T @ A)
            d = 1.0 - S.sum() ** 2              # disparità di Procrustes
            if d < migliore[0]:
                migliore = (d, perm, (M @ (U @ Vt)) * S.sum())
    return migliore


# ================================================================ dati Hipparcos
SERVER_VIZIER = [
    "vizier.cds.unistra.fr",      # Strasburgo (principale)
    "vizier.cfa.harvard.edu",     # Harvard (USA)
    "vizier.iucaa.in",            # IUCAA (India)
    "vizier.nao.ac.jp",           # NAOJ (Giappone)
]


def scarica_hipparcos(mag_limite: float):
    """
    Scarica Hipparcos (van Leeuwen 2007) provando più server VizieR.
    Il risultato viene salvato in una copia locale: le esecuzioni successive
    non hanno più bisogno di internet.
    """
    from astropy.table import Table
    cache = CARTELLA_SCRIPT / "gaia_sky_out" / "blind_test_v2" / f"hipparcos_hp{mag_limite:.1f}.ecsv"
    if cache.exists():
        tab = Table.read(cache)
        log.info("Hipparcos letto dalla copia locale %s (%d stelle).", cache.name, len(tab))
        return tab

    from astroquery.vizier import Vizier
    log.info("Scarico Hipparcos (I/311/hip2) fino a Hp < %.1f ...", mag_limite)
    for server in SERVER_VIZIER:
        try:
            log.info("  provo il server %s ...", server)
            v = Vizier(columns=["HIP", "RArad", "DErad", "Plx", "e_Plx", "Hpmag"],
                       column_filters={"Hpmag": f"<{mag_limite}"}, row_limit=-1,
                       timeout=120, vizier_server=server)
            t0 = time.time()
            ris = v.query_constraints(catalog="I/311/hip2")
            if len(ris) == 0 or len(ris[0]) == 0:
                log.warning("  nessun dato da %s", server)
                continue
            tab = ris[0]
            log.info("  %d stelle ricevute da %s (%.0f s).", len(tab), server, time.time() - t0)
            tab.write(cache, overwrite=True)
            log.info("  copia locale salvata: %s", cache.name)
            return tab
        except Exception as e:
            log.warning("  %s non risponde: %s", server, str(e).splitlines()[0][:120])
    raise RuntimeError("Nessun server VizieR raggiungibile. Riprova più tardi.")


def stelle_per_costellazione(tab, min_plx_snr: float):
    import astropy.units as u
    from astropy.coordinates import SkyCoord, get_constellation

    ra = np.asarray(tab["RArad"], dtype=float)
    dec = np.asarray(tab["DErad"], dtype=float)
    plx = np.asarray(tab["Plx"], dtype=float)
    eplx = np.asarray(tab["e_Plx"], dtype=float)
    mag = np.asarray(tab["Hpmag"], dtype=float)
    hip = np.asarray(tab["HIP"])
    ok = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(plx) & (plx > 0) \
        & (plx / np.where(eplx > 0, eplx, np.inf) >= min_plx_snr)
    log.info("Stelle con parallasse affidabile (plx/err >= %.0f): %d su %d",
             min_plx_snr, int(ok.sum()), len(ok))
    cost = get_constellation(SkyCoord(ra[ok] * u.deg, dec[ok] * u.deg))

    gruppi = {}
    for nome in sorted(set(cost)):
        m = cost == nome
        if m.sum() < N_NODI:
            continue
        idx = np.argsort(mag[ok][m])[:N_NODI]         # le 16 più brillanti
        gruppi[nome] = {
            "ra": ra[ok][m][idx], "dec": dec[ok][m][idx],
            "dist_pc": 1000.0 / plx[ok][m][idx], "mag": mag[ok][m][idx],
            "hip": hip[ok][m][idx],
        }
    log.info("Costellazioni con almeno %d stelle utilizzabili: %d", N_NODI, len(gruppi))
    return gruppi


# ================================================================ geometria
def xyz(ra_deg, dec_deg, dist):
    r, d = np.radians(ra_deg), np.radians(dec_deg)
    return np.stack([dist * np.cos(d) * np.cos(r),
                     dist * np.cos(d) * np.sin(r),
                     dist * np.sin(d)], axis=-1)


def proietta(stelle_xyz: np.ndarray, oss: np.ndarray):
    """Proiezione gnomonica attorno alla direzione media, vista da 'oss'."""
    v = stelle_xyz - oss
    n = np.linalg.norm(v, axis=1, keepdims=True)
    u = v / n
    c = u.mean(axis=0)
    c /= np.linalg.norm(c)
    cosang = u @ c
    if cosang.min() < 0.2:          # figura troppo estesa (osservatore "dentro")
        return None
    e1 = np.cross([0, 0, 1.0], c)
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross([0, 1.0, 0], c)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(c, e1)
    x = (u @ e1) / cosang
    y = (u @ e2) / cosang
    return np.stack([-x, y], axis=1)   # -x: est a sinistra, come si vede il cielo


def direzioni_fibonacci(n: int):
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n)
    theta = np.pi * (1 + 5 ** 0.5) * k
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi),
                     np.cos(phi)], axis=1)


def costruisci_libreria(gruppi, n_dir: int, distanze_ly):
    """Tutte le forme: (costellazione, punto di vista) -> 16 punti normalizzati."""
    dirs = direzioni_fibonacci(n_dir)
    punti_vista = [(np.zeros(3), 0.0, None)]
    for d_ly in distanze_ly:
        d_pc = d_ly / ANNI_LUCE_PER_PC
        for k, u in enumerate(dirs):
            punti_vista.append((u * d_pc, d_ly, k))
    forme, info = [], []
    for nome, g in gruppi.items():
        sx = xyz(g["ra"], g["dec"], g["dist_pc"])
        for oss, d_ly, k in punti_vista:
            p = proietta(sx, oss)
            if p is None:
                continue
            forme.append(normalizza(p))
            if k is None:
                ra_o, dec_o = 0.0, 0.0
            else:
                u = dirs[k]
                ra_o = np.degrees(np.arctan2(u[1], u[0])) % 360
                dec_o = np.degrees(np.arcsin(u[2]))
            info.append((nome, d_ly, ra_o, dec_o))
    forme = np.array(forme)
    desc = np.array([descrittore(f) for f in forme], dtype=np.float32)
    return forme, desc, info


def miglior_match(nodi_norm, forme, desc, top_k: int, maschera=None, finali: int = 40):
    """
    Ricerca in due stadi:
      1. prefiltro con il descrittore (distanze reciproche ordinate) -> top_k forme
      2. Procrustes rapido (4 angoli, 4 iterazioni) su quelle -> le 'finali' migliori
      3. Procrustes completo (12 angoli x specchio) sulle finali
    La stessa identica procedura è usata per il dipinto e per i punti casuali.
    """
    dn = descrittore(nodi_norm).astype(np.float32)
    dist = ((desc - dn) ** 2).sum(axis=1)
    if maschera is not None:
        dist = np.where(maschera, dist, np.inf)
    cand = np.argsort(dist)[:top_k]
    cand = cand[np.isfinite(dist[cand])]
    rapidi = np.array([procrustes_senza_ordine(nodi_norm, forme[i], n_angoli=4, n_iter=4)[0]
                       for i in cand])
    best = (2.0, -1, None)
    for i in cand[np.argsort(rapidi)[:finali]]:
        d, _, allineata = procrustes_senza_ordine(nodi_norm, forme[i])
        if d < best[0]:
            best = (d, i, allineata)
    return best


# ================================================================ grafici
def grafico_confronto(nodi_norm, forma_allineata, titolo, file_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(nodi_norm[:, 0], nodi_norm[:, 1], s=60, c="tab:blue", label="nodi del dipinto")
    ax.scatter(forma_allineata[:, 0], forma_allineata[:, 1], s=40, marker="x",
               c="tab:red", label="stelle (allineate)")
    for a, b in zip(nodi_norm, forma_allineata):
        ax.plot([a[0], b[0]], [a[1], b[1]], c="gray", lw=0.6)
    ax.set_title(titolo, fontsize=10)
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(file_png, dpi=130)
    plt.close(fig)


def grafico_null(null_d, oss_d, titolo, file_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(null_d, bins=30, color="lightgray", edgecolor="gray",
            label="16 punti casuali (stessa ricerca)")
    ax.axvline(oss_d, color="tab:red", lw=2, label=f"dipinto: d = {oss_d:.3f}")
    ax.set_xlabel("miglior distanza di Procrustes trovata")
    ax.set_ylabel("numero di prove")
    ax.set_title(titolo, fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(file_png, dpi=130)
    plt.close(fig)


# ================================================================ main
def main():
    p = argparse.ArgumentParser(description="Test cieco v2 (Hipparcos, senza ordine)")
    p.add_argument("--nodi", default=None, help="CSV con colonne x_norm,y_norm (default: 16 nodi del repo)")
    p.add_argument("--mag", type=float, default=6.0, help="magnitudine limite Hipparcos (default 6.0)")
    p.add_argument("--plx-snr", type=float, default=5.0, help="parallasse/errore minimo (default 5)")
    p.add_argument("--direzioni", type=int, default=300, help="direzioni per ogni distanza (default 300)")
    p.add_argument("--distanze", default="50,100,200,300,400,500,541,600,800,1000",
                   help="distanze dell'osservatore in anni luce")
    p.add_argument("--top-k", type=int, default=4000, help="forme pre-selezionate per ogni ricerca (default 4000)")
    p.add_argument("--null", type=int, default=300, help="insiemi casuali per il p-value (default 300, ~12 min)")
    p.add_argument("--seed", type=int, default=2026)
    a = p.parse_args()

    out = CARTELLA_SCRIPT / "gaia_sky_out" / "blind_test_v2"
    out.mkdir(parents=True, exist_ok=True)
    file_log = configura_log(out)
    log.info("Cartella: %s", out.resolve())
    log.info("File di log: %s", file_log.name)
    rng = np.random.default_rng(a.seed)

    # ---- nodi
    if a.nodi:
        from astropy.table import Table
        t = Table.read(a.nodi, format="ascii.csv")
        nodi = np.stack([np.asarray(t["x_norm"], float), np.asarray(t["y_norm"], float)], axis=1)
    else:
        nodi = NODI_DEFAULT
    if len(nodi) != N_NODI:
        raise ValueError(f"Servono {N_NODI} nodi, trovati {len(nodi)}")
    nodi_norm = normalizza(nodi)
    log.info("Nodi del dipinto: %d", len(nodi))

    # ---- stelle e libreria di forme
    tab = scarica_hipparcos(a.mag)
    gruppi = stelle_per_costellazione(tab, a.plx_snr)
    if "Draco" in gruppi:
        g = gruppi["Draco"]
        log.info("Dragone, 16 stelle usate (HIP, mag, distanza al):")
        for h, m, d in zip(g["hip"], g["mag"], g["dist_pc"] * ANNI_LUCE_PER_PC):
            log.info("   HIP %-7s  Hp=%.2f  %.0f al", h, m, d)
    distanze = [float(x) for x in a.distanze.split(",")]
    log.info("Costruisco le forme: %d costellazioni x (1 + %d direzioni x %d distanze)...",
             len(gruppi), a.direzioni, len(distanze))
    t0 = time.time()
    forme, desc, info = costruisci_libreria(gruppi, a.direzioni, distanze)
    log.info("  %d forme pronte (%.0f s).", len(forme), time.time() - t0)
    nomi = np.array([x[0] for x in info])
    da_terra = np.array([x[1] == 0.0 for x in info])

    # ---- STADIO A: dalla Terra, tutte le costellazioni, confronto esatto
    log.info("=" * 64)
    log.info("STADIO A - dalla Terra (confronto esatto, senza ordine)")
    risultati = {}
    for i in np.where(da_terra)[0]:
        d, _, _ = procrustes_senza_ordine(nodi_norm, forme[i])
        risultati[nomi[i]] = {"d_terra": d}
    classifica = sorted(risultati.items(), key=lambda kv: kv[1]["d_terra"])
    for pos, (n, r) in enumerate(classifica[:10], 1):
        log.info("  %2d. %-20s d = %.4f", pos, n, r["d_terra"])
    pos_draco = [n for n, _ in classifica].index("Draco") + 1 if "Draco" in risultati else None
    if pos_draco:
        log.info("  Dragone dalla Terra: posizione %d su %d, d = %.4f",
                 pos_draco, len(classifica), risultati["Draco"]["d_terra"])

    # ---- STADIO B: miglior punto di vista per ogni costellazione
    log.info("=" * 64)
    log.info("STADIO B - miglior punto di vista per ogni costellazione")
    t0 = time.time()
    for j, n in enumerate(risultati, 1):
        if j % 10 == 0:
            log.info("  %d/%d costellazioni (%.0f s)", j, len(risultati), time.time() - t0)
        d, i, alli = miglior_match(nodi_norm, forme, desc, a.top_k, maschera=(nomi == n))
        risultati[n].update({"d_migliore": d, "da_ly": info[i][1], "ra_oss": info[i][2],
                             "dec_oss": info[i][3], "allineata": alli})
    log.info("  fatto in %.0f s", time.time() - t0)
    classifica_b = sorted(risultati.items(), key=lambda kv: kv[1]["d_migliore"])
    for pos, (n, r) in enumerate(classifica_b[:10], 1):
        log.info("  %2d. %-20s d = %.4f   da %4.0f al verso RA %5.1f Dec %+5.1f",
                 pos, n, r["d_migliore"], r["da_ly"], r["ra_oss"], r["dec_oss"])
    if "Draco" in risultati:
        r = risultati["Draco"]
        pos_b = [n for n, _ in classifica_b].index("Draco") + 1
        log.info("  Dragone: posizione %d, d = %.4f da %.0f al (RA %.1f, Dec %+.1f)",
                 pos_b, r["d_migliore"], r["da_ly"], r["ra_oss"], r["dec_oss"])

    best_nome, best = classifica_b[0]

    # ---- p-value: stessa ricerca su 16 punti casuali
    log.info("=" * 64)
    log.info("P-VALUE - stessa ricerca ripetuta su %d insiemi di 16 punti casuali", a.null)
    null_d = []
    t0 = time.time()
    for k in range(a.null):
        casuali = normalizza(rng.uniform(0, 1, (N_NODI, 2)))
        d, _, _ = miglior_match(casuali, forme, desc, a.top_k)
        null_d.append(d)
        if (k + 1) % 10 == 0:
            el = time.time() - t0
            log.info("  %d/%d  (restano ~%.0f min)", k + 1, a.null,
                     el / (k + 1) * (a.null - k - 1) / 60)
    null_d = np.array(null_d)
    p_val = (np.sum(null_d <= best["d_migliore"]) + 1) / (len(null_d) + 1)
    log.info("  distribuzione casuale: min %.4f  mediana %.4f  max %.4f",
             null_d.min(), np.median(null_d), null_d.max())
    log.info("  miglior risultato del dipinto: %s, d = %.4f", best_nome, best["d_migliore"])
    log.info("  p-value = %.4f  (frazione di insiemi casuali con d <= %.4f)",
             p_val, best["d_migliore"])
    if "Draco" in risultati:
        p_draco = (np.sum(null_d <= risultati["Draco"]["d_migliore"]) + 1) / (len(null_d) + 1)
        log.info("  p-value usando il risultato del Dragone (d = %.4f): %.4f",
                 risultati["Draco"]["d_migliore"], p_draco)

    # ---- salvataggi
    from astropy.table import Table
    Table(rows=[{"costellazione": n, "d_terra": round(r["d_terra"], 5),
                 "d_migliore": round(r["d_migliore"], 5), "da_ly": r["da_ly"],
                 "ra_oss": round(r["ra_oss"], 2), "dec_oss": round(r["dec_oss"], 2)}
                for n, r in classifica_b]).write(out / "classifica_costellazioni.csv",
                                                 format="ascii.csv", overwrite=True)
    Table({"d_migliore_casuale": null_d}).write(out / "null_distribuzione.csv",
                                                format="ascii.csv", overwrite=True)
    grafico_confronto(nodi_norm, best["allineata"],
                      f"Miglior match: {best_nome} da {best['da_ly']:.0f} al, d={best['d_migliore']:.3f}",
                      out / "confronto_migliore.png")
    if "Draco" in risultati:
        r = risultati["Draco"]
        grafico_confronto(nodi_norm, r["allineata"],
                          f"Dragone da {r['da_ly']:.0f} al, d={r['d_migliore']:.3f}",
                          out / "confronto_draco.png")
    grafico_null(null_d, best["d_migliore"],
                 f"p-value = {p_val:.3f} ({a.null} insiemi casuali)", out / "distribuzione_null.png")
    log.info("FATTO. File in: %s", out.resolve())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.warning("Interrotto dall'utente.")
    except Exception:
        log.exception("Errore:")
    finally:
        try:
            input("\nPremi Invio per chiudere...")
        except EOFError:
            pass
