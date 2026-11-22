#!/usr/bin/env python3
"""
cerca_transiti_tess.py
Cerca possibili transiti planetari NON catalogati nelle curve di luce TESS
delle stelle del corridoio (output di gaia_sky_corridoio.py).

Per ogni stella:
  1. cerca su MAST le curve di luce TESS già pronte (SPOC, TESS-SPOC, QLP)
  2. le scarica, le unisce e le "appiattisce" (toglie la variabilità lenta della stella)
  3. cerca segnali periodici a forma di scatola con BLS (Box Least Squares)
  4. calcola SNR, numero di transiti, controllo dispari/pari ed eclisse secondaria
     (se i transiti dispari e pari hanno profondità diverse, è quasi certamente
     una binaria a eclisse e non un pianeta)
  5. salva un grafico per ogni segnale interessante

IMPORTANTE: un segnale trovato qui è un CANDIDATO DA VERIFICARE, non una scoperta.
Le cause più comuni di falsi positivi sono binarie a eclisse (anche stelle vicine
che "contaminano" il pixel: i pixel di TESS sono grandi 21"), variabilità stellare
e artefatti strumentali.

Uso:
    pip install lightkurve astropy matplotlib
    python cerca_transiti_tess.py                 # solo le 90 gemelle solari
    python cerca_transiti_tess.py --tutte         # tutte le 3.224 stelle (molte ore)
    python cerca_transiti_tess.py --max 10        # prova veloce sulle prime 10

Si può interrompere (Ctrl+C) e rilanciare: le stelle già analizzate vengono saltate.

Output (in gaia_sky_out\\<zona>\\transiti):
    risultati_transiti.csv     una riga per stella analizzata
    candidati\\*.png            grafici dei segnali con SNR sopra soglia
    log_transiti_<data_ora>.txt
"""

import argparse
import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

CARTELLA_SCRIPT = Path(__file__).resolve().parent
AUTORI = ["SPOC", "TESS-SPOC", "QLP"]   # pipeline ufficiali con curve già pronte
RAGGIO_RICERCA_ARCSEC = 10
PERIODO_MIN, PERIODO_MAX = 0.5, 15.0    # giorni (un settore TESS dura ~27 giorni)
DURATE = [0.04, 0.06, 0.08, 0.1, 0.12, 0.16, 0.2, 0.25]  # giorni (~1 - 5 ore)
SOGLIA_SNR = 7.0
MIN_TRANSITI = 2
SOGLIA_BINARIA = 4.0     # sigma oltre cui dispari/pari o secondaria indicano una binaria

COLONNE = [
    "source_id", "gemella_solare", "ra", "dec", "dist_ly", "phot_g_mean_mag",
    "esito", "n_curve", "settori", "autori", "n_punti", "giorni_coperti",
    "periodo_d", "t0_btjd", "durata_h", "profondita_ppm", "snr", "n_transiti",
    "dispari_pari_sigma", "secondaria_sigma", "raggio_pianeta_terre_stimato", "candidato", "noto", "note",
]

log = logging.getLogger("transiti")
warnings.filterwarnings("ignore")


def configura_log(cartella: Path) -> Path:
    file_log = cartella / f"log_transiti_{datetime.now():%Y%m%d_%H%M%S}.txt"
    formato = logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(file_log, encoding="utf-8")):
        h.setFormatter(formato)
        log.addHandler(h)
    log.setLevel(logging.INFO)
    return file_log


def stelle_note(cartella_zona: Path) -> set:
    """source_id di stelle con pianeti/candidati già noti (da esopianeti_match.csv)."""
    f = cartella_zona / "esopianeti_match.csv"
    if not f.exists():
        log.info("esopianeti_match.csv non trovato: nessun filtro sui sistemi noti.")
        return set()
    from astropy.table import Table
    try:
        t = Table.read(f, format="ascii.csv")
        ids = {int(x) for x in t["gaia_source_id"]}
        log.info("Stelle con pianeti/candidati già noti: %d", len(ids))
        return ids
    except Exception:
        return set()


def gia_analizzate(file_csv: Path) -> set:
    if not file_csv.exists():
        return set()
    from astropy.table import Table
    try:
        t = Table.read(file_csv, format="ascii.csv")
        return {int(x) for x in t["source_id"]}
    except Exception:
        return set()


def aggiungi_riga(file_csv: Path, riga: dict):
    import csv
    nuovo = not file_csv.exists()
    with open(file_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLONNE)
        if nuovo:
            w.writeheader()
        w.writerow({k: riga.get(k, "") for k in COLONNE})


# ------------------------------------------------------------------ analisi
def scarica_curva(ra: float, dec: float):
    """Cerca e scarica le curve TESS della stella. Restituisce (lc, info) o (None, motivo)."""
    import astropy.units as u
    import lightkurve as lk
    from astropy.coordinates import SkyCoord

    coord = SkyCoord(ra * u.deg, dec * u.deg)
    ricerca = lk.search_lightcurve(coord, radius=RAGGIO_RICERCA_ARCSEC * u.arcsec,
                                   mission="TESS", author=AUTORI)
    if len(ricerca) == 0:
        return None, "nessuna curva TESS"

    # Se ci sono più bersagli entro il raggio, tiene il più vicino
    if "distance" in ricerca.table.colnames:
        dmin = min(ricerca.table["distance"])
        ricerca = ricerca[ricerca.table["distance"] <= dmin + 0.5]

    collezione = ricerca.download_all(quality_bitmask="default")
    if collezione is None or len(collezione) == 0:
        return None, "download fallito"

    pezzi, settori, autori = [], set(), set()
    for lc in collezione:
        try:
            lc = lc.remove_nans().normalize()
            if len(lc) < 100:
                continue
            pezzi.append(lc)
            settori.add(int(lc.meta.get("SECTOR", -1)))
            autori.add(str(lc.meta.get("AUTHOR", lc.meta.get("ORIGIN", "?"))))
        except Exception:
            continue
    if not pezzi:
        return None, "curve vuote"

    lc = lk.LightCurveCollection(pezzi).stitch()
    info = {
        "n_curve": len(pezzi),
        "settori": " ".join(str(s) for s in sorted(settori)),
        "autori": " ".join(sorted(autori)),
    }
    return lc, info


def analizza(lc):
    """Appiattisce la curva, esegue BLS e calcola le metriche del segnale migliore."""
    import numpy as np

    lc = lc.remove_nans()
    dt = float(np.nanmedian(np.diff(lc.time.value)))
    # finestra di ~0.75 giorni: toglie la variabilità lenta ma non i transiti (< 5 ore)
    finestra = max(int(0.75 / dt) | 1, 11)
    piatta = lc.flatten(window_length=finestra).remove_outliers(sigma_upper=4, sigma_lower=20)
    # porta tutto a ~10 minuti per velocità e uniformità tra pipeline diverse
    if dt < 0.006:
        piatta = piatta.bin(time_bin_size=0.00694)
    piatta = piatta.remove_nans()

    t = piatta.time.value
    giorni = float(t.max() - t.min())
    pmax = min(PERIODO_MAX, giorni / MIN_TRANSITI)
    if pmax <= PERIODO_MIN:
        return piatta, None, {"giorni_coperti": round(giorni, 1),
                              "n_punti": len(piatta), "note": "copertura troppo breve"}

    periodi = np.exp(np.linspace(np.log(PERIODO_MIN), np.log(pmax), 30000))
    pg = piatta.to_periodogram(method="bls", period=periodi, duration=DURATE,
                               frequency_factor=500)
    P = float(pg.period_at_max_power.value)
    t0 = float(pg.transit_time_at_max_power.value)
    dur = float(pg.duration_at_max_power.value)
    prof = float(pg.depth_at_max_power.value)

    # Secondo passaggio: riappiattisce la curva originale ESCLUDENDO i transiti,
    # così il filtro non "mangia" parte della profondità del segnale.
    maschera_orig = lc.create_transit_mask(period=P, transit_time=t0, duration=dur * 1.5)
    piatta = lc.flatten(window_length=finestra, mask=maschera_orig).remove_outliers(
        sigma_upper=4, sigma_lower=20)
    if dt < 0.006:
        piatta = piatta.bin(time_bin_size=0.00694)
    piatta = piatta.remove_nans()
    t = piatta.time.value

    flux = piatta.flux.value
    maschera = piatta.create_transit_mask(period=P, transit_time=t0, duration=dur)
    # profondità misurata direttamente sui punti in transito
    if maschera.sum() > 0:
        prof = float(1 - np.nanmedian(flux[maschera]))
    fuori = flux[~maschera]
    sigma = float(np.nanstd(fuori))
    n_in = int(maschera.sum())
    snr = prof / sigma * np.sqrt(n_in) if sigma > 0 and n_in > 0 else 0.0

    # numero di transiti effettivamente coperti da dati
    epoche = np.round((t[maschera] - t0) / P)
    n_tr = len(np.unique(epoche))

    # controllo dispari/pari
    dp_sigma = None
    if n_tr >= 2:
        pari = maschera & (np.round((t - t0) / P) % 2 == 0)
        dispari = maschera & (np.round((t - t0) / P) % 2 == 1)
        if pari.sum() > 3 and dispari.sum() > 3:
            d_p = 1 - np.nanmedian(flux[pari])
            d_d = 1 - np.nanmedian(flux[dispari])
            err = 1.25 * sigma * np.sqrt(1 / pari.sum() + 1 / dispari.sum())  # 1.25: errore della mediana
            dp_sigma = round(float(abs(d_p - d_d) / err), 1) if err > 0 else None

    # controllo eclisse secondaria: un calo a metà periodo indica una binaria
    sec_sigma = None
    sec = piatta.create_transit_mask(period=P, transit_time=t0 + P / 2, duration=dur)
    if sec.sum() > 3:
        d_sec = 1 - np.nanmedian(flux[sec])
        err_sec = sigma / np.sqrt(sec.sum())
        sec_sigma = round(float(d_sec / err_sec), 1) if err_sec > 0 else None

    return piatta, pg, {
        "secondaria_sigma": sec_sigma,
        "n_punti": len(piatta),
        "giorni_coperti": round(giorni, 1),
        "periodo_d": round(P, 5),
        "t0_btjd": round(t0, 4),
        "durata_h": round(dur * 24, 2),
        "profondita_ppm": round(prof * 1e6, 0),
        "snr": round(float(snr), 1),
        "n_transiti": n_tr,
        "dispari_pari_sigma": dp_sigma,
    }


def grafico(piatta, pg, m: dict, titolo: str, file_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    piatta.scatter(ax=ax[0], s=1, color="k")
    ax[0].set_title(titolo)
    pg.plot(ax=ax[1], color="k", lw=0.6)
    ax[1].axvline(m["periodo_d"], color="r", alpha=0.4, lw=3)
    fold = piatta.fold(period=m["periodo_d"], epoch_time=m["t0_btjd"])
    fold.scatter(ax=ax[2], s=2, color="k", alpha=0.4, label="dati")
    # media a blocchi calcolata a mano (più robusta di fold.bin su curve già binnate)
    import numpy as np
    fase = fold.time.value
    fl = fold.flux.value
    passo = m["durata_h"] / 24 / 5
    bordi = np.arange(fase.min(), fase.max() + passo, passo)
    idx = np.digitize(fase, bordi)
    cx = [fase[idx == k].mean() for k in np.unique(idx) if (idx == k).sum() > 2]
    cy = [np.median(fl[idx == k]) for k in np.unique(idx) if (idx == k).sum() > 2]
    ax[2].plot(cx, cy, "o", color="r", ms=4, label="media")
    ax[2].legend(loc="lower right")
    ax[2].set_xlim(-3 * m["durata_h"] / 24, 3 * m["durata_h"] / 24)
    ax[2].set_title(f"P={m['periodo_d']} d  prof={m['profondita_ppm']:.0f} ppm  "
                    f"SNR={m['snr']}  transiti={m['n_transiti']}  "
                    f"dispari/pari={m['dispari_pari_sigma']}  secondaria={m['secondaria_sigma']} sigma")
    fig.tight_layout()
    fig.savefig(file_png, dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser(description="Ricerca transiti TESS nel corridoio")
    p.add_argument("--zona", default="paper")
    p.add_argument("--cartella", default=str(CARTELLA_SCRIPT / "gaia_sky_out"))
    p.add_argument("--tutte", action="store_true",
                   help="analizza tutte le stelle G/K, non solo le gemelle")
    p.add_argument("--max", type=int, default=0, help="analizza al massimo N stelle")
    p.add_argument("--snr", type=float, default=SOGLIA_SNR, help="soglia SNR candidati")
    a = p.parse_args()

    zona = Path(a.cartella) / a.zona
    out = zona / "transiti"
    (out / "candidati").mkdir(parents=True, exist_ok=True)
    file_log = configura_log(out)
    log.info("Cartella: %s", out.resolve())
    log.info("File di log: %s", file_log.name)

    try:
        import lightkurve  # noqa: F401
    except ImportError:
        log.error("Manca lightkurve: pip install lightkurve astropy matplotlib")
        raise

    from astropy.table import Table
    file_stelle = zona / ("corridoio_gk.csv" if a.tutte else "gemelle_solari.csv")
    stelle = Table.read(file_stelle, format="ascii.csv")
    log.info("Stelle in %s: %d", file_stelle.name, len(stelle))

    file_ris = out / "risultati_transiti.csv"
    fatte = gia_analizzate(file_ris)
    note = stelle_note(zona)
    da_fare = [s for s in stelle if int(s["source_id"]) not in fatte]
    if a.max:
        da_fare = da_fare[:a.max]
    log.info("Già analizzate: %d. Da analizzare ora: %d.", len(fatte), len(da_fare))
    log.info("Soglia candidato: SNR >= %.1f, almeno %d transiti, dispari/pari e secondaria < %.0f sigma",
             a.snr, MIN_TRANSITI, SOGLIA_BINARIA)

    n_cand = 0
    t_inizio = time.time()
    for i, s in enumerate(da_fare, 1):
        sid = int(s["source_id"])
        base = {
            "source_id": sid, "gemella_solare": int(s["gemella_solare"]),
            "ra": round(float(s["ra"]), 5), "dec": round(float(s["dec"]), 5),
            "dist_ly": round(float(s["dist_ly"]), 1),
            "phot_g_mean_mag": round(float(s["phot_g_mean_mag"]), 2),
            "noto": int(sid in note),
        }
        trascorsi = time.time() - t_inizio
        stima = trascorsi / (i - 1) * (len(da_fare) - i + 1) / 60 if i > 1 else 0
        log.info("[%d/%d] Gaia DR3 %d (G=%.2f)%s  - restano ~%.0f min",
                 i, len(da_fare), sid, base["phot_g_mean_mag"],
                 "  [sistema già noto]" if base["noto"] else "", stima)
        try:
            lc, info = scarica_curva(base["ra"], base["dec"])
            if lc is None:
                log.info("   %s", info)
                aggiungi_riga(file_ris, {**base, "esito": info})
                continue
            log.info("   %d curve, settori %s (%s)", info["n_curve"], info["settori"],
                     info["autori"])
            piatta, pg, m = analizza(lc)
            riga = {**base, **info, **m, "esito": "analizzata"}
            if pg is None:
                log.info("   %s", m.get("note"))
                aggiungi_riga(file_ris, riga)
                continue

            # raggio stimato: R_p = R_star * sqrt(profondità); R_star ~ 1 R_sole per le gemelle
            r_stella = 1.0 if base["gemella_solare"] else 0.8
            if m["profondita_ppm"] > 0:
                riga["raggio_pianeta_terre_stimato"] = round(
                    109.1 * r_stella * (m["profondita_ppm"] / 1e6) ** 0.5, 1)

            ok_dp = m["dispari_pari_sigma"] is None or m["dispari_pari_sigma"] < SOGLIA_BINARIA
            ok_sec = m["secondaria_sigma"] is None or m["secondaria_sigma"] < SOGLIA_BINARIA
            cand = (m["snr"] >= a.snr and m["n_transiti"] >= MIN_TRANSITI
                    and ok_dp and ok_sec)
            riga["candidato"] = int(cand)
            log.info("   P=%.4f d  prof=%.0f ppm  durata=%.1f h  SNR=%.1f  transiti=%d  "
                     "dispari/pari=%s sigma  secondaria=%s sigma  R~%s Terre",
                     m["periodo_d"], m["profondita_ppm"], m["durata_h"], m["snr"],
                     m["n_transiti"], m["dispari_pari_sigma"], m["secondaria_sigma"],
                     riga.get("raggio_pianeta_terre_stimato"))
            if m["snr"] >= a.snr:
                nome = f"{'CAND_' if cand else 'scartato_'}{sid}.png"
                grafico(piatta, pg, m, f"Gaia DR3 {sid}", out / "candidati" / nome)
                if cand:
                    n_cand += 1
                    log.info("   >>> SEGNALE DA VERIFICARE (grafico: %s)", nome)
                elif not ok_dp:
                    log.info("   dispari/pari diversi: probabile binaria a eclisse")
                elif not ok_sec:
                    log.info("   calo anche a metà periodo: probabile binaria a eclisse")
            aggiungi_riga(file_ris, riga)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            log.warning("   errore: %s", e)
            aggiungi_riga(file_ris, {**base, "esito": f"errore: {e}"[:120]})

    log.info("=" * 60)
    log.info("FATTO in %.0f min. Segnali da verificare in questa esecuzione: %d",
             (time.time() - t_inizio) / 60, n_cand)
    log.info("Risultati: %s", file_ris.resolve())
    log.info("Ricorda: un segnale qui è un candidato, non una scoperta.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.warning("Interrotto dall'utente. Rilancia per riprendere da dove eri rimasto.")
    except Exception:
        log.exception("Errore:")
    finally:
        try:
            input("\nPremi Invio per chiudere...")
        except EOFError:
            pass
