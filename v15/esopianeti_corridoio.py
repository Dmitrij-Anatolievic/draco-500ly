#!/usr/bin/env python3
"""
esopianeti_corridoio.py
Incrocia le stelle del corridoio (corridoio_gk.csv, prodotto da gaia_sky_corridoio.py)
con il NASA Exoplanet Archive aggiornato a oggi:
  - pscomppars : pianeti confermati (parametri compositi)
  - toi        : candidati TESS (TESS Objects of Interest)
  - cumulative : candidati Kepler (KOI)

Uso:
    pip install astroquery astropy
    python esopianeti_corridoio.py                       # usa gaia_sky_out\\paper
    python esopianeti_corridoio.py --zona repo           # usa gaia_sky_out\\repo
    python esopianeti_corridoio.py --raggio 5            # raggio di match in arcsec

Output (nella stessa cartella del CSV del corridoio):
    esopianeti_match.csv     tutte le corrispondenze trovate
    log_esopianeti_<data_ora>.txt

Criteri "potenzialmente abitabile" (APPROSSIMATIVI, solo per segnalare candidati):
    raggio planetario <= 1.6 raggi terrestri
    irraggiamento tra 0.35 e 1.1 volte quello terrestre (stelle simili al Sole)
Non è una conferma di abitabilità.
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

CARTELLA_SCRIPT = Path(__file__).resolve().parent
RAGGIO_MAX_TERRA = 1.6
INSOL_MIN, INSOL_MAX = 0.35, 1.1

log = logging.getLogger("esopianeti")

# Tabelle dell'archivio NASA e colonne da leggere.
# Ogni voce: nome tabella -> (colonne, mappa verso nomi comuni)
TABELLE = {
    "pscomppars": {
        "descrizione": "pianeti confermati",
        "select": "pl_name,hostname,ra,dec,sy_dist,pl_rade,pl_insol,pl_eqt,"
                  "pl_orbper,discoverymethod,disc_facility",
        "nome": "pl_name",
        "stato": None,
        "rade": "pl_rade",
        "insol": "pl_insol",
        "eqt": "pl_eqt",
        "periodo": "pl_orbper",
    },
    "toi": {
        "descrizione": "candidati TESS (TOI)",
        "select": "toi,tid,tfopwg_disp,ra,dec,st_dist,pl_rade,pl_insol,pl_eqt,pl_orbper",
        "nome": "toi",
        "stato": "tfopwg_disp",
        "rade": "pl_rade",
        "insol": "pl_insol",
        "eqt": "pl_eqt",
        "periodo": "pl_orbper",
    },
    "cumulative": {
        "descrizione": "candidati Kepler (KOI)",
        "select": "kepoi_name,kepid,koi_disposition,ra,dec,koi_prad,koi_insol,"
                  "koi_teq,koi_period",
        "nome": "kepoi_name",
        "stato": "koi_disposition",
        "rade": "koi_prad",
        "insol": "koi_insol",
        "eqt": "koi_teq",
        "periodo": "koi_period",
    },
}


def configura_log(cartella: Path) -> Path:
    file_log = cartella / f"log_esopianeti_{datetime.now():%Y%m%d_%H%M%S}.txt"
    formato = logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s", "%H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formato)
    su_file = logging.FileHandler(file_log, encoding="utf-8")
    su_file.setFormatter(formato)
    log.setLevel(logging.INFO)
    log.addHandler(console)
    log.addHandler(su_file)
    return file_log


def leggi_corridoio(file_csv: Path):
    from astropy.table import Table
    if not file_csv.exists():
        raise FileNotFoundError(
            f"Non trovo {file_csv}. Lancia prima gaia_sky_corridoio.py."
        )
    tab = Table.read(file_csv, format="ascii.csv")
    log.info("Lette %d stelle dal corridoio (%s).", len(tab), file_csv.name)
    return tab


def scarica_tabella(nome: str, info: dict):
    """Scarica una tabella intera dall'archivio NASA (poche migliaia di righe)."""
    from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
    log.info("Scarico '%s' (%s)...", nome, info["descrizione"])
    t0 = time.time()
    tab = NasaExoplanetArchive.query_criteria(table=nome, select=info["select"])
    log.info("  ricevute %d righe in %.0f s.", len(tab), time.time() - t0)
    return tab


def valore(riga, colonna):
    """Legge un valore numerico gestendo celle vuote/mascherate e unità astropy."""
    if colonna is None or colonna not in riga.colnames:
        return None
    v = riga[colonna]
    try:
        v = getattr(v, "value", v)
        if v is None or (hasattr(v, "mask") and v.mask):
            return None
        v = float(v)
        return None if v != v else v  # scarta NaN
    except (TypeError, ValueError):
        return None


def incrocia(corridoio, archivio, info: dict, nome_tab: str, raggio_arcsec: float):
    import astropy.units as u
    import numpy as np
    from astropy.coordinates import SkyCoord

    def gradi(col):
        return np.asarray(getattr(col, "value", col), dtype=float)

    c_gaia = SkyCoord(gradi(corridoio["ra"]) * u.deg, gradi(corridoio["dec"]) * u.deg)
    ra_a, dec_a = gradi(archivio["ra"]), gradi(archivio["dec"])
    validi = np.isfinite(ra_a) & np.isfinite(dec_a)
    archivio = archivio[validi]
    c_arch = SkyCoord(ra_a[validi] * u.deg, dec_a[validi] * u.deg)

    # Per ogni oggetto dell'archivio cerca la stella del corridoio più vicina
    idx, sep, _ = c_arch.match_to_catalog_sky(c_gaia)
    ok = sep.arcsec <= raggio_arcsec
    log.info("  corrispondenze entro %.1f\": %d", raggio_arcsec, int(ok.sum()))

    risultati = []
    for i in np.where(ok)[0]:
        riga_a = archivio[i]
        stella = corridoio[idx[i]]
        rade = valore(riga_a, info["rade"])
        insol = valore(riga_a, info["insol"])
        abitabile = (
            rade is not None and insol is not None
            and rade <= RAGGIO_MAX_TERRA and INSOL_MIN <= insol <= INSOL_MAX
        )
        stato = str(riga_a[info["stato"]]) if info["stato"] else "CONFIRMED"
        risultati.append({
            "tabella": nome_tab,
            "oggetto": str(riga_a[info["nome"]]),
            "stato": stato,
            "gaia_source_id": int(stella["source_id"]),
            "gemella_solare": int(stella["gemella_solare"]),
            "sep_arcsec": round(float(sep.arcsec[i]), 2),
            "dist_ly_gaia": round(float(stella["dist_ly"]), 1),
            "raggio_terre": rade,
            "irraggiamento_terra": insol,
            "t_eq_K": valore(riga_a, info["eqt"]),
            "periodo_giorni": valore(riga_a, info["periodo"]),
            "potenz_abitabile": int(abitabile),
        })
    return risultati


def main():
    parser = argparse.ArgumentParser(description="Esopianeti nel corridoio Gaia")
    parser.add_argument("--zona", default="paper", help="sottocartella: paper o repo")
    parser.add_argument("--cartella", default=str(CARTELLA_SCRIPT / "gaia_sky_out"),
                        help="cartella principale di output di gaia_sky_corridoio.py")
    parser.add_argument("--raggio", type=float, default=5.0,
                        help="raggio di match in arcsec (default 5)")
    args = parser.parse_args()

    cartella = Path(args.cartella) / args.zona
    cartella.mkdir(parents=True, exist_ok=True)
    file_log = configura_log(cartella)
    log.info("Cartella: %s", cartella.resolve())
    log.info("File di log: %s", file_log.name)
    log.info("Raggio di match: %.1f arcsec (copre il moto proprio tra epoche diverse)",
             args.raggio)

    try:
        import astroquery  # noqa: F401
    except ImportError:
        log.error("Manca astroquery: pip install astroquery astropy")
        raise

    corridoio = leggi_corridoio(cartella / "corridoio_gk.csv")

    tutti = []
    for nome_tab, info in TABELLE.items():
        try:
            archivio = scarica_tabella(nome_tab, info)
            tutti.extend(incrocia(corridoio, archivio, info, nome_tab, args.raggio))
        except Exception as e:
            log.error("Tabella '%s' non disponibile: %s", nome_tab, e)

    from astropy.table import Table
    file_out = cartella / "esopianeti_match.csv"
    if tutti:
        Table(rows=tutti).write(file_out, format="ascii.csv", overwrite=True)
    else:
        file_out.write_text("nessuna corrispondenza\n", encoding="utf-8")

    # Riepilogo
    log.info("=" * 60)
    log.info("RIEPILOGO")
    for nome_tab, info in TABELLE.items():
        n = sum(1 for r in tutti if r["tabella"] == nome_tab)
        log.info("  %-11s (%s): %d", nome_tab, info["descrizione"], n)
    n_gem = sum(1 for r in tutti if r["gemella_solare"] == 1)
    log.info("  di cui attorno a gemelle solari: %d", n_gem)

    abitabili = [r for r in tutti if r["potenz_abitabile"] == 1]
    log.info("  potenzialmente abitabili (criteri approssimativi): %d", len(abitabili))
    for r in tutti:
        log.info("  %s %-18s %-12s Gaia DR3 %d  d=%.0f al  R=%s  S=%s  %s%s",
                 r["tabella"], r["oggetto"], r["stato"], r["gaia_source_id"],
                 r["dist_ly_gaia"], r["raggio_terre"], r["irraggiamento_terra"],
                 "GEMELLA " if r["gemella_solare"] else "",
                 "<- POTENZ. ABITABILE" if r["potenz_abitabile"] else "")
    log.info("FATTO. Risultati in: %s", file_out.resolve())


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
