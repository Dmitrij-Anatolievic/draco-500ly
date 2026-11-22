#!/usr/bin/env python3
"""
gaia_sky_corridoio.py
Scarica da Gaia DR3 le stelle G/K del "corridoio" del paper
"Draco at 500 Light-Years" e le salva in file CSV pronti per Gaia Sky.

Uso:
    pip install astroquery astropy matplotlib
    python gaia_sky_corridoio.py               # corridoio del paper (RA 260, Dec +25)
    python gaia_sky_corridoio.py --zona repo   # zona di GAIA_QUERY.pdf (l=84, b=2)

Output (nella cartella "gaia_sky_out\\<zona>", accanto allo script):
    corridoio_gk.csv        tutte le stelle G/K del corridoio
    gemelle_solari.csv      solo le gemelle solari
    corridoio_gk.vot        stesso contenuto in VOTable (con unità e UCD)
    mappa_corridoio.png     mappa RA/Dec di controllo
    log_<data_ora>.txt      log completo dell'esecuzione

In Gaia Sky: Ctrl+O -> "Load file" -> scegli il CSV -> tipo "Star catalog".
Carica i due CSV come dataset separati per dare un colore diverso alle gemelle.
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

ANNI_LUCE_PER_PC = 3.261563777
CARTELLA_SCRIPT = Path(__file__).resolve().parent
INTERVALLO_CONTROLLO_S = 10      # ogni quanti secondi controlla lo stato della query
TIMEOUT_QUERY_S = 30 * 60        # dopo 30 minuti rinuncia

log = logging.getLogger("corridoio")

# Zone disponibili. Parametri presi dal paper (sez. 8) e da GAIA_QUERY.pdf del repo.
ZONE = {
    # Paper v14, sez. 8: RA=260, Dec=+25, cono 15 gradi, 390-560 anni luce
    "paper": {
        "descrizione": "Paper v14: RA 260, Dec +25, cono 15 gradi",
        "tipo": "cono",
        "ra": 260.0,
        "dec": 25.0,
        "raggio": 15.0,
        "dmin_ly": 390.0,
        "dmax_ly": 560.0,
    },
    # GAIA_QUERY.pdf: l 76-92, b -6..10, 138-168 pc
    "repo": {
        "descrizione": "GAIA_QUERY.pdf: l 76-92, b -6..+10 (circa RA 310, Dec +45)",
        "tipo": "galattica",
        "lmin": 76.0,
        "lmax": 92.0,
        "bmin": -6.0,
        "bmax": 10.0,
        "dmin_ly": 138.0 * ANNI_LUCE_PER_PC,
        "dmax_ly": 168.0 * ANNI_LUCE_PER_PC,
    },
}

# Criteri del paper per tipo spettrale e gemelle solari
TEFF_GK = (3700, 6000)          # K: 3700-5200, G: 5200-6000 (GAIA_QUERY.pdf)
TEFF_SOLE, DTEFF = 5778, 200
LOGG_SOLE, DLOGG = 4.44, 0.2
MH_SOLE, DMH = 0.0, 0.1
MIN_PARALLAX_OVER_ERROR = 5     # come in GAIA_QUERY.pdf


def configura_log(cartella: Path) -> Path:
    """Log su console e su file nella cartella di output."""
    file_log = cartella / f"log_{datetime.now():%Y%m%d_%H%M%S}.txt"
    formato = logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s", "%H:%M:%S")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formato)
    su_file = logging.FileHandler(file_log, encoding="utf-8")
    su_file.setFormatter(formato)

    log.setLevel(logging.INFO)
    log.addHandler(console)
    log.addHandler(su_file)
    return file_log


def costruisci_query(zona: dict) -> str:
    """Restituisce la query ADQL. La distanza è data dall'inversione della parallasse."""
    plx_max = 1000.0 / (zona["dmin_ly"] / ANNI_LUCE_PER_PC)   # mas
    plx_min = 1000.0 / (zona["dmax_ly"] / ANNI_LUCE_PER_PC)   # mas

    if zona["tipo"] == "cono":
        filtro_pos = (
            "CONTAINS(POINT('ICRS', ra, dec), "
            f"CIRCLE('ICRS', {zona['ra']}, {zona['dec']}, {zona['raggio']})) = 1"
        )
    else:
        filtro_pos = (
            f"l BETWEEN {zona['lmin']} AND {zona['lmax']} "
            f"AND b BETWEEN {zona['bmin']} AND {zona['bmax']}"
        )

    return f"""
SELECT source_id, ra, dec, l, b, parallax, parallax_error, parallax_over_error,
       pmra, pmdec, radial_velocity,
       phot_g_mean_mag, bp_rp,
       teff_gspphot, logg_gspphot, mh_gspphot, distance_gspphot
FROM gaiadr3.gaia_source
WHERE {filtro_pos}
  AND parallax BETWEEN {plx_min:.4f} AND {plx_max:.4f}
  AND parallax_over_error > {MIN_PARALLAX_OVER_ERROR}
  AND teff_gspphot BETWEEN {TEFF_GK[0]} AND {TEFF_GK[1]}
"""


def scarica(query: str):
    """Invia la query all'archivio ESA in background e ne segue lo stato."""
    log.info("Importo astroquery...")
    try:
        from astroquery.gaia import Gaia
    except ImportError:
        log.error("Manca astroquery: pip install astroquery astropy matplotlib")
        raise

    Gaia.ROW_LIMIT = -1  # nessun limite di righe
    log.info("Invio la query all'archivio Gaia ESA...")
    t0 = time.time()
    job = Gaia.launch_job_async(query, background=True)
    log.info("Query accettata. ID job: %s", getattr(job, "jobid", "?"))

    fase_precedente = None
    while True:
        try:
            fase = job.get_phase(update=True)
        except Exception as e:  # rete instabile: riprova al giro dopo
            log.warning("Non riesco a leggere lo stato del job (%s), riprovo...", e)
            fase = fase_precedente

        trascorsi = time.time() - t0
        if fase != fase_precedente:
            log.info("Stato query: %s", fase)
            fase_precedente = fase
        else:
            log.info("...ancora %s (%.0f s trascorsi)", fase, trascorsi)

        if fase == "COMPLETED":
            break
        if fase in ("ERROR", "ABORTED"):
            raise RuntimeError(f"La query è terminata con stato {fase}")
        if trascorsi > TIMEOUT_QUERY_S:
            raise TimeoutError(
                f"Nessuna risposta dopo {TIMEOUT_QUERY_S // 60} minuti. "
                "L'archivio è probabilmente sovraccarico: riprova più tardi."
            )
        time.sleep(INTERVALLO_CONTROLLO_S)

    log.info("Query completata in %.0f s. Scarico i risultati...", time.time() - t0)
    tab = job.get_results()
    log.info("Ricevute %d righe.", len(tab))
    return tab


def prepara_tabella(tab):
    """Aggiunge le colonne che Gaia Sky riconosce (name) e il flag gemella."""
    import numpy as np

    tab["dist_ly"] = ANNI_LUCE_PER_PC * 1000.0 / tab["parallax"]
    tab["dist_ly"].unit = "lyr"

    teff = np.ma.filled(tab["teff_gspphot"].astype(float), np.nan)
    logg = np.ma.filled(tab["logg_gspphot"].astype(float), np.nan)
    mh = np.ma.filled(tab["mh_gspphot"].astype(float), np.nan)
    gemella = (
        (np.abs(teff - TEFF_SOLE) <= DTEFF)
        & (np.abs(logg - LOGG_SOLE) <= DLOGG)
        & (np.abs(mh - MH_SOLE) <= DMH)
    )
    tab["gemella_solare"] = gemella.astype(int)
    tab["tipo"] = np.where(teff >= 5200, "G", "K")

    # Nome visibile in Gaia Sky al passaggio del mouse
    tab["name"] = [
        f"Gaia DR3 {sid}" + (" (gemella)" if g else "")
        for sid, g in zip(tab["source_id"], gemella)
    ]
    return tab


def salva(tab, cartella: Path):
    log.info("Salvo il VOTable...")
    tab.write(cartella / "corridoio_gk.vot", format="votable", overwrite=True)

    # CSV: Gaia Sky ricava la distanza dalla colonna "parallax" (in mas)
    colonne_csv = [
        "name", "source_id", "ra", "dec", "parallax", "pmra", "pmdec",
        "radial_velocity", "phot_g_mean_mag", "bp_rp", "teff_gspphot",
        "logg_gspphot", "mh_gspphot", "dist_ly", "tipo", "gemella_solare",
    ]
    tab_csv = tab[colonne_csv]
    log.info("Salvo corridoio_gk.csv...")
    tab_csv.write(cartella / "corridoio_gk.csv", format="ascii.csv", overwrite=True)

    gemelle = tab_csv[tab_csv["gemella_solare"] == 1]
    log.info("Salvo gemelle_solari.csv...")
    gemelle.write(cartella / "gemelle_solari.csv", format="ascii.csv", overwrite=True)
    return gemelle


def mappa(tab, zona: dict, file_png: Path):
    log.info("Disegno la mappa...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(9, 7))
    norm = tab["gemella_solare"] == 0
    sc = ax.scatter(tab["ra"][norm], tab["dec"][norm], c=tab["dist_ly"][norm],
                    s=6, cmap="viridis", alpha=0.7, label="G/K")
    gem = tab["gemella_solare"] == 1
    ax.scatter(tab["ra"][gem], tab["dec"][gem], s=60, marker="*",
               color="gold", edgecolor="black", linewidth=0.5,
               label="gemelle solari")
    fig.colorbar(sc, ax=ax, label="distanza (anni luce)")
    ax.invert_xaxis()  # convenzione del cielo: RA cresce verso sinistra
    ax.set_xlabel("RA (gradi)")
    ax.set_ylabel("Dec (gradi)")
    ax.set_title(f"{zona['descrizione']}\n{len(tab)} stelle G/K, {int(np.sum(gem))} gemelle")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(file_png, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Corridoio Gaia DR3 per Gaia Sky")
    parser.add_argument("--zona", choices=ZONE.keys(), default="paper",
                        help="paper = RA 260/Dec +25 (default), repo = l 84/b 2")
    parser.add_argument("--out", default=str(CARTELLA_SCRIPT / "gaia_sky_out"),
                        help="cartella di output (default: accanto allo script)")
    parser.add_argument("--solo-query", action="store_true",
                        help="stampa la query ADQL senza eseguirla")
    args = parser.parse_args()

    zona = ZONE[args.zona]
    cartella = Path(args.out) / args.zona
    cartella.mkdir(parents=True, exist_ok=True)   # creata subito, non alla fine
    file_log = configura_log(cartella)

    log.info("Cartella di output: %s", cartella.resolve())
    log.info("File di log: %s", file_log.name)
    log.info("Zona: %s", zona["descrizione"])
    log.info("Distanza: %.0f-%.0f anni luce", zona["dmin_ly"], zona["dmax_ly"])
    query = costruisci_query(zona)
    log.info("Query ADQL:%s", query)
    if args.solo_query:
        return

    tab = prepara_tabella(scarica(query))
    gemelle = salva(tab, cartella)
    mappa(tab, zona, cartella / "mappa_corridoio.png")

    n_g = int((tab["tipo"] == "G").sum())
    log.info("Stelle G/K: %d (G: %d, K: %d)", len(tab), n_g, len(tab) - n_g)
    log.info("Gemelle solari: %d", len(gemelle))
    for riga in gemelle:
        log.info("  %s  G=%.2f  d=%.0f al",
                 riga["name"], riga["phot_g_mean_mag"], riga["dist_ly"])
    log.info("FATTO. File salvati in: %s", cartella.resolve())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.warning("Interrotto dall'utente.")
    except Exception:
        log.exception("Errore:")
    finally:
        # Con il doppio clic la finestra resterebbe chiusa: aspetta un Invio
        try:
            input("\nPremi Invio per chiudere...")
        except EOFError:
            pass
