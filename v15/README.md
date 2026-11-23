# v15 — corrected analysis / analisi corretta

Preprint v15: https://doi.org/10.5281/zenodo.24459203 (all versions: https://doi.org/10.5281/zenodo.19319994)

Everything in this folder reproduces the numbers of v15. Fixed random seeds; logs of every run in `gaia_sky_out/`.
Tutto in questa cartella riproduce i numeri della v15. Semi casuali fissi; i log di ogni esecuzione sono in `gaia_sky_out/`.

```bash
pip install astroquery astropy scipy numpy matplotlib pillow
cd v15
python blind_test_v2.py          # corrected blind test (16 nodes of v14)
python blind_test_v3.py          # 3 node sets x 2 random models, 600 directions x 13 distances
python blind_test_v3_draco.py    # declared hypothesis: Draco only
python blind_test_v3_controlli.py 100   # final controls: all 83 constellations, jitter robustness
python rileva_colori.py "../IMG_20171006_080030 copia.jpg"   # colour dots, criteria fixed in advance
python test_colori.py            # green and yellow dots
```

The Hipparcos cache (`gaia_sky_out/blind_test_v2/hipparcos_hp6.0.ecsv`, VizieR I/311/hip2, Hp ≤ 6) is included,
so the geometric tests run offline. `gaia_sky_corridoio.py`, `esopianeti_corridoio.py` and `cerca_transiti_tess.py`
need internet access (Gaia archive, NASA Exoplanet Archive, MAST).

| File | Content |
| --- | --- |
| `data/blue_nodes_15_rilevati.csv` (repo root) | the 15 blue dashes detected on the photo |
| `puntini_rilevati.csv` | green / yellow / orange / white dots detected automatically |
| `gaia_sky_out/blind_test_v3/riepilogo_v3.csv` | open question, 3 node sets x 2 random models |
| `gaia_sky_out/blind_test_v3/draco_ipotesi_dichiarata.csv` | Draco only, all distances and 400–600 ly |
| `gaia_sky_out/blind_test_v3/controllo_C_tutte.csv` | p for all 83 constellations |
| `gaia_sky_out/blind_test_v3/robustezza_jitter.csv` | 20 replicates with nodes moved by ±0.01 |
| `gaia_sky_out/blind_test_v3/null_draco_controlli.csv` | random-set distribution used for Draco |
| `gaia_sky_out/colori/risultati_colori.csv` | green and yellow dots |
| `rank_400_600.py`, `null_draco_controlli.py` | helper scripts for the 400–600 ly ranking and the Draco random-set distribution |

Decision rule fixed before the final controls: signal only if p(Draco) < 0.05, Draco 1st among all constellations
by p, and p < 0.05 in at least 16 of 20 jitter replicates. Result: 0.059, 6th of 83, 0 of 20 — no signal.
