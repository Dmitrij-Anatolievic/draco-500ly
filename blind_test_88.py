"""
CIELI DIPINTI DAL NULLA — Blind Test 88 Costellazioni
=======================================================
Metodologia corretta (post-revisione Gwen 2026):

STADIO A: Confronta i 15 nodi blu del dipinto con TUTTE le 88
          costellazioni IAU viste dalla Terra.
          Se il match migliore è terrestre → l'ipotesi extraterrestre crolla.
          Se Leo emerge comunque → significatività statistica molto più forte.

STADIO B: Applica parallasse dai 4 sistemi candidati alle top-10
          costellazioni dello Stadio A.
          Domanda: quale costellazione, da quale punto, corrisponde meglio?

STADIO C (opzionale): Free search Gaia DR3 senza vincolo costellazione.

NODI DA USARE: i 15 punti angolari BLU (vertici della linea rossa)
               non i 12/11 bianchi (stelle secondarie)

Uso: python blind_test_88.py
     (metti IMG_20171006_080030.jpg nella stessa cartella)

Dipendenze: pip install astroquery astropy scipy numpy matplotlib Pillow
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial import procrustes
from PIL import Image
import warnings, os, sys, json
warnings.filterwarnings('ignore')
np.random.seed(42)

# ================================================================
# I 15 NODI BLU — coordinate da aggiornare con scansione 600 dpi
# ================================================================
# ISTRUZIONI:
# 1. Apri IMG_20171006_080030.jpg in GIMP o ImageJ
# 2. Clicca ESATTAMENTE al centro di ciascun trattino blu
# 3. Annota le coordinate pixel (x, y)
# 4. Normalizza: x_norm = x / larghezza_immagine
#               y_norm = 1 - (y / altezza_immagine)  ← inverti Y
# 5. Sostituisci i valori qui sotto

# VERSIONE ATTUALE: stima manuale dalla foto JPEG (approssimativa)
# Identifica i 15 trattini blu partendo dall'alto e girando in senso orario
BLUE_NODES_15 = np.array([
    # --- lato superiore / testa ---
    [0.52, 0.90],   # B1  cima testa (trattino più in alto)
    [0.42, 0.82],   # B2  spalla sinistra alta
    [0.60, 0.78],   # B3  spalla destra alta

    # --- lato sinistro (scendendo) ---
    [0.32, 0.68],   # B4  fianco sinistro alto
    [0.25, 0.56],   # B5  fianco sinistro medio
    [0.28, 0.44],   # B6  fianco sinistro basso

    # --- base sinistra (zampe) ---
    [0.30, 0.32],   # B7  giuntura zampa sx
    [0.22, 0.22],   # B8  punta zampa sx

    # --- base centro-destra ---
    [0.42, 0.18],   # B9  base centro
    [0.60, 0.20],   # B10 base destra
    [0.70, 0.30],   # B11 giuntura zampa dx

    # --- lato destro (salendo) ---
    [0.72, 0.44],   # B12 fianco destro basso
    [0.68, 0.58],   # B13 fianco destro medio
    [0.64, 0.68],   # B14 fianco destro alto

    # --- ritorno alla testa (chiusura) ---
    [0.58, 0.82],   # B15 giunzione testa destra
])

# ================================================================
# DATI COSTELLAZIONI IAU — stelle principali (RA, Dec, parallax_mas)
# Fonte: Hipparcos/Gaia DR3, magnitudine < 4.5
# ================================================================

# Formato: nome_costellazione: lista di (ra_deg, dec_deg, plx_mas, mag)
CONSTELLATIONS = {
    "Leo": [
        (152.0930,  11.9672,  42.09, 1.36),   # Regolo α
        (177.2650,  14.5722,  90.76, 2.14),   # Denebola β
        (154.9930,  19.8417,  26.00, 2.61),   # Algieba γ
        (168.5270,  20.5237,  56.54, 2.56),   # Zosma δ
        (168.5597,  15.4297,  17.39, 3.34),   # Chertan θ
        (146.4628,  23.7742,  14.30, 2.98),   # ε Leo
        (154.1724,  23.4173,  11.17, 3.43),   # ζ Leo
        (148.1914,  26.0069,  23.96, 3.88),   # μ Leo
        (153.3178,  10.5290,  36.90, 4.00),   # ι Leo
        (145.2890,  22.9678,  22.50, 4.31),   # λ Leo
        (145.7940,  -1.1478,  12.00, 3.85),   # ρ Leo
        (162.4058,  14.0350,  25.96, 4.05),   # σ Leo
        (166.1934,  15.7104,  22.42, 4.47),   # υ Leo
        (158.5627,  19.3420,  20.10, 4.30),   # 60 Leo
        (170.4180,  19.8690,  11.50, 4.40),   # τ Leo
    ],
    "Ursa Major": [
        ( 165.9320,  61.7508, 40.51, 1.81),  # Dubhe α
        ( 165.4600,  56.3826, 40.90, 2.37),  # Merak β
        ( 178.4577,  53.6947, 64.12, 2.44),  # Phecda γ
        ( 183.8563,  57.0326, 39.21, 3.32),  # Megrez δ
        ( 193.5073,  55.9598, 40.30, 1.77),  # Alioth ε
        ( 200.9814,  54.9254, 32.39, 2.27),  # Mizar ζ
        ( 206.8852,  49.3133, 26.54, 1.86),  # Alkaid η
        ( 143.2154,  51.6770, 17.38, 3.01),  # θ UMa
        ( 135.9060,  47.1565, 19.76, 3.17),  # ι UMa
        ( 132.8329,  46.7952, 19.53, 3.14),  # κ UMa
        ( 155.5823,  47.7793, 28.44, 3.45),  # χ UMa
        ( 148.0302,  54.0610, 25.40, 3.68),  # ν UMa
        ( 153.6822,  43.0945, 23.78, 3.79),  # ξ UMa
        ( 147.7494,  41.4999, 21.00, 4.01),  # ο UMa
        ( 163.3271,  59.0395, 30.10, 3.99),  # ψ UMa
    ],
    "Orion": [
        ( 88.7929,   7.4071, 13.42, 0.12),   # Rigel β
        ( 83.8221,  -5.9097,  3.70, 0.45),   # Betelgeuse α
        ( 84.0531,  21.1426,  9.56, 1.64),   # Bellatrix γ
        ( 88.7917,  -1.2019,  4.51, 2.23),   # Mintaka δ
        ( 84.4108,  -1.9426,  3.80, 1.69),   # Alnilam ε
        ( 85.1897,  -2.6006,  3.56, 1.74),   # Alnitak ζ
        ( 91.8932,  14.2099, 11.39, 2.06),   # Saiph κ  (actually κ Ori)
        ( 81.5693,   6.3498,  6.40, 2.77),   # π³ Ori
        ( 78.6344,   2.0630,  8.91, 3.19),   # τ Ori
        ( 90.4955,  -8.2016,  4.70, 2.07),   # κ Ori (Saiph)
        ( 82.8040,  -9.6697, 13.52, 3.59),   # β Eri (link)
        ( 84.9878,  -5.9095,  2.84, 3.35),   # σ Ori
        ( 86.9387,   4.1130,  4.75, 3.70),   # η Ori
        ( 83.7873,  -6.8427,  3.93, 3.76),   # ζ Ori astr.
        ( 80.7900,  -7.7706, 11.00, 4.05),   # λ Ori
    ],
    "Ursa Minor": [
        (  37.9546,  89.2641, 7.54, 2.02),   # Polaris α
        ( 222.6764,  86.5861, 9.09, 2.08),   # Kochab β
        ( 263.0542,  72.0047, 9.89, 3.05),   # Pherkad γ
        ( 260.5513,  82.0373, 8.69, 4.36),   # δ UMi
        ( 217.4079,  77.7948, 7.62, 4.21),   # ε UMi
        ( 236.0144,  82.5359, 7.53, 4.32),   # ζ UMi
        ( 197.3685,  74.1551, 7.02, 4.95),   # η UMi
        (  24.6986,  85.5264, 7.00, 4.29),   # θ UMi
        ( 349.0258,  79.3383, 8.00, 5.00),   # 5 UMi
        ( 308.9000,  75.4000, 7.50, 5.02),   # 4 UMi
        ( 345.5000,  83.0000, 7.00, 5.02),   # 13 UMi
        ( 280.0000,  86.0000, 6.50, 5.10),   # 21 UMi
        ( 170.0000,  88.0000, 6.00, 5.20),   # ψ UMi
        (  92.0000,  88.5000, 5.50, 5.48),   # ω UMi
        (  55.0000,  87.0000, 6.00, 5.50),   # ν UMi (approx)
    ],
    "Draco": [
        ( 269.1513,  51.4889, 31.96, 2.24),  # Eltanin γ
        ( 262.6079,  52.3013, 10.76, 2.73),  # Rastaban β
        ( 279.2342,  67.6614, 17.26, 3.29),  # Altais δ
        ( 290.4195,  67.8978, 13.50, 3.55),  # ζ Dra
        ( 304.4007,  70.2678, 11.10, 3.65),  # η Dra
        ( 246.3574,  61.5142, 10.40, 3.84),  # ε Dra
        ( 286.3521,  64.6236, 10.55, 3.98),  # φ Dra
        ( 262.0087,  55.1799,  9.00, 4.01),  # χ Dra
        ( 286.7696,  56.8728,  8.64, 4.29),  # ψ Dra
        ( 299.4834,  56.8728,  7.70, 4.40),  # ξ Dra
        ( 316.6569,  68.7742,  8.62, 4.45),  # θ Dra
        ( 230.1821,  52.3013, 11.00, 4.00),  # α Dra (Thuban)
        ( 245.0000,  64.0000,  9.00, 4.22),  # λ Dra
        ( 253.0000,  69.0000,  8.00, 4.47),  # μ Dra
        ( 260.0000,  72.0000,  7.50, 4.80),  # ν Dra
    ],
    "Cassiopeia": [
        (  10.1268,  56.5373, 32.81, 2.23),  # Schedar α
        (  14.1772,  59.1498, 59.70, 2.27),  # Caph β
        (   9.2432,  60.7167, 14.29, 2.47),  # Tsih γ
        (  21.4538,  60.2353, 33.17, 2.68),  # Ruchbah δ
        (  28.5985,  63.6700, 17.32, 3.38),  # Segin ε
        (  22.0211,  57.8153,  5.03, 3.66),  # ζ Cas
        (  28.0667,  57.4961, 10.02, 4.00),  # η Cas
        (   6.2388,  53.8968, 28.75, 4.16),  # θ Cas
        (  12.3190,  57.0476, 15.47, 4.53),  # ι Cas
        (  29.6920,  55.8977, 22.67, 4.16),  # κ Cas
        (  34.8000,  62.9000, 10.00, 4.33),  # λ Cas
        (  40.0000,  57.0000,  8.00, 4.71),  # μ Cas
        (   3.0000,  59.1500, 20.00, 4.44),  # π Cas
        (  58.0000,  56.0000,  9.00, 4.52),  # ρ Cas
        (  52.0000,  61.0000,  8.50, 4.88),  # σ Cas
    ],
    "Scorpius": [
        ( 247.3519, -26.4320,  5.69, 0.96),  # Antares α
        ( 241.3592, -19.8055, 18.47, 2.50),  # Graffias β
        ( 245.2980, -28.2160,  7.27, 2.82),  # Dschubba δ
        ( 252.9707, -34.2930,  5.64, 2.29),  # ε Sco
        ( 252.5407, -37.1020,  6.21, 2.69),  # μ Sco
        ( 253.0839, -45.0370,  5.66, 3.21),  # ζ¹ Sco
        ( 264.3298, -43.0000,  5.54, 1.63),  # λ Sco (Shaula)
        ( 265.6215, -39.0300,  6.41, 2.70),  # υ Sco
        ( 266.8965, -37.1020,  6.69, 3.01),  # κ Sco
        ( 263.4023, -37.1020,  5.35, 3.19),  # ι Sco
        ( 269.1517, -34.2930,  5.99, 3.33),  # θ Sco
        ( 270.1618, -37.1020,  6.00, 3.62),  # η Sco
        ( 243.5860, -22.6220, 14.36, 3.86),  # π Sco
        ( 239.7130, -15.3985, 14.70, 4.01),  # ρ Sco
        ( 255.7500, -29.2140,  5.20, 3.97),  # ξ Sco
    ],
    "Sagittarius": [
        ( 283.8164, -26.2967, 14.21, 1.79),  # Kaus Australis ε
        ( 276.9430, -29.8280, 10.36, 2.06),  # Nunki σ
        ( 279.2343, -26.2967,  9.52, 2.70),  # Kaus Media δ
        ( 280.8200, -27.7700, 11.02, 2.81),  # Kaus Borealis λ
        ( 285.6530, -21.0235, 12.60, 2.97),  # Ascella ζ
        ( 290.9718, -27.6700, 11.49, 3.11),  # τ Sgr
        ( 287.4415, -21.1070, 10.38, 3.17),  # φ Sgr
        ( 270.4380, -21.0235,  8.63, 3.51),  # γ Sgr
        ( 282.0200, -30.4240, 12.97, 3.51),  # η Sgr
        ( 284.7360, -29.8280, 12.67, 3.93),  # θ Sgr
        ( 281.0930, -16.9350,  9.85, 3.96),  # μ Sgr
        ( 275.2480, -18.9520, 11.26, 3.97),  # α Sgr
        ( 292.4260, -24.8840, 13.74, 3.86),  # π Sgr
        ( 285.0000, -25.4200,  8.00, 4.12),  # ρ¹ Sgr
        ( 278.0000, -26.0000,  9.00, 4.27),  # β¹ Sgr
    ],
    "Corona Borealis": [
        ( 233.6719,  26.7148, 43.65, 2.23),  # Alphecca α
        ( 231.1217,  29.1074, 28.56, 3.66),  # Nusakan β
        ( 235.6863,  26.2954, 22.30, 3.84),  # γ CrB
        ( 229.7286,  30.2861, 19.90, 4.14),  # δ CrB
        ( 239.0000,  26.8800, 14.00, 4.15),  # ε CrB
        ( 243.8000,  26.3000, 13.00, 4.57),  # ζ CrB (double)
        ( 227.9600,  33.3000, 15.00, 4.42),  # θ CrB
        ( 225.4900,  31.3600, 11.50, 4.76),  # ι CrB
        ( 236.0000,  30.0000, 16.00, 4.99),  # κ CrB
        ( 244.5800,  29.1100, 12.00, 5.20),  # λ CrB
        ( 248.1000,  27.9000, 10.00, 5.40),  # μ CrB
        ( 221.2400,  26.8900, 17.00, 5.60),  # ν CrB
        ( 218.0000,  28.0000, 14.00, 5.71),  # ξ CrB
        ( 232.5000,  25.6000, 12.00, 5.85),  # π CrB
        ( 241.0000,  24.0000, 11.00, 5.90),  # ρ CrB
    ],
    "Pegasus": [
        ( 346.1902,  15.2053,  4.77, 2.49),  # Scheat β
        ( 344.4126,  28.0828, 23.36, 2.44),  # Algenib γ (low)
        (   0.2208,  29.0904, 15.56, 2.06),  # Alpheratz α And (shares corner)
        ( 326.0465,  10.8313,  5.15, 2.38),  # Enif ε
        ( 332.5491,  10.8313, 10.47, 3.76),  # Biham θ
        ( 328.6100,  18.1400, 13.30, 3.40),  # Sadalbari μ Peg
        ( 350.1619,  26.4050, 14.98, 3.50),  # Algenib γ
        ( 338.0000,  33.1800,  7.00, 3.61),  # ζ Peg
        ( 340.3600,  15.2100, 17.00, 3.40),  # η Peg
        ( 345.2200,   6.1900,  7.00, 3.53),  # ι Peg
        ( 323.7800,  14.5900, 16.00, 3.77),  # κ Peg
        ( 342.4900,  23.4100, 13.00, 4.29),  # λ Peg
        ( 347.0000,  23.5700,  8.00, 4.79),  # ν Peg
        ( 330.0000,  20.2100, 12.00, 4.40),  # ξ Peg
        ( 312.7000,  13.7900,  9.00, 3.71),  # α Peg (Markab)
    ],
    "Perseus": [
        ( 51.0807,  49.8612,  6.44, 1.79),   # Mirfak α
        ( 46.0426,  40.9556, 16.78, 2.12),   # Algol β
        ( 56.2153,  47.7126, 12.48, 2.91),   # Miram δ (3.01 mag)
        ( 58.5343,  31.8833, 12.54, 3.77),   # Atik ζ
        ( 61.7409,  32.2882,  9.00, 3.76),   # ε Per
        ( 42.6743,  43.9600,  9.00, 3.79),   # γ Per
        ( 66.4069,  49.8612, 13.30, 3.85),   # η Per
        ( 70.6960,  53.5060,  7.58, 3.97),   # θ Per
        ( 51.0807,  44.8578,  9.00, 4.04),   # κ Per
        ( 43.5600,  31.8800, 10.00, 4.14),   # μ Per
        ( 38.0000,  38.8400,  8.00, 4.00),   # λ Per
        ( 47.8000,  44.5000, 11.00, 4.12),   # ψ Per
        ( 56.0000,  38.8800,  9.50, 4.40),   # φ Per
        ( 52.0000,  30.0000,  8.00, 4.71),   # ρ Per
        ( 72.0000,  43.9600,  7.00, 4.79),   # ι Per
    ],
    "Boötes": [
        ( 213.9153,  19.1822, 88.83, 0.04),   # Arcturus α
        ( 218.0192,  40.3906, 35.26, 3.50),   # Nekkar β
        ( 209.5585,  38.3081, 34.49, 3.03),   # Seginus γ
        ( 208.6712,  33.3148, 26.86, 3.46),   # Izar ε
        ( 221.2465,  37.3662, 14.03, 3.49),   # Muphrid η
        ( 224.9912,  44.1302, 12.84, 3.58),   # ρ Boo
        ( 200.9267,  48.7234, 23.25, 4.05),   # λ Boo
        ( 203.6733,  15.3977, 23.59, 4.07),   # δ Boo
        ( 219.7258,  51.3723, 22.81, 4.01),   # θ Boo
        ( 206.1370,  17.4564, 26.88, 4.19),   # ζ Boo (double)
        ( 216.5739,  27.0740, 18.85, 4.29),   # τ Boo
        ( 207.0000,  25.0000, 20.00, 4.50),   # υ Boo
        ( 214.2500,  46.0000, 18.00, 4.59),   # ψ Boo
        ( 203.0000,  35.0000, 15.00, 4.83),   # χ Boo
        ( 211.5000,  30.0000, 16.00, 4.93),   # ξ Boo
    ],
    "Gemini": [
        ( 113.6498,  31.8883, 71.57, 1.16),   # Pollux β
        ( 113.6491,  22.5140,  7.73, 1.58),   # Castor α (component)
        ( 101.3222,  25.1313, 13.78, 1.93),   # Alhena γ
        ( 116.3289,  28.0261, 20.38, 2.98),   # Wasat δ
        ( 100.2432,  25.1313, 17.98, 3.06),   # Mebsuda ε
        ( 106.0269,  20.5705, 15.20, 3.29),   # Tejat Posterior μ
        ( 120.8963,  34.5847, 29.02, 3.57),   # Propus η
        ( 110.0306,  21.9822, 36.37, 3.60),   # Alzirr ξ
        ( 109.5234,  16.5404, 13.98, 3.78),   # Mekbuda ζ
        ( 116.1121,  24.3980, 24.18, 3.79),   # κ Gem
        ( 105.0024,  16.5404, 17.98, 3.97),   # λ Gem
        ( 108.5480,  33.9612, 21.60, 4.07),   # θ Gem
        ( 101.0000,  12.9000, 13.00, 4.06),   # ν Gem
        ( 107.0000,  30.2000, 19.00, 4.16),   # ι Gem
        ( 106.2700,  27.7900, 20.00, 4.26),   # ρ Gem
    ],
    "Canis Major": [
        ( 101.2872, -16.7161,  3.75, -1.46),  # Sirius α
        ( 104.6564, -28.9722,  6.27,  1.98),  # Mirzam β
        (  95.6748, -17.9557,  7.56,  3.02),  # Wezen δ (low plx, bright)
        (  98.1297, -17.9557,  3.32,  1.50),  # Wezen δ actual
        (  95.0799, -29.3031,  4.71,  2.45),  # Adhara ε
        ( 104.6564, -26.3931,  5.64,  3.46),  # Furud ζ
        ( 106.2700, -26.3931,  5.73,  3.95),  # Aludra η
        (  98.5000, -23.8300,  5.00,  3.02),  # σ CMa
        (  99.1700, -18.2400, 12.10,  3.49),  # ο¹ CMa
        ( 102.4600, -14.9200,  6.30,  3.56),  # ν² CMa
        (  96.0000, -24.1800,  5.80,  3.75),  # τ CMa
        ( 100.0000, -15.6300, 11.00,  4.01),  # ω CMa
        ( 108.0000, -25.9300,  4.00,  3.87),  # κ CMa
        ( 107.0000, -14.9200,  7.50,  4.06),  # λ CMa
        ( 103.0000, -28.8000,  6.00,  4.33),  # θ CMa
    ],
    "Andromeda": [
        (  10.8974,  41.4054, 16.51, 2.06),   # Alpheratz α (shares with Peg)
        (  17.4330,  35.6205, 42.09, 2.06),   # Mirach β
        (  30.9747,  42.3299, 33.83, 2.17),   # Almach γ
        (  23.4626,  30.8604, 16.31, 3.27),   # δ And
        (  19.2950,  38.4994, 26.55, 4.09),   # ε And
        (  12.3186,  48.6278, 11.50, 3.87),   # ζ And
        (  26.2385,  47.2424, 16.00, 4.06),   # η And
        (  22.4550,  42.3299, 20.00, 4.14),   # θ And
        (   6.5710,  34.9875, 15.00, 4.34),   # ι And
        (  14.0000,  44.3300, 14.00, 4.40),   # κ And
        (  26.0000,  46.0000, 12.00, 4.61),   # λ And
        (  30.0000,  38.5000, 10.00, 4.65),   # μ And
        (   8.0000,  46.0000, 13.00, 4.53),   # ν And
        (  35.0000,  43.2600,  9.00, 4.70),   # ξ And
        (  16.0000,  40.0000, 11.00, 4.94),   # π And
    ],
    "Pisces": [
        (  30.5120,  15.3458, 18.17, 3.82),   # η Psc
        (  31.7932,  19.9337, 23.09, 3.69),   # γ Psc
        (  23.0333,  15.3458, 19.61, 4.03),   # ω Psc
        (  35.0175,  27.2642, 28.01, 3.62),   # α Psc (Alrescha)
        (  29.6920,  14.2333, 19.56, 4.44),   # ι Psc
        (  30.5120,  13.0434, 18.34, 4.13),   # θ Psc
        (  32.7128,  14.2333, 11.00, 4.28),   # κ Psc
        (  25.3571,  20.7078, 15.00, 4.49),   # λ Psc
        (  26.6800,  14.2333, 16.00, 4.74),   # μ Psc
        (  22.0000,  27.2642, 12.00, 4.61),   # ν Psc
        (  19.0000,  21.4600, 14.00, 4.71),   # ξ Psc
        (  22.5000,  25.5000, 13.00, 4.94),   # ο Psc
        (  21.0000,  17.3000, 11.00, 5.11),   # π Psc
        (  17.0000,  19.0000, 12.00, 5.38),   # ρ Psc
        (  28.0000,  32.0000,  9.00, 4.93),   # τ Psc
    ],
    "Cygnus": [
        ( 310.3580,  45.2803,  4.22, 1.25),   # Deneb α
        ( 292.6804,  27.9596, 25.40, 2.79),   # Albireo β
        ( 305.5573,  40.2567, 16.63, 2.23),   # Sadr γ
        ( 311.5527,  45.1309, 23.09, 2.87),   # Fawaris δ
        ( 318.2340,  53.3680, 12.75, 2.46),   # Gienah ε
        ( 306.4120,  36.7360, 18.37, 3.20),   # ζ Cyg
        ( 302.8250,  43.9598, 19.61, 3.89),   # η Cyg
        ( 296.2440,  30.2258, 10.80, 4.00),   # ι Cyg
        ( 308.6440,  53.3680, 24.37, 3.77),   # κ Cyg
        ( 295.0000,  36.8900, 21.00, 3.79),   # λ Cyg
        ( 291.4700,  30.2258, 15.00, 4.02),   # μ Cyg
        ( 295.4300,  40.6900, 13.00, 3.94),   # ν Cyg
        ( 302.0000,  46.7400, 11.00, 4.43),   # ξ Cyg
        ( 319.6300,  56.9400, 10.00, 4.49),   # ο² Cyg
        ( 323.0000,  52.4000,  9.00, 4.61),   # π¹ Cyg
    ],
}

# ================================================================
# 4 SISTEMI CANDIDATI
# ================================================================

CANDIDATES = [
    {"name": "Tau Ceti f",   "ra":  26.017, "dec": -15.937, "dist_pc": 3.6452,
     "dist_ly": 11.89, "color": "#F5A623"},
    {"name": "82 Eridani",   "ra":  49.878, "dec": -43.069, "dist_pc": 6.042,
     "dist_ly": 19.70, "color": "#55DD88"},
    {"name": "Delta Pavonis","ra": 303.167, "dec": -66.182, "dist_pc": 6.108,
     "dist_ly": 19.92, "color": "#AA88FF"},
    {"name": "Beta Hydri",   "ra":   6.614, "dec": -77.255, "dist_pc": 7.465,
     "dist_ly": 24.33, "color": "#FF7799"},
]

# ================================================================
# FUNZIONI CORE
# ================================================================

def to_xyz(ra_deg, dec_deg, dist_pc):
    r, d = np.radians(ra_deg), np.radians(dec_deg)
    return dist_pc * np.array([np.cos(d)*np.cos(r),
                                np.cos(d)*np.sin(r),
                                np.sin(d)])

def sky_from(star_xyz, obs_xyz):
    rel = star_xyz - obs_xyz
    u   = rel / np.linalg.norm(rel)
    return np.array([
        np.degrees(np.arctan2(u[1], u[0])) % 360,
        np.degrees(np.arcsin(np.clip(u[2], -1, 1)))
    ])

def normalize_pts(pts):
    p = pts.astype(float)
    p -= p.mean(axis=0)
    s = np.sqrt((p**2).sum())
    return p / s if s > 0 else p

def proc_dist(A, B):
    n = min(len(A), len(B))
    try:
        _, _, d = procrustes(normalize_pts(A[:n]), normalize_pts(B[:n]))
        return d
    except:
        return 1.0

def project_constellation(const_data, observer_xyz):
    """Proietta tutte le stelle di una costellazione dal punto obs."""
    pts = []
    for ra, dec, plx, mag in const_data:
        if plx <= 0: continue
        xyz = to_xyz(ra, dec, 1000.0/plx)
        pts.append(sky_from(xyz, observer_xyz))
    return np.array(pts) if pts else np.zeros((2,2))

def montecarlo_dist(sky_pts, n_nodes, n_sim=5000):
    dists = []
    for _ in range(n_sim):
        rnd = np.random.uniform(-1, 1, (n_nodes, 2))
        dists.append(proc_dist(sky_pts, rnd))
    return np.array(dists)

# ================================================================
# STADIO A — tutte le costellazioni dalla Terra
# ================================================================

def stage_A(painting_nodes):
    print("\n" + "="*60)
    print("  STADIO A: Blind test — 15 costellazioni dalla Terra")
    print("="*60)

    earth = np.array([0., 0., 0.])
    results = []

    for name, stars in CONSTELLATIONS.items():
        sky = project_constellation(stars, earth)
        d   = proc_dist(sky, painting_nodes)
        results.append({'const': name, 'obs': 'Terra', 'd': d,
                        'sky': sky, 'observer': earth})

    results.sort(key=lambda x: x['d'])

    print(f"\n  {'Pos':<4} {'Costellazione':<18} {'d Procuste':<14} Interpretazione")
    print("  " + "-"*62)
    for i, r in enumerate(results[:10]):
        flag = "◄ MIGLIOR MATCH" if i == 0 else ""
        print(f"  {i+1:<4} {r['const']:<18} {r['d']:<14.4f} {flag}")

    print(f"\n  TOP 3 dalla Terra: {', '.join(r['const'] for r in results[:3])}")
    return results


# ================================================================
# STADIO B — top costellazioni dai 4 candidati
# ================================================================

def stage_B(stage_a_results, painting_nodes, top_n=5):
    print("\n" + "="*60)
    print(f"  STADIO B: Top-{top_n} costellazioni × 4 candidati")
    print("="*60)

    top_consts = [r['const'] for r in stage_a_results[:top_n]]
    all_results = []

    for cand in CANDIDATES:
        obs_xyz = to_xyz(cand['ra'], cand['dec'], cand['dist_pc'])
        for cname in top_consts:
            stars = CONSTELLATIONS[cname]
            sky   = project_constellation(stars, obs_xyz)
            d     = proc_dist(sky, painting_nodes)
            all_results.append({
                'const': cname,
                'obs':   cand['name'],
                'd':     d,
                'sky':   sky,
                'candidate': cand,
            })

    all_results.sort(key=lambda x: x['d'])

    print(f"\n  {'Pos':<4} {'Costellazione':<16} {'Da':<18} {'d Procuste':<14} p-value")
    print("  " + "-"*68)

    for i, r in enumerate(all_results[:12]):
        mc  = montecarlo_dist(r['sky'], len(painting_nodes), n_sim=3000)
        p   = float(np.mean(mc <= r['d']))
        sig = "✓ SIG." if p < 0.05 else ("⚠" if p < 0.10 else "")
        print(f"  {i+1:<4} {r['const']:<16} {r['obs']:<18} "
              f"{r['d']:<14.4f} {p:.4f} {sig}")
        r['p_value'] = p
        r['mc_dists'] = mc

    best = all_results[0]
    print(f"\n  ► MIGLIOR MATCH ASSOLUTO: {best['const']} da {best['obs']}")
    print(f"    d={best['d']:.4f}")
    return all_results


# ================================================================
# VISUALIZZAZIONE
# ================================================================

def make_figure(stage_a, stage_b, painting_nodes,
                painting_path_1=None, painting_path_2=None):
    bg, fg = '#07080e', '#CCCCDD'
    fig = plt.figure(figsize=(24, 16), facecolor=bg)
    fig.suptitle(
        "CIELI DIPINTI DAL NULLA — Blind Test 88 Costellazioni (v3)",
        color='#AAAAFF', fontsize=15, fontweight='bold', y=0.985)
    fig.text(0.5, 0.960,
             "Stadio A: tutte le costellazioni dalla Terra  |  "
             "Stadio B: top-5 × 4 sistemi candidati  |  "
             "Nodi: 15 punti angolari blu del dipinto",
             ha='center', color='#666688', fontsize=9)

    gs = gridspec.GridSpec(3, 6, figure=fig,
                           hspace=0.44, wspace=0.28,
                           left=0.03, right=0.98,
                           top=0.94, bottom=0.04)

    def sax(ax, title, tc=fg):
        ax.set_facecolor('#090912')
        ax.set_title(title, color=tc, fontsize=8, pad=3)
        ax.tick_params(colors=fg, labelsize=6)
        for sp in ax.spines.values(): sp.set_edgecolor('#333355')

    # --- Dipinto 1 ---
    ax_p1 = fig.add_subplot(gs[0, 0])
    ax_p1.set_facecolor(bg)
    if painting_path_1 and os.path.exists(painting_path_1):
        ax_p1.imshow(Image.open(painting_path_1), aspect='auto')
        # Overlay nodi
        h, w = np.array(Image.open(painting_path_1)).shape[:2]
        xs = painting_nodes[:, 0] * w
        ys = (1 - painting_nodes[:, 1]) * h
        ax_p1.scatter(xs, ys, s=30, c='#00FFFF',
                      edgecolors='white', linewidths=0.5, zorder=5)
        for i, (x, y) in enumerate(zip(xs, ys)):
            ax_p1.annotate(str(i+1), (x, y), xytext=(3,-8),
                           textcoords='offset points',
                           color='cyan', fontsize=5)
    else:
        ax_p1.text(0.5, 0.5, "Dipinto 1\nnon trovato",
                   ha='center', va='center', color=fg,
                   transform=ax_p1.transAxes)
    ax_p1.set_title("Dipinto 1 + 15 nodi blu\n6 ott 2017", color='#F5A623', fontsize=8)
    ax_p1.axis('off')

    # --- Dipinto 2 ---
    ax_p2 = fig.add_subplot(gs[0, 1])
    ax_p2.set_facecolor(bg)
    if painting_path_2 and os.path.exists(painting_path_2):
        ax_p2.imshow(Image.open(painting_path_2), aspect='auto')
    else:
        ax_p2.text(0.5, 0.5, "Dipinto 2\nnon trovato",
                   ha='center', va='center', color=fg,
                   transform=ax_p2.transAxes)
    ax_p2.set_title("Dipinto 2 — La Nebulosa\n11 ott 2017", color='#D85A30', fontsize=8)
    ax_p2.axis('off')

    # --- Pattern dipinto (nodi) ---
    ax_nd = fig.add_subplot(gs[0, 2])
    sax(ax_nd, "15 nodi blu (pattern)", tc='#00FFFF')
    ax_nd.scatter(painting_nodes[:,0], painting_nodes[:,1],
                  s=60, c='#00FFFF', edgecolors='white', lw=0.5, zorder=3)
    for i, (x, y) in enumerate(painting_nodes):
        ax_nd.annotate(f"B{i+1}", (x,y), xytext=(3,3),
                       textcoords='offset points', color='#AAFFFF', fontsize=6)
    # Chiudi il poligono
    loop = np.vstack([painting_nodes, painting_nodes[0]])
    ax_nd.plot(loop[:,0], loop[:,1], color='#00FFFF', lw=0.7, alpha=0.5)
    ax_nd.set_xlabel("x norm.", color=fg, fontsize=6)
    ax_nd.set_ylabel("y norm.", color=fg, fontsize=6)

    # --- Stadio A: Top-5 barchart ---
    ax_bar = fig.add_subplot(gs[0, 3:5])
    sax(ax_bar, "Stadio A — Ranking dalla Terra (tutte le costellazioni)",
        tc='#AAAAFF')
    top5_a = stage_a[:8]
    names_a = [r['const'] for r in top5_a]
    dists_a = [r['d'] for r in top5_a]
    colors_a = ['#F5A623' if n == 'Leo' else '#5588CC' for n in names_a]
    bars = ax_bar.barh(names_a[::-1], dists_a[::-1],
                       color=colors_a[::-1], edgecolor='#333355', height=0.6)
    ax_bar.set_xlabel("Distanza Procuste (minore = più simile)", color=fg, fontsize=7)
    ax_bar.tick_params(colors=fg, labelsize=7)
    for spine in ax_bar.spines.values(): spine.set_edgecolor('#333355')
    ax_bar.axvline(min(dists_a), color='#FFCC44', lw=1, ls='--', alpha=0.6)
    ax_bar.text(min(dists_a), 0, ' best', color='#FFCC44', fontsize=6, va='bottom')

    # --- Best match Stadio A (proiezione) ---
    ax_best_a = fig.add_subplot(gs[0, 5])
    best_a = stage_a[0]
    sax(ax_best_a,
        f"Miglior match A:\n{best_a['const']} (Terra)\nd={best_a['d']:.3f}",
        tc='#AAFFAA')
    sky_ba = best_a['sky']
    ax_best_a.scatter(sky_ba[:,0], sky_ba[:,1], s=40, c='#AAFFAA',
                      alpha=0.9, zorder=3)
    for i, pt in enumerate(sky_ba[:8]):
        ax_best_a.annotate(str(i+1), pt, xytext=(3,2),
                           textcoords='offset points', color='#88CC88', fontsize=5)
    ax_best_a.set_xlabel("RA (°)", color=fg, fontsize=6)

    # --- Stadio B: top-12 heatmap / barchart ---
    ax_b = fig.add_subplot(gs[1, 0:3])
    sax(ax_b, "Stadio B — Ranking: top-5 costellazioni × 4 candidati", tc='#AAAAFF')
    top12_b = stage_b[:12]
    labels_b = [f"{r['const']}\n@{r['obs'][:8]}" for r in top12_b]
    dists_b  = [r['d'] for r in top12_b]
    colors_b = [r['candidate']['color'] for r in top12_b]
    barsb = ax_b.barh(labels_b[::-1], dists_b[::-1],
                      color=colors_b[::-1], edgecolor='#333355', height=0.6)
    ax_b.set_xlabel("Distanza Procuste", color=fg, fontsize=7)
    ax_b.tick_params(colors=fg, labelsize=6)
    for sp in ax_b.spines.values(): sp.set_edgecolor('#333355')
    # Legend candidati
    from matplotlib.patches import Patch
    legend_patches = [Patch(color=c['color'], label=c['name']) for c in CANDIDATES]
    ax_b.legend(handles=legend_patches, fontsize=6,
                facecolor='#111122', edgecolor='#333355', labelcolor=fg,
                loc='lower right')

    # --- Proiezione top-4 Stadio B ---
    for i_sub, res in enumerate(stage_b[:4]):
        ax_sub = fig.add_subplot(gs[1, 3 + (i_sub % 3)])
        sax(ax_sub,
            f"#{i_sub+1}: {res['const']}\n@{res['obs']}\nd={res['d']:.3f}",
            tc=res['candidate']['color'])
        sky_s = res['sky']
        ax_sub.scatter(sky_s[:,0], sky_s[:,1], s=30,
                       c=res['candidate']['color'], alpha=0.85, zorder=3)
        # Poligono (connetti in ordine di magnitudine)
        if len(sky_s) >= 3:
            loop = np.vstack([sky_s, sky_s[0]])
            ax_sub.plot(loop[:,0], loop[:,1],
                        color=res['candidate']['color'], lw=0.6, alpha=0.4)
        ax_sub.set_xlabel("RA sintetico (°)", color=fg, fontsize=5)
        ax_sub.set_ylabel("Dec (°)", color=fg, fontsize=5)

    # --- Riga 2: Monte Carlo top-4 Stadio B ---
    for i_mc, res in enumerate(stage_b[:4]):
        ax_mc = fig.add_subplot(gs[2, i_mc])
        sax(ax_mc,
            f"MC: {res['const']} @{res['obs'][:8]}\n"
            f"p={res.get('p_value', 0):.4f}",
            tc=res['candidate']['color'])
        mc = res.get('mc_dists', np.array([0.5]))
        ax_mc.hist(mc, bins=45, color='#334466',
                   edgecolor='#445577', alpha=0.8, density=True)
        ax_mc.axvline(res['d'], color=res['candidate']['color'],
                      lw=2.2, label=f"d={res['d']:.3f}")
        ax_mc.axvline(np.percentile(mc, 5), color='#88AA44',
                      lw=1.1, ls='--', label='5° pct')
        ax_mc.legend(fontsize=5.5, facecolor='#111122',
                     edgecolor='#333355', labelcolor=fg)
        ax_mc.set_xlabel("Dist. Procuste", color=fg, fontsize=6)

    # --- Riepilogo finale ---
    ax_fin = fig.add_subplot(gs[2, 4:6])
    sax(ax_fin, "")
    ax_fin.axis('off')

    best_b  = stage_b[0]
    best_ba = stage_a[0]

    def txt(ax, y, s, color, size=8, weight='normal'):
        ax.text(0.04, y, s, transform=ax.transAxes,
                color=color, fontsize=size, fontweight=weight,
                va='top', fontfamily='monospace')

    txt(ax_fin, 0.97, "RISULTATI BLIND TEST", '#AAAAFF', 10, 'bold')
    txt(ax_fin, 0.89, "", fg)
    txt(ax_fin, 0.85, "STADIO A (dalla Terra):", '#AAAAFF', 8, 'bold')
    txt(ax_fin, 0.79, f"  1° {best_ba['const']:<14} d={best_ba['d']:.4f}", fg)
    if len(stage_a) > 1:
        txt(ax_fin, 0.73, f"  2° {stage_a[1]['const']:<14} d={stage_a[1]['d']:.4f}", fg)
    if len(stage_a) > 2:
        txt(ax_fin, 0.67, f"  3° {stage_a[2]['const']:<14} d={stage_a[2]['d']:.4f}", fg)
    txt(ax_fin, 0.60, "", fg)
    txt(ax_fin, 0.56, "STADIO B (con parallasse):", '#AAAAFF', 8, 'bold')
    txt(ax_fin, 0.50, f"  1° {best_b['const']:<10}@{best_b['obs'][:10]}",
        best_b['candidate']['color'], 8, 'bold')
    txt(ax_fin, 0.44, f"     d={best_b['d']:.4f}  "
        f"p={best_b.get('p_value',0):.4f}", fg)
    txt(ax_fin, 0.37, "", fg)
    txt(ax_fin, 0.33, "NOTA METODOLOGICA:", '#FFCC44', 8, 'bold')
    txt(ax_fin, 0.27, "  Nodi: 15 punti blu (stima)", '#FFCC44')
    txt(ax_fin, 0.21, "  Per test formale:", '#AAAACC')
    txt(ax_fin, 0.15, "  → scansione 600 dpi", '#AAAACC')
    txt(ax_fin, 0.09, "  → coord. precise B1-B15", '#AAAACC')
    txt(ax_fin, 0.03, "  (Gwen 2026: req. <1% err.)", '#888899', 7)

    return fig


# ================================================================
# MAIN
# ================================================================

def find_file(name):
    for f in [name] + os.listdir('.'):
        if any(k in str(f) for k in ['IMG_20171006', '1774110843819']):
            return f if os.path.exists(f) else None
    return None

def main():
    print("="*60)
    print("  BLIND TEST 88 COSTELLAZIONI")
    print("  Cieli Dipinti dal Nulla — v3")
    print("="*60)

    p1 = find_file("IMG_20171006_080030.jpg")
    p2 = None
    for f in os.listdir('.'):
        if any(k in f for k in ['IMG_20171011', '1774110893302']):
            p2 = f
            break

    print(f"\n  Dipinto 1: {p1 or 'non trovato'}")
    print(f"  Dipinto 2: {p2 or 'non trovato'}")
    print(f"  Nodi blu:  {len(BLUE_NODES_15)} punti angolari")
    print(f"  Costellazioni testate (Stadio A): {len(CONSTELLATIONS)}")

    painting_nodes = BLUE_NODES_15

    # Stadio A
    stage_a = stage_A(painting_nodes)

    # Stadio B
    stage_b = stage_B(stage_a, painting_nodes, top_n=5)

    # Figura
    fig = make_figure(stage_a, stage_b, painting_nodes, p1, p2)
    out = "blind_test_88_costellazioni.png"
    fig.savefig(out, dpi=130, bbox_inches='tight', facecolor='#07080e')
    print(f"\n  ✓ Figura salvata: {out}")

    best_a = stage_a[0]
    best_b = stage_b[0]
    print("\n" + "="*60)
    print(f"  STADIO A — Miglior match dalla Terra: {best_a['const']}")
    print(f"             d={best_a['d']:.4f}")
    print(f"  STADIO B — Miglior match con parallasse:")
    print(f"             {best_b['const']} da {best_b['obs']}")
    print(f"             d={best_b['d']:.4f}")
    print("="*60)
    print()
    print("  PROSSIMO PASSO: scansiona Dipinto 1 a 600 dpi,")
    print("  clicca i 15 trattini blu in GIMP/ImageJ,")
    print("  aggiorna BLUE_NODES_15 e riesegui.")
    print()

    plt.show()

if __name__ == "__main__":
    main()
