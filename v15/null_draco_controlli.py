# Distribuzione casuale (100 insiemi, seme 11) usata per il p del Dragone nei controlli finali
import sys; sys.path.insert(0,str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np, blind_test_v2 as v2, blind_test_v3 as v3, blind_test_v3_controlli as c
from astropy.table import Table
import logging; logging.disable(logging.CRITICAL)
t=Table.read(str(__import__("pathlib").Path(__file__).resolve().parent)+'/gaia_sky_out/blind_test_v2/hipparcos_hp6.0.ecsv'); v2.N_NODI=15
g=v2.stelle_per_costellazione(t,5.0)
f,desc,info=v2.costruisci_libreria({'Draco':g['Draco']},600,[400,500,541,600])
nodi=v3.RILEVATI15; _,sep=v3.coppie_strette(nodi)
riq=((nodi[:,0].min(),nodi[:,0].max()),(nodi[:,1].min(),nodi[:,1].max()))
rng=np.random.default_rng(11)
cas=[v2.normalizza(c.a_coppie_corretto(rng,15,riq,sep)) for _ in range(100)]
nd=np.sort([v2.miglior_match(x,f,desc,500)[0] for x in cas])
print('10 piu bassi:',np.round(nd[:10],4))
print('d dipinto',round(v2.miglior_match(v2.normalizza(nodi),f,desc,500)[0],4))
Table({'d_casuale_draco':nd}).write(str(__import__("pathlib").Path(__file__).resolve().parent)+'/gaia_sky_out/blind_test_v3/null_draco_controlli.csv',format='ascii.csv',overwrite=True)
