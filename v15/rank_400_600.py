# Classifica per d delle costellazioni a 400-600 al (uso: python rank_400_600.py rilevati15 repo16 repo14)
import sys; sys.path.insert(0,str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np, blind_test_v2 as v2, blind_test_v3 as v3
from astropy.table import Table
import logging; logging.basicConfig(level=logging.WARNING)
tab=Table.read(''+str(__import__("pathlib").Path(__file__).resolve().parent)+'/gaia_sky_out/blind_test_v2/hipparcos_hp6.0.ecsv')
for nome in sys.argv[1:]:
    nodi=v3.INSIEMI[nome]; v2.N_NODI=len(nodi)
    g=v2.stelle_per_costellazione(tab,5.0)
    f,desc,info=v2.costruisci_libreria(g,600,[400,500,541,600])
    nomi=np.array([x[0] for x in info]); dly=np.array([x[1] for x in info]); m5=dly>=400
    nn=v2.normalizza(nodi); r={}
    for c in g: r[c]=v2.miglior_match(nn,f,desc,1500,maschera=(nomi==c)&m5)[0]
    cl=sorted(r.items(),key=lambda x:x[1])
    pos=[c for c,_ in cl].index('Draco')+1
    print(nome,'Draco pos',pos,'su',len(cl),'d=%.4f'%r['Draco'],'| top5',[(c,round(d,4)) for c,d in cl[:5]],flush=True)
    Table(rows=[{'costellazione':c,'d_400_600':round(d,5)} for c,d in cl]).write(str(__import__("pathlib").Path(__file__).resolve().parent)+f'/gaia_sky_out/blind_test_v3/rank_400_600_{nome}.csv',format='ascii.csv',overwrite=True)
