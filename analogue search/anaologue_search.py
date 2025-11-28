# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 11:11:58 2024

@author: wangxuebing
"""

import pandas as pd
import numpy as np
import spectra_similarity as spesim
import warnings
warnings.filterwarnings("ignore")
#import database file (negative/positive)
database_F=pd.read_pickle("database_neg(noncommercial).pkl")

# input target precurosr and MS/MS (it is also possible to search for certain fragments/ neutral losses)
mz=465.03738
msms="73.02657:12 75.00655:36 75.01163:12 93.19914:12 101.02698:14 105.85258:12 110.46616:64 114.33952:13 117.01713:12 119.03525:279 120.5322:13 125.15148:13 131.97975:13 143.3336:13 169.24025:13 180.18596:13 183.00481:13 184.99701:25 186.97052:13 189.22395:13 193.71132:37 197.44646:13 201.28119:13 216.98892:55 220.05794:13 224.18301:13 226.07353:13 249.96861:13 254.98615:460 255.26784:13 271.19717:13 274.99725:13 282.97992:53 283.00299:13 288.88779:13 292.98047:120 293.02408:13 294.7457:13 302.98242:111 303.98602:14 309.758:13 315.20319:13 322.99139:21 323.12347:12 328.62045:13 343.81268:13 356.05536:13 389.26361:13 392.05994:12 419.26077:12 445.03305:12 465.04367:23 465.20847:18 465.24442:14 465.25919:26 465.27399:15 465.293:37 465.32895:15"
#MS2 error 
error=0.01
#whether ignore precursor match in cacluating similarity
filter_pre=1
#whether to filter MS2 by retain only top6 (intensity) fragment in any 50Da window of MS2
filter_50Da=1
# setting thresh for similarity score 
thresh=0
database_F["analogue_score"]=-1
database_F["analogue_score_num"]=-1
for i in range(len(database_F)):
    if(i%1000==0):
        print(i)
    scores=spesim.GNPS(msms,database_F["MS2"][i],mz,float(database_F["mz"][i]), error=0.01, filter_pre=1, filter_50Da=1)
    database_F["analogue_score"][i]=scores[0]
    database_F["analogue_score_num"][i]=scores[1]
database_F3=database_F.sort_values(by=["analogue_score"],ascending=False)
database_F4=database_F3[database_F3["analogue_score"]>thresh]
database_F4.to_csv(str(mz)+"_analogues_output.csv",index=False)



