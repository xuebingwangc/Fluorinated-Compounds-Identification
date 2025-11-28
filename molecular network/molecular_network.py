# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 08:33:04 2025

@author: wangxuebing
"""

import  pandas as pd
import numpy as np
import spectra_similarity as spesim
# MS2 preprocessing
def MSMSspe1(str1, thresh=0.01):
    try:
        count = str1.count(":")
        msmsspe = np.zeros(shape=(count, 2))
        strlist = str1.split(" ")
        for i in range(count):
            s = strlist[i].split(":")
            msmsspe[i][0] = eval(s[0])
            msmsspe[i][1] = eval(s[1])
        max_val = np.max(msmsspe[:, 1]) if count > 0 else 0
        if max_val > 0:
            msmsspe = msmsspe[msmsspe[:, 1] > thresh * max_val]
        return msmsspe
    except:
        return np.zeros(shape=(0, 2))
# find neighbors of identified fluorinated compounds (seeds) by Flink 
def spe_sim_outnetwork(df_parent,df,deltamz=[2,200],error=0.01,thresh_score=0.5,thresh_num=1,thresh_intensity=0.01,filter_pre=1, filter_50Da=1,filter_minmz=100,non_filter=[39.00518,68.99576,77.96552,78.99895,80.99576,82.96085,84.99067,92.99576,98.95577],nl_filter=[43.98983,79.95682]):
    name_parent=[]
    id_parent=[]
    mz_parent=[]
    rt_parent=[]
    smiles_parent=[]
    formula_parent=[]
    scores=[]
    nums=[]
    index_neg=[]
    df_parent=df_parent.reset_index(drop=True)
    df=df.dropna(subset=["MS/MS spectrum"]).reset_index(drop=True)
    for i in range(len(df_parent)):
        for j in range(len(df)):
            if(abs(df_parent["Precursor m/z"][i]-df["Average Mz"][j])>12 and abs(df_parent["Precursor m/z"][i]-df["Average Mz"][j])<200 and (df["Alignment ID"][j]  not in list(df_parent["Alignment ID"]))):
                score1,num1=spesim.Flink(df_parent["MS/MS spectrum"][i], df["MS/MS spectrum"][j], df_parent["Precursor m/z"][i], df["Average Mz"][j],thresh=thresh_intensity,error=error,filter_pre=filter_pre, filter_50Da=filter_50Da,filter_minmz=filter_minmz,non_filter=non_filter,nl_filter=nl_filter)
                if(score1>=thresh_score and num1>=thresh_num):
                    index_neg.append(j)
                    name_parent.append(df_parent['Name'][i])
                    mz_parent.append(df_parent['Precursor m/z'][i])
                    id_parent.append(df_parent["Alignment ID"][i])
                    rt_parent.append(df_parent["Average Rt(min)"][i])
                    smiles_parent.append(df_parent["SMILES"][i])
                    formula_parent.append(df_parent["Formula"][i])
                    scores.append(score1)
                    nums.append(num1)
    df_new=df.iloc[index_neg].reset_index(drop=True)
    df_new["parent_name"]=name_parent
    df_new["parent_id"]=id_parent
    df_new["parent_mz"]=mz_parent
    df_new["parent_rt"]=rt_parent
    df_new["parent_smiles"]=smiles_parent
    df_new["parent_formula"]=formula_parent
    df_new["Flink_score"]=scores
    df_new["Flink_num"]=nums
    return df_new
#annotate common fragments/neutral losses of parent and its neighbor
def commonfrag(str1,str2,error=0.01):
    list1=str1.split("/")
    list1=[eval(i) for i in list1 if i!=""]
    list2=str2.split("/")
    list2=[eval(i) for i in list2 if i!=""]
    result = [(x, y) for x in list1 for y in list2 if abs(x - y) <= error]
    return result
def commonfrags(df_new,df_parent,error=0.01,thresh=0.01):
    frag0=[]
    nl0=[]
    for i in range(len(df_new)):
        ms2=df_new["MS/MS spectrum"][i]
        mzp=float(df_new["Average Mz"][i])
        spe=MSMSspe1(df_new["MS/MS spectrum"][i],thresh=thresh)
        frag1=""
        nl1=""
        for j in range(len(spe)):
            frag1+=str(spe[j,0])+"/"
            nl1+=str(round(mzp-spe[j,0],5))+"/"
        frag0.append(frag1)  
        nl0.append(nl1)
    df_new["MZ"]=frag0
    df_new["NL"]=nl0
    frag1=list(df_new["MZ"])
    nl1=list(df_new["NL"])
    frag2=[]
    nl2=[]
    for i in range(len(df_parent)):
        ms2=df_parent["MS/MS spectrum"][i]
        mzp=float(df_parent["Precursor m/z"][i])
        spe=MSMSspe1(df_parent["MS/MS spectrum"][i],thresh=thresh)
        frag3=""
        nl3=""
        for j in range(len(spe)):
            frag3+=str(spe[j,0])+"/"
            nl3+=str(round(mzp-spe[j,0],5))+"/"
        frag2.append(frag3)  
        nl2.append(nl3)
    df_parent["MZ"]=frag2
    df_parent["NL"]=nl2
    df_parent_subset = df_parent[["Alignment ID", "MZ", "NL"]].rename(
        columns={
            "Alignment ID": "parent_Alignment_ID",  # rename columns
            "MZ": "MZ_parent",
            "NL": "NL_parent"
        }
    )
    
    df_new = df_new.merge(
        df_parent_subset,
        left_on="parent_id",       
        right_on="parent_Alignment_ID",  
        how="left"
    )
    df_new.drop("parent_Alignment_ID", axis=1, inplace=True)
    common_frag=[]
    common_nl=[]
    for i in range(len(df_new)):
        frag4=commonfrag(df_new["MZ"][i],df_new["MZ_parent"][i],error=error)
        nl4=commonfrag(df_new["NL"][i],df_new["NL_parent"][i],error=error)
        common_frag.append(frag4)
        common_nl.append(nl4)
    df_new["common_frag"]=common_frag
    df_new["common_nl"]=common_nl
    return df_new

#input parent seeds
df_parent=pd.read_csv("test_parent.csv")
#input all peaks in pos/neg as neighbor candidates
df=pd.read_csv("allpeaks_neg.csv")
#find neighbors
#deltamz: precursor mz searching range, error : MS/MS fragment mass error (Da), thresh_score: spectra similarity socre threshhold, thresh_num: treshhold number of matched fragments/neutral losses in spectra similarity caculation, thresh_intensity:relative abundance threshhold in MS/MS processing
#filter_pre: fliter precursor in caclating simliarity,  filter_50Da: filter top6 intensity in any 50 Da window,filter_minmz: fragment low mz fragment, non_filter: charaterristic fragments lower than minmz but not filtered (present settings for neg), nl_filter:common neutral losses to filter  
df_new=spe_sim_outnetwork(df_parent,df,deltamz=[2,200],error=0.01,thresh_score=0.5,thresh_num=1,thresh_intensity=0.01,filter_pre=1, filter_50Da=1,filter_minmz=100,non_filter=[39.00518,68.99576,77.96552,78.99895,80.99576,82.96085,84.99067,92.99576,98.95577],nl_filter=[43.98983,79.95682])
#annotation of common fragments and neutral losses of parent and neighbors
df_new1=commonfrags(df_new,df_parent,error=0.01,thresh=0.01)
#output
df_new1.to_csv("test_parent_MNoutput.csv",index=False)


