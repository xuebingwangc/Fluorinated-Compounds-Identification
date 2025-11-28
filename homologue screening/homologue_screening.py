

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 15:47:26 2025

@author: wangxuebing
"""
#function 1 data preprocessing
import pandas as pd
from collections import Counter
import numpy as np
import networkx as nx

def MSMSspe(str1, thresh=0.01):
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
def sum_ms2(str1,mz,thresh=0.01,filter_pre=1):
    msmsspe=MSMSspe(str1,thresh=thresh)
    if(filter_pre):
        msmsspe = msmsspe[(msmsspe[:, 0] < mz - 17)]
    return sum(msmsspe[:,1])
    

def read_data(filename,thresh=0.01):
    data=pd.read_csv(filename)
    data=data.dropna(subset=["MS/MS spectrum"]).reset_index(drop=True)
    mz=[]
    nl=[]
    height=[]
    for i in range(len(data)):
        spe=MSMSspe(data["MS/MS spectrum"][i],thresh=thresh)
        mzp=float(data["Average Mz"][i])
        mz1=""
        nl1=""
        height1=""
        for j in range(len(spe)):
            mz1+=str(spe[j,0])+"/"
            nl1+=str(round(mzp-spe[j,0],5))+"/"
            height1+=str(spe[j,1])+"/"
        mz.append(mz1)  
        nl.append(nl1)
        height.append(height1)
    data["MZ"]=mz
    data["NL"]=nl
    data["height"]=height
    data.to_csv(filename,index=False)    
    return filename
#function 2 homologue finding (including retention time judge)
def class_rtjudge3(list_n,list_rt,rt_error=1):
    if(list_rt[1]<list_rt[0]-rt_error or list_rt[2]<list_rt[1]-rt_error):
        return 0

    pre_rt3=((list_rt[1]-list_rt[0])/(list_n[1]-list_n[0]))*(list_n[2]-list_n[1])+list_rt[1]
    if(pre_rt3<list_rt[2]-rt_error):
        return 0
    return 1
def class_rtjudge(line_index,line_n,line_rt,rt_error=1):
    line_index_match=[]
    line_n_match=[]
    line_rt_match=[]
    for i in range(len(line_index)):
        if(line_n[-1]-line_n[i]<2):
            continue
        for j in range(i+1,len(line_index)):
            if(line_n[-1]-line_n[j]<1):
                break
            if(line_n[j]==line_n[i]):
                continue
            for k in range(j+1,len(line_index)):
                if(line_n[k]==line_n[j]):
                    continue
                if(class_rtjudge3([line_n[i],line_n[j],line_n[k]],[line_rt[i],line_rt[j],line_rt[k]],rt_error=rt_error)):
                    if(line_index[i] not in line_index_match):
                        line_index_match.append(line_index[i])
                        line_n_match.append(line_n[i])
                        line_rt_match.append(line_rt[i])
                    if(line_index[j] not in line_index_match):
                        line_index_match.append(line_index[j])
                        line_n_match.append(line_n[j])
                        line_rt_match.append(line_rt[j])
                    if(line_index[k] not in line_index_match):
                        line_index_match.append(line_index[k])
                        line_n_match.append(line_n[k])
                        line_rt_match.append(line_rt[k])
    return line_index_match,line_n_match
def class_find(filename,homo=49.99681,mass_error=5,rt_error=1,min_n=3,flag_md=1):
    data=pd.read_csv(filename)
    filename_output= filename.split(".csv")[0]+"_"+str(homo)+".csv"
    if(flag_md):
        index=[]
        for i in range(len(data)):
            mz_md=float(data["Average Mz"][i])/49.99681*50
            md=mz_md-int(mz_md)
            if(md>0.85 or md<0.15):
                index.append(i)
        data_md=data.iloc[index].reset_index(drop=True)
        data_md=data_md.sort_values(by="Average Mz").reset_index(drop=True)
    else:
        data_md=data.sort_values(by="Average Mz").reset_index(drop=True)
    homo_maxnum=int((data_md["Average Mz"][len(data_md)-1]-data_md["Average Mz"][0])/homo)
    class_index=[]
    class_num=0
    peaks_used=[]
    line_n=[]
    for i in range(len(data_md)):
        if(i in peaks_used):
            continue
        lines1=[]
        homo_n=[]
        for j in range(homo_maxnum+1):
            mz_n=data_md["Average Mz"][i]+j*homo
            if(mz_n>data_md["Average Mz"][len(data_md)-1]+1):
                break
            data_md1=data_md[(data_md["Average Mz"]>mz_n*(1-mass_error*0.000001))&(data_md["Average Mz"]<mz_n*(1+mass_error*0.000001))]
            if(len(data_md1)==0):
                continue
            index1=list(data_md1.index)
            index1=[m for m in index1 if(m>=i)]
            lines1+=index1
            homo_n+=len(index1)*[j]
        if(len(set(homo_n))<min_n):
            continue
        #retention time filter
        line_rt=list(data_md.iloc[lines1]["Average Rt(min)"])
        lines1,homo_n=class_rtjudge(lines1,homo_n,line_rt,rt_error=rt_error)
        if(len(set(homo_n))<min_n):
            continue
        
        class_num+=1
        class_index+=len(lines1)*[class_num]
        peaks_used+=lines1
        line_n+=homo_n    
    data_output=data_md.iloc[peaks_used].reset_index(drop=True)     
    data_output["Class"]=class_index 
    data_output["n"]=line_n
    data_output=data_output.reindex(columns=["Class","n","Average Mz","Average Rt(min)","MZ","NL","height","Alignment ID","MS1 isotopic spectrum","MS/MS spectrum"],fill_value=1)
    data_output.to_csv(filename_output,index=False)
    return filename_output

#function 3 merge of classes with same ID
def merge_classes(filename, homo=49.99681,alignment_col='Alignment ID', class_col='Class'):
    df=pd.read_csv(filename)
    G = nx.Graph()
    for _, group in df.groupby(alignment_col):
        classes = group[class_col].unique()
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                G.add_edge(classes[i], classes[j])

    connected_components = list(nx.connected_components(G))
    all_classes = set(df[class_col].unique())
    mapped_classes = set()
    class_mapping = {}
    for new_class, component in enumerate(connected_components, start=1):
        for old_class in component:
            class_mapping[old_class] = new_class
            mapped_classes.add(old_class)
    
    isolated_classes = all_classes - mapped_classes
    next_class_id = len(connected_components) + 1
    for old_class in isolated_classes:
        class_mapping[old_class] = next_class_id
        next_class_id += 1
    df[class_col] = df[class_col].map(class_mapping)
    df = df.drop_duplicates(subset=alignment_col, keep='first').reset_index(drop=True)
    df = df.sort_values(by=[class_col, "Average Mz"]).reset_index(drop=True)
    def calculate_n(group):
        first_average_mz = group["Average Mz"].iloc[0]  
        group["n"] = group["Average Mz"].apply(lambda x: round((x - first_average_mz) / homo))
        group.iloc[0, group.columns.get_loc("n")] = 0  
        return group
    df["n"] = 0  
    df = df.groupby(class_col, group_keys=False).apply(calculate_n)
    df.to_csv(filename,index=False)
    return df
    
#function 4 mark fragments with database
def fragment_mark(filename_data,filename_database,error_ms2=0.01,thresh=0.01):
    data=pd.read_csv(filename_data)
    database=pd.read_csv(filename_database)
    database_frag=database[database["type"]=="fragment"].reset_index(drop=True)
    database_nl=database[database["type"]=="neutral loss"].reset_index(drop=True)
    database_frag=database_frag.sort_values(by="m/z").reset_index(drop=True)
    database_nl=database_nl.sort_values(by="m/z").reset_index(drop=True)
    frag=[]
    frag_height=[]
    for i in range(len(data)):   
        frag1=""
        frag1_height=0
        try:
            mz=data["MZ"][i].split("/")[:-1]
            height=data["height"][i].split("/")[:-1]
            
        except:
            frag.append("")
            frag_height.append("")
            continue
        if(len(mz)<1):
            frag.append("")
            frag_height.append("")
            continue
        mz=[eval(j) for j in mz]
        for j in range(len(mz)):
            for m in range(len(database_frag)):
                if(abs(mz[j]-database_frag["m/z"][m])<error_ms2):
                    frag1+=database_frag["ion"][m]+"/"
                    frag1_height+=float(height[j])
                    break
        frag.append(frag1)
        if(frag1_height!=0):
            try:
                sum_height=sum_ms2(data["MS/MS spectrum"][i],float(data["Average Mz"][i]),thresh=thresh,filter_pre=1)
                frag_height.append(frag1_height/sum_height)
            except:
                sum_height=sum_ms2(data["MS/MS spectrum"][i],float(data["Average Mz"][i]),thresh=0,filter_pre=0)
                frag_height.append(frag1_height/sum_height)
        else:
            frag_height.append(0)
    data["PFAS_frag"]=frag
    data["PFAS_frag_height_percent"]=frag_height
    nl=[]
    nl_height=[]
    for i in range(len(data)):   
        nl1=""
        nl1_height=0
        try:
            mz=data["NL"][i].split("/")[:-1]
            height=data["height"][i].split("/")[:-1]
            
        except:
            nl.append("")
            nl_height.append("")
            continue
        if(len(mz)<1):
            nl.append("")
            nl_height.append("")
            continue
        mz=[eval(i) for i in mz]
        for j in range(len(mz)):
            for m in range(len(database_nl)):
                if(abs(mz[j]-database_nl["m/z"][m])<error_ms2):
                    nl1+=database_nl["ion"][m]+"/"
                    nl1_height+=float(height[j])
                    break
        nl.append(nl1)
        if(nl1_height!=0):
            try:
                sum_height=sum_ms2(data["MS/MS spectrum"][i],float(data["Average Mz"][i]),thresh=thresh,filter_pre=1)
                nl_height.append(nl1_height/sum_height)
            except:
                sum_height=sum_ms2(data["MS/MS spectrum"][i],float(data["Average Mz"][i]),thresh=0,filter_pre=0)
                nl_height.append(nl1_height/sum_height)
        else:
            nl_height.append(0)
    data["PFAS_nl"]=nl
    data["PFAS_nl_height_percent"]=nl_height
    data.to_csv(filename_data,index=False)   
def count_common_elements(list1, list2):
    counter1 = Counter(list1)
    counter2 = Counter(list2)
    common = counter1 & counter2  
    return sum(common.values())
def safe_split(value):
    if isinstance(value, str) and value.strip():  
        seen = set()
        result = []
        for item in value.split('/'):
            if item != '' and item not in seen:
                seen.add(item)
                result.append(item)
        return result
    elif isinstance(value, float) and np.isnan(value):  
        return []  
    else:
        return []
def add_count_column(filename):
    df = pd.read_csv(filename)
    df['count'] = 0
    for class_id in df['Class'].unique():
        class_df = df[df['Class'] == class_id]
        original_indices = class_df.index
        class_df = class_df.reset_index(drop=True)
        class_df['frag_split'] = class_df['PFAS_frag'].apply(safe_split)
        class_df['nl_split'] = class_df['PFAS_nl'].apply(safe_split)
        for i in range(len(class_df)):
            max_m = 0  
            max_n = 0  
            frag_i = class_df.loc[i, 'frag_split']
            nl_i = class_df.loc[i, 'nl_split']
            if not frag_i and not nl_i:
                df.loc[class_df.index[i], 'count'] = 0
                continue
            for j in range(len(class_df)):
                if i != j:
                    n_i = class_df.loc[i, 'n']
                    n_j = class_df.loc[j, 'n']
                    if n_i == n_j:
                        continue
                    frag_j = class_df.loc[j, 'frag_split']
                    nl_j = class_df.loc[j, 'nl_split']
                    if frag_i and frag_j:
                        max_m = max(max_m, count_common_elements(frag_i, frag_j))
                    if nl_i and nl_j:
                        max_n = max(max_n, count_common_elements(nl_i, nl_j))
            df.loc[original_indices[i], 'count'] = max_m + max_n
    max_counts = df.groupby(['Class', 'n'])['count'].max().reset_index()
    count_class = max_counts.groupby('Class')['count'].sum().reset_index()
    count_class.rename(columns={'count': 'count_class'}, inplace=True)
    df = pd.merge(df, count_class, on='Class', how='left')
    df = df.sort_values(by="count_class", ascending=False).reset_index(drop=True)
    class_mapping = {old_class: new_class for new_class, old_class in enumerate(df["Class"].unique(), start=1)}
    df["Class"] = df["Class"].map(class_mapping)
    df = df.sort_values(by=["Class", "Average Mz"]).reset_index(drop=True)
    df.to_csv(filename, index=False)
    return df
import spectra_similarity as spesim
def spe_sim_internalnetwork(filename,thresh=0.01,error=0.01, thresh_sim=0.3,filter_pre=1, filter_50Da=1,filter_minmz=100,non_filter=[39.00518,68.99576,77.96552,78.99895,80.99576,82.96085,84.99067,92.99576,98.95577],nl_filter=[43.98983,79.95682]):
    df = pd.read_csv(filename)
    df1 = df.sort_values(by=['Class', 'n'], ascending=[True, True])
    df1=df1.reset_index(drop=True)
    max_spesim_score=[]
    max_spesim_num=[]
    for i in range(len(df1)):
        df2=df1[df1["Class"]==df1["Class"][i]]
        df3=df2[df2["n"]!=df1["n"][i]]
        df4=df3.copy()
        df4=df4.reset_index(drop=True)
        score=0
        num=0
        for j in range(len(df4)):
            score1,num1=spesim.Flink(df1["MS/MS spectrum"][i],df4["MS/MS spectrum"][j],float(df1["Average Mz"][i]),float(df4["Average Mz"][j]),thresh=thresh,error=error, filter_pre=filter_pre, filter_50Da=filter_50Da,filter_minmz=filter_minmz,non_filter=non_filter,nl_filter=nl_filter)
            if(score1>score):
                score=score1
                num=num1
            if(score1==score and num1>num):
                num=num1
        max_spesim_score.append(score)
        max_spesim_num.append(num)
    df1["max_Flink_score"]=max_spesim_score
    df1["max_Flink_num"]=max_spesim_num
    df1['max_Flink_score_class'] = df1.groupby('Class')['max_Flink_score'].transform('max')
    df1['max_Flink_num_class'] = df1.groupby('Class')['max_Flink_num'].transform('max')
    df1 = df1[df1["max_Flink_score_class"] > thresh_sim].reset_index(drop=True)
    df1.to_csv(filename, index=False)
    return df1

"""read data"""
# input data file name 
data_filename1=r"E:\wxb\yjq_cs\homologue screening\code upload\homologue screening\test_homo.csv"
#fragment filter by relative abundance
thresh=0.01
data_filename=read_data(data_filename1,thresh=thresh)
"""find homologues"""
#homologue unit mass CF2,CF2CH2,CF2O,C3F6O
homo1=49.99681
#homo2=64.01246
#homo3=65.99172
#homo4=165.98533
#MS1 error for homologue screening, relative error (ppm) 
error_ms1=5
#retentime time error (min), requeire n+1 had a retentime time higher than n under error_rt (very general setting)
error_rt=2
# whether use mass defect filter (>0.85 and <0.15)
flag_md=1
#minimum number of homologues
min_n=3
filename1=class_find(data_filename,homo=homo1,mass_error=error_ms1,rt_error=error_rt,min_n=min_n,flag_md=flag_md)
"""merge classes with same ID"""
merge_classes(filename1,homo=homo1)
"""fragment annnotation by database"""
#filenames of databases (provided fragment database was from negative mode)
filename_database=r"E:\wxb\yjq_cs\homologue screening\code upload\homologue screening\fragment database.csv"
#MS2 error for fragmengt annotation
error_ms2=0.01
fragment_mark(filename1,filename_database,error_ms2)
add_count_column(filename1)
"""annotate the spectra similarity score of the classes"""
#whether ignore precursor match in cacluating similarity
filter_pre=1
#whether to filter MS2 by retain only top6 (intensity) fragment in any 50Da window of MS2
filter_50Da=1
#filter mz<filter_minmz if too many noise/unwanted matches,it could be set to 0 for not using
filter_minmz=100
#some of the charateristic fragments that do mot filter, could be set empty list for not using 
non_filter=[39.00518,68.99576,77.96552,78.99895,80.99576,82.96085,84.99067,92.99576,98.95577]
# some common netural losses that is unwanted (causing high false positive),could be set empty list for not using 
nl_filter=[43.98983,79.95682]
thresh_sim=0.3
spe_sim_internalnetwork(filename1,thresh=thresh,error=error_ms2, thresh_sim=thresh_sim,filter_pre=filter_pre, filter_50Da=filter_50Da,filter_minmz=filter_minmz,non_filter=non_filter,nl_filter=nl_filter)



