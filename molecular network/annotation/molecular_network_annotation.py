# -*- coding: utf-8 -*-
"""
Created on Tue May 13 14:53:30 2025

@author: wangxuebing
"""

import numpy as np 
import pandas as pd
import re
from collections import defaultdict
import os
import rdkit.Chem as Chem

"""annotate formula with KEGG reaction formula change (formulas with negative atom number are filtered)"""
#input KEGG reaction, their formula changes and its mass differences
pmd_kegg=pd.read_csv("KEGG_MD.csv")
#import molecular network results
data_mn=pd.read_csv("test_parent_MNoutput.csv")
#input mass error for matching formula
error=0.005
#input ion mode (1 for positive mode,-1 for negative mode)
mode=-1

data_mn["pmd_exp"]=data_mn["Average Mz"]-data_mn["parent_mz"]
data_mn['pmd_kegg'] = None
for i, row in data_mn.iterrows():
    pmd_value = row['pmd_exp']
    mask = (pmd_kegg['pmd'] >= pmd_value - error) & (pmd_kegg['pmd'] <= pmd_value + error)
    matched_trans = pmd_kegg.loc[mask, 'trans'].unique().tolist()
    data_mn.at[i, 'pmd_kegg'] = matched_trans

# formula analysis
def parse_formula(formula):
    pattern = r'([A-Z][a-z]*)(-?\d*)'
    elements = re.findall(pattern, formula)
    element_dict = defaultdict(int)
    for elem, count in elements:
        if count == '':
            count = 1
        elif count == '-':
            count = -1
        else:
            count = int(count)
        element_dict[elem] += count
    
    return dict(element_dict)

def combine_formulas(formula1, formula2):
    dict1 = parse_formula(formula1)
    dict2 = parse_formula(formula2)
    
    combined = defaultdict(int)
    for elem in set(dict1) | set(dict2):
        combined[elem] = dict1.get(elem, 0) + dict2.get(elem, 0)
    
    return dict(combined)

def dict_to_formula(element_dict):
    if any(count < 0 for count in element_dict.values()):
        return None  
    
    formula = []
    for elem in sorted(element_dict.keys(), key=lambda x: (x != 'C', x != 'H', x)):
        count = element_dict[elem]
        if count == 0:
            continue  
        elif count == 1:
            formula.append(elem)  
        else:
            formula.append(f"{elem}{count}") 
    return ''.join(formula)

def process_row(row):
    parent = row['parent_formula']
    trans_list = row['pmd_kegg']
    
    if not trans_list:
        return []  
    
    result_formulas = []
    for trans in trans_list:
        try:
            combined = combine_formulas(parent, trans)
            formula = dict_to_formula(combined)
            if formula is not None:  
                result_formulas.append(formula)
        except:
            continue  
    
    return result_formulas
data_mn['Formulas_pmd'] = data_mn.apply(process_row, axis=1)
#data_mn.to_csv("test_parent_MNoutput_formula_annotated.csv",index=False)

""" combine biotranformer prediction results from different pathway, and annotated molecular network results"""

def iso(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles),isomericSmiles=0)

#  combine biotranformer prediction results
folder_path = "biotransformer_result"
csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]

dfs = []
for file in csv_files:
    df = pd.read_csv(file)
    dfs.append(df)

merged_df = pd.concat(dfs, ignore_index=True)
# remove duplicates in biotransformer results by SMILES
biotransformer_all = merged_df.drop_duplicates(subset=["SMILES", "Precursor SMILES"])
columns = biotransformer_all.columns.tolist()
biotransformer_all = biotransformer_all[~biotransformer_all.apply(lambda row: row.tolist() == columns, axis=1)]
biotransformer_all = biotransformer_all.reset_index(drop=True)
biotransformer_all=biotransformer_all.reset_index(drop=True)

# caculate mz, [M+H]+ or [M-H]-
biotransformer_all["Major Isotope Mass"] = pd.to_numeric(biotransformer_all["Major Isotope Mass"], errors="coerce")
biotransformer_all["mz"] = biotransformer_all["Major Isotope Mass"] +mode* 1.00728

# standardize smiles
data_mn["parent_smiles"] = data_mn["parent_smiles"].apply(iso)
biotransformer_all["Precursor SMILES"]=biotransformer_all["Precursor SMILES"].apply(iso)

#match with smiles and mz
def add_biotransformer_columns(data_mn, biotransformer_all, error=0.002):
    data_mn['biotrans_smiles'] = [[] for _ in range(len(data_mn))]
    data_mn['biotrans_formula'] = [[] for _ in range(len(data_mn))]
    data_mn['biotrans_mz'] = [[] for _ in range(len(data_mn))]
    for idx, row in data_mn.iterrows():
        parent_smiles = row['parent_smiles']
        avg_mz = row['Average Mz']
        matches = biotransformer_all[
            (biotransformer_all['Precursor SMILES'] == parent_smiles) &
            (abs(biotransformer_all['mz'] - avg_mz) <= error)]
        if not matches.empty:
            data_mn.at[idx, 'biotrans_smiles'] = matches['SMILES'].tolist()
            data_mn.at[idx, 'biotrans_formula'] = matches['Molecular formula'].tolist()
            data_mn.at[idx, 'biotrans_mz'] = matches['mz'].tolist()
    
    return data_mn

data_mn = add_biotransformer_columns(data_mn, biotransformer_all,error=error)
data_mn.to_csv("test_parent_MNoutput_annotated.csv",index=False)
