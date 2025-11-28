

# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 16:07:42 2024

@author: wxb
"""


import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
from joblib import load
from joblib import dump
import pandas as pd  
import numpy as np  
from tqdm import tqdm
import time
import scipy
import scipy.stats

# MS2 data preprocessing

def MSMSspe(str1,mz,filter_pre=1,filter_50Da=1):
    count = str1.count(":")  
    msmsspe = np.zeros(shape=(count, 2))  
    strlist = str1.split(" ") 
    for i in range(count):
        s = strlist[i].split(":")  
        msmsspe[i][0] = eval(s[0]) 
        msmsspe[i][1] = eval(s[1]) 
    if(filter_pre):
        msmsspe = msmsspe[(msmsspe[:, 0] < mz - 17)| (msmsspe[:, 0] > mz + 17)]
    if(len(msmsspe)==0):
        return msmsspe
    if(filter_50Da):
        min_mz = int(np.min(msmsspe[:, 0]) // 50 * 50)
        max_mz = int(np.max(msmsspe[:, 0]) // 50 * 50 + 50)
        filtered_msmsspe = []
        
        for lower_bound in range(min_mz, max_mz, 50):
            upper_bound = lower_bound + 50
            group = msmsspe[(msmsspe[:, 0] >= lower_bound) & (msmsspe[:, 0] < upper_bound)]
            if len(group) > 0:
                top_6 = group[np.argsort(group[:, 1])][-6:]
                filtered_msmsspe.append(top_6)
        msmsspe=filtered_msmsspe.copy()
    if len(msmsspe) > 0:
        msmsspe = np.vstack(msmsspe)
    else:
        msmsspe = np.array(msmsspe)
    msmsspe=msmsspe[msmsspe[:,1]>0]
    return msmsspe  


def MSMSspe_pfas(str1,mz,thresh=0.01,filter_pre=1,filter_50Da=1,filter_minmz=100,non_filter=[39.00518,68.99576,77.96552,78.99895,80.99576,82.96085,84.99067,92.99576,98.95577],nl_filter=[43.98983,79.95682],ms2_error=0.01):
    count = str1.count(":")  
    msmsspe = np.zeros(shape=(count, 2))  
    strlist = str1.split(" ")  
    for i in range(count):
        s = strlist[i].split(":") 
        msmsspe[i][0] = eval(s[0])  
        msmsspe[i][1] = eval(s[1])  
    if(filter_pre):
        msmsspe = msmsspe[(msmsspe[:, 0] < mz - 17)| (msmsspe[:, 0] > mz + 17)]
    msmsspe = np.array([row for row in msmsspe if row[0] >= filter_minmz or np.any(np.abs(non_filter - row[0]) <= ms2_error)])
    msmsspe = np.array([row for row in msmsspe if not any(np.abs((mz - row[0]) - nl) <= ms2_error for nl in nl_filter)])
    if(len(msmsspe)==0):
        return msmsspe
    if(filter_50Da):
        min_mz = int(np.min(msmsspe[:, 0]) // 50 * 50)
        max_mz = int(np.max(msmsspe[:, 0]) // 50 * 50 + 50)
        filtered_msmsspe = []
        
        for lower_bound in range(min_mz, max_mz, 50):
            upper_bound = lower_bound + 50
            group = msmsspe[(msmsspe[:, 0] >= lower_bound) & (msmsspe[:, 0] < upper_bound)]
            if len(group) > 0:
                top_6 = group[np.argsort(group[:, 1])][-6:]
                filtered_msmsspe.append(top_6)
        msmsspe=filtered_msmsspe.copy()
    if len(msmsspe) > 0:
        msmsspe = np.vstack(msmsspe)
    else:
        msmsspe = np.array(msmsspe)
    max_msms=max(msmsspe[:,1])
    msmsspe=msmsspe[msmsspe[:,1]>thresh*max_msms]
    return msmsspe  

def MSMSspe1(str1):
    count=str1.count(":")
    msmsspe=np.zeros(shape=(count,2))
    strlist=str1.split(" ")
    for i in range(count):
        s=strlist[i].split(":")
        msmsspe[i][0]=eval(s[0])
        msmsspe[i][1]=eval(s[1])
    msmsspe=msmsspe[msmsspe[:,1]>0]
    return msmsspe

#GNPS spectra similarity algorithm
def GNPS(str1, str2, mz1_, mz2_, error=0.01, filter_pre=1, filter_50Da=1):
    spec1 = MSMSspe(str1, mz1_, filter_pre, filter_50Da) 
    spec2 = MSMSspe(str2, mz2_, filter_pre, filter_50Da)  

    if len(spec1) == 0 or len(spec2) == 0: 
        return 0, 0

    spec1[:, 1] = np.sqrt(spec1[:, 1]) / np.sqrt(np.sum(spec1[:, 1]))
    spec2[:, 1] = np.sqrt(spec2[:, 1]) / np.sqrt(np.sum(spec2[:, 1]))

    diff = mz2_ - mz1_ 

    match1 = np.abs(spec2[:, 0][:, np.newaxis] - spec1[:, 0])
    match2 = np.abs(spec2[:, 0][:, np.newaxis] - (spec1[:, 0] + diff))


    mask1 = match1 <= error
    mask2 = match2 <= error

    score_matrix = np.zeros((len(spec1), len(spec2)))

    for i in range(len(spec1)):
        for j in range(len(spec2)):
            if mask1[j, i]:
                score_matrix[i, j] = spec1[i, 1] * spec2[j, 1]
            elif mask2[j, i]:
                score_matrix[i, j] = spec1[i, 1] * spec2[j, 1]

    cost_matrix = -score_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    final_score = score_matrix[row_ind, col_ind].sum()  

    matched_fragments_count = len(row_ind)

    return final_score, matched_fragments_count

#Flink spectra similarity algorithm
def Flink(str1, str2, mz1_, mz2_, thresh=0.01,error=0.01, filter_pre=1, filter_50Da=1,filter_minmz=100,non_filter=[39.00518,68.99576,77.96552,78.99895,80.99576,82.96085,84.99067,92.99576,98.95577],nl_filter=[43.98983,79.95682]):
    spec1 = MSMSspe_pfas(str1, mz1_,thresh,filter_pre, filter_50Da,filter_minmz,non_filter,nl_filter,error)  # 解析第一个质谱字符串
    spec2 = MSMSspe_pfas(str2, mz2_, thresh,filter_pre, filter_50Da,filter_minmz,non_filter,nl_filter,error)  # 解析第二个质谱字符串
    if len(spec1) == 0 or len(spec2) == 0:  
        return 0, 0

    spec1[:, 1] = np.sqrt(spec1[:, 1]) / np.sqrt(np.sum(spec1[:, 1]))
    spec2[:, 1] = np.sqrt(spec2[:, 1]) / np.sqrt(np.sum(spec2[:, 1]))

    diff = mz2_ - mz1_  

    match1 = np.abs(spec2[:, 0][:, np.newaxis] - spec1[:, 0])
    match2 = np.abs(spec2[:, 0][:, np.newaxis] - (spec1[:, 0] + diff))

    mask1 = match1 <= error
    mask2 = match2 <= error

    score_matrix = np.zeros((len(spec1), len(spec2)))

    for i in range(len(spec1)):
        for j in range(len(spec2)):
            if mask1[j, i]:
                score_matrix[i, j] = spec1[i, 1] * spec2[j, 1]
            elif mask2[j, i]:
                score_matrix[i, j] = spec1[i, 1] * spec2[j, 1]

    cost_matrix = -score_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    final_score = score_matrix[row_ind, col_ind].sum()  

    matched_fragments_count = len(row_ind)

    return final_score, matched_fragments_count

#DP spectra similarity algorithm
def DP(str1,str2,mz1,mz2,error=0.01,filter_pre=1,filter_50Da=1):
    spec1 = MSMSspe(str1,mz1,filter_pre,filter_50Da) 
    spec2 = MSMSspe(str2,mz2,filter_pre,filter_50Da)  
    if(len(spec1)==0 or len(spec2)==0):
        return 0,0
    spec=spec2.copy() 
    alignment=np.zeros(shape=(len(spec1),3)) # caculate alignment matrix
    alignment[:,0:2]=spec1
    for i in range(len(spec1)):
        match=abs(spec[:,0]-spec1[i,0])
        if(min(match)<=error):
            alignment[i,2]=spec[np.argmin(match),1]
            spec=np.delete(spec,np.argmin(match),axis=0) # avoid of rematch
        if(len(spec)==0):
            break
    alignment=alignment[alignment[:,2]!=0]
    if(len(alignment)==0):
        return 0,0
    else:
        return np.dot(alignment[:,1],alignment[:,2])/np.sqrt(((np.dot(spec1[:,1],spec1[:,1]))*(np.dot(spec2[:,1],spec2[:,1])))),len(alignment)