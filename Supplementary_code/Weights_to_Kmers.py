import pandas as pd
import glob
import os,time, argparse, pickle, math
import math, time, itertools, glob, os, argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("-i","--folderinputresults",type=str)
    parser.add_argument("-t","--datatype",type=str) #KMER or #MUTATIONS
    args = parser.parse_args()
    return(args)

def define_variables(args):
    global data_type, loaded_index_dic, weight_directory, run
    data_type = args.datatype
    run = str(args.folderinputresults)
    weight_directory= "{}_RUNS/{}/Weights/".format(data_type,run)

   
def list_all_result_files():
    global epic
    allFiles = glob.glob(weight_directory+"TopWeights_*.csv") #Load list of all files to include in tables
    df=pd.DataFrame()
    df_cols = ["EPIC","DRUG","METHOD","DATA","TEST_SET","LABEL","PATH"]
    df_lst = []
    for file in allFiles:
        save_prefix = file.split("TopWeights_")[1].split("__.csv")[0] #keep between "Results_" and ".csv"
        file_variables = save_prefix.split(sep="_") #create list of all terms of interest
        epic, drug, method,data,test_set, label = file_variables[0],file_variables[1],file_variables[3],file_variables[4],file_variables[5],file_variables[6]
        df_lst.append([epic, drug, method,data,test_set, label,file])
    epic = epic
    df = pd.DataFrame(df_lst, columns=df_cols)
    return(df, epic)

def generate_and_save_kmer_table(index,row):

    epic,drug, test_set,method,data,label= row["EPIC"],row["DRUG"],row["TEST_SET"],row["METHOD"],row["DATA"],row["LABEL"]

    weights = pd.read_csv(row["PATH"]).rename(columns={"Unnamed: 0":"Rank"})

    patternmap_director = "/well/bag/lachap/allmodels/KMER_WILSON/"
    patternmap = pd.read_csv(patternmap_director+"{}{}.Full.Xfiltermap_0030_{}.csv".format(epic,drug,test_set), index_col=0)

    updated_weights = weights.merge(patternmap,how="inner",on="Filtered_index")
    top_patterns = [i for i in updated_weights["Pattern_Index"]]


    df_cols = ["Pattern_Index","Kmer"]
    df_lst = []

    for pattern in top_patterns[0:100]:
        for i in loaded_index_dic[pattern]:
            if i==0:
                n_min,n_max,n_index  = 0,99999,i
            else:
                n_max = int(math.ceil(i/100000))*100000-1
                n_min = n_max-99999
                n_index = i-n_min
            x = pd.read_csv('KMER_WILSON/KMER_LISTS/{}/{}.Full_kmerlist_{}_{}.txt'.format(epic,epic,n_min,n_max), header=None)
            kmer = x.loc[n_index,0]
            df_lst.append([pattern,kmer])
            

    df = pd.DataFrame(df_lst, columns=df_cols)
    df = df.merge(updated_weights, how="inner",on="Pattern_Index")
    df = df[["Rank","Kmer","Weight","Pattern_Index","Intermediate_Index","Filtered_index"]]

    df.to_csv(weight_directory+"TopWeightsKmers_{}_{}_Full_{}_{}_{}_{}_{}.csv".format(epic,drug,method,data, test_set,label, run))



args=parse_arguments()

define_variables(args)

df, epic = list_all_result_files()

with open('KMER_WILSON/{}.Full.index_dic.pickle'.format(epic), 'rb') as handle:
        loaded_index_dic = pickle.load(handle)

print("ok biatch")

for index, row in df.iterrows():
    generate_and_save_kmer_table(index,row)
    print("hi")