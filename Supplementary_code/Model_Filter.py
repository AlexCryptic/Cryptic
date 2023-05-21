import numpy as np
import pandas as pd
import statistics as stat
import math, time, itertools, glob, os, argparse

#from merf import MERF#

from sklearn import linear_model, svm, ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix, accuracy_score, roc_auc_score, confusion_matrix, roc_curve, auc, f1_score
from sklearn.feature_selection import VarianceThreshold, f_classif,  SelectKBest, chi2, f_regression
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV
import xgboost as xgb
from xgboost.sklearn import XGBClassifier, XGBRegressor

import matplotlib.pyplot as plt
import seaborn as sns
plt.switch_backend('agg')

#################################################
#################################################
#################################################

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("-a","--abxlist",type=str) #INH
    parser.add_argument("-n","--numbersamples",type=str) #0050
    parser.add_argument("-m","--method",type=str) #0050
    parser.add_argument("-c","--ncores", type=int) #4
    parser.add_argument("-f","--feature_type",type=str) #"KMER" or "MUTATIONS"
    parser.add_argument("-t","--test_set",type=str) #"Test" or 4
    parser.add_argument("-p","--phenotype",type=str) #"LOG2MIC_NEW
    parser.add_argument("-s","--percentage_remaining",type=float) #"LOG2MIC_NEW
    parser.add_argument("-d","--druglist",type=str)
    args = parser.parse_args()
    return(args)

    # return(0)

def define_variables(args):
    global drugname, numbersamples, ncores, feature_type, phenotype, test_set, seed, percentage_remaining, method, druglist
    global folderinput, directory_data, directory_weights, directory_results, radical, save_variable

    
    drugname = args.abxlist
    numbersamples = args.numbersamples
    ncores= args.ncores
    method= args.method
    feature_type = args.feature_type
    phenotype= args.phenotype
    test_set=args.test_set
    percentage_remaining=args.percentage_remaining
    druglist=args.druglist

    seed=40

    folderinput = feature_type+"_WILSON/"
    folderoutput= feature_type+"_WILSON/"


    directory_data= folderinput
    radical = "%s.%s."%(drugname,numbersamples)
    
def define_Y():
    Y_full = pd.read_csv(directory_data+radical+"Yfull.csv")
    return(Y_full)

def read_train_test(X,Y_full,test):
    #feed either "Test" or the site to test on as "test"
    if test == "SplitOld":
        train_index = Y_full.index[Y_full['SPLIT'] == "Train"].tolist()
        test_index = Y_full.index[Y_full['SPLIT'] == "Test"].tolist()
    elif test == "SplitSite": #Sites 10 and 2
        train_index = Y_full.index[Y_full['SPLIT_SITE'] == "Train"].tolist()
        test_index = Y_full.index[Y_full['SPLIT_SITE'] == "Test"].tolist()
    elif test == "SplitRand":
        train_index = Y_full.index[Y_full['SPLIT_RAND'] == "Train"].tolist()
        test_index = Y_full.index[Y_full['SPLIT_RAND'] == "Test"].tolist()
    elif isinstance(int(test),int) == True:
        site_name = "site.{:02d}".format(int(test))
        train_index = Y_full.index[Y_full['SITE_NAME'] != site_name].tolist()
        test_index = Y_full.index[Y_full['SITE_NAME'] == site_name].tolist()
    else:
        test_set = int(test_set)
        train_index = Y_full.index[Y_full['SITEID'] != test].tolist()
        test_index = Y_full.index[Y_full['SITEID'] == test].tolist()
    X_train, X_test,Yfull_train, Yfull_test = X.iloc[train_index], X.iloc[test_index],Y_full.iloc[train_index], Y_full.iloc[test_index]
    return(X_train, X_test,Yfull_train, Yfull_test)
    
def define_X():
    if feature_type == "MUTATIONS":
        X = pd.read_hdf(directory_data+radical+"matrix.h5").rename_axis("FEATURE", axis="columns").rename_axis("SAMPLE", axis="index")
    elif feature_type == "KMER":
        X = pd.read_hdf(directory_data+radical+"features.h5").rename_axis("FEATURE", axis="columns").rename_axis("SAMPLE", axis="index")

    X = X.reset_index(drop=False)
    X.set_index("SAMPLE",inplace=True)
    
    return(X)
    




def remove_useless_features(X):
    a = X.T.to_numpy()
    X_fs = pd.DataFrame(a[(~(a==0).all(1))&(~(a==1).all(1))].T)
    fs = pd.DataFrame(X.columns[(~(a==0).all(1))&(~(a==1).all(1))]).rename(columns={'FEATURE':0})
    #fs = VarianceThreshold()
    #fs.fit(X)
    #X_fs = pd.DataFrame(fs.transform(X))
    return X_fs, fs

def select_features(X_train, y_train, X_test, percentage_remaining):
    import warnings
    warnings.filterwarnings("ignore")
    full_features = X_train.shape[1]
    goal_features = int(full_features*percentage_remaining)
    # configure to select all features
    fs = SelectKBest(score_func=f_regression, k=goal_features)
    # learn relationship from training data
    fs.fit(X_train, Y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    #save fs as dataframe
    fs = pd.DataFrame(fs.get_support(indices=True))
    fs.columns = [0]
    return X_train_fs, X_test_fs, fs
    #get the scores for each feature with "fs.scores_"
    #great source for code

def join_fs(fs_clean, fs_filtered):
    df = pd.merge(fs_filtered, fs_clean,  how="inner", left_on=0, right_index=True,).reset_index(drop=False).drop(columns=["0_x"]).rename(columns={'index':"Filtered_index",'key_0':'Intermediate_Index','0_y':'Pattern_Index'})
    return(df)
    
def save_filtered_train_test_files(drug,X_drug_clean, X_train_fs, X_test_fs, Y_full_drug, Y_full_train, Y_full_test, fs, percentage_remaining):
    test_set_name = str(test_set)
    pathway = directory_data
    drugradical="%s%s.%s."%(drugname,drug,numbersamples)

    percentage_remaining_name = str(str(percentage_remaining).ljust(5,"0")).replace(".","")
    
    loc_X_train_fs = ("{}{}Xtrainfs_{}_{}.h5".format(pathway,drugradical,percentage_remaining_name,test_set_name))
    loc_X_test_fs = ("{}{}Xtestfs_{}_{}.h5".format(pathway,drugradical,percentage_remaining_name,test_set_name))
    loc_Y_full_drug = ("{}{}Yfulldrug_{}.csv".format(pathway,drugradical,test_set_name))
    loc_Y_full_train = ("{}{}Yfulltrain_{}.csv".format(pathway,drugradical,test_set_name))
    loc_Y_full_test = ("{}{}Yfulltest_{}.csv".format(pathway,drugradical,test_set_name))
    loc_filter_map = ("{}{}Xfiltermap_{}_{}.csv".format(pathway,drugradical,percentage_remaining_name,test_set_name))

    Y_full_train.to_csv(loc_Y_full_train)
    Y_full_test.to_csv(loc_Y_full_test)
    Y_full_drug.to_csv(loc_Y_full_drug)

    fs.to_csv(loc_filter_map)

    store_X_train_fs = pd.HDFStore(loc_X_train_fs,complevel=9, complib='blosc')
    store_X_train_fs['X_train_fs'] = pd.DataFrame(X_train_fs)
    store_X_train_fs.close()

    store_X_test_fs = pd.HDFStore(loc_X_test_fs,complevel=9, complib='blosc')
    store_X_test_fs['X_test_fs'] = pd.DataFrame(X_test_fs)
    store_X_test_fs.close()

    

    



############################
###########################
args=parse_arguments()

define_variables(args)

#read X and Y, and read train and test
X = define_X()
Y_full = define_Y()

for drug in ["INH","RIF","MXF"]:#["INH","RIF","MXF","LEV","AMI","KAN","ETH","RFB","BDQ","DLM","CFZ","LZD"]:
    drug_index = Y_full[Y_full[drug].isin(["S","R"])].index
    #define the startng matrices
    X_drug = X.loc[drug_index,:].reset_index(drop=True)
    Yfull_drug = Y_full.loc[drug_index,:].reset_index(drop=True)
    #remove features that are present or absent in all samples
    X_drug_clean, fs1 = remove_useless_features(X_drug)
    #define train test matrices
    X_train, X_test,Yfull_train, Yfull_test=read_train_test(X_drug_clean,Yfull_drug,test_set)
    print(X, X_drug, X_drug_clean,X_train)
    Y_train = Yfull_train["Log"+drug].astype(float)
    #feature selection on train test
    X_train_fs, X_test_fs, fs2 = select_features(X_train, Y_train, X_test, percentage_remaining)
    #merge both fs for later
    fs_merged = join_fs(fs1,fs2)
    #save files
    save_filtered_train_test_files(drug,X_drug_clean, X_train_fs, X_test_fs, Yfull_drug,Yfull_train, Yfull_test, fs_merged, percentage_remaining)
