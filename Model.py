import numpy as np
import pandas as pd
import statistics as stat
import math, time, itertools, glob, os, argparse
from numpy import array, count_nonzero
from sklearn import linear_model, svm, ensemble, neural_network
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix, accuracy_score, roc_auc_score, confusion_matrix, roc_curve, auc, f1_score
from sklearn.feature_selection import f_classif,  SelectKBest, chi2, f_regression
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV
import xgboost as xgb
from xgboost.sklearn import XGBClassifier, XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
plt.switch_backend('agg')

n=0.03

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("-a","--abxlist",type=str) #INH
    parser.add_argument("-n","--numbersamples",type=str) #0050
    parser.add_argument("-m","--methodlist",type=str) #"xgbr0"
    parser.add_argument("-c","--ncores", type=int) #4
    parser.add_argument("-f","--feature_type",type=str) #"KMER" or "MUTATIONS"
    parser.add_argument("-r","--run",type=str) #"Run_001"
    parser.add_argument("-t","--test_set",type=str) #"Test" or 4
    parser.add_argument("-p","--phenotype",type=str) #"LOG2MIC_NEW
    args = parser.parse_args()
    return(args)

def define_variables(args):
    global drugname, numbersamples, method, ncores, feature_type, run, phenotype, test_set, seed
    global folderinput, folderoutput, directory_data, directory_weights, directory_results, radical, save_variable
    
    drugname = args.abxlist
    numbersamples = args.numbersamples
    method = args.methodlist
    ncores= args.ncores
    feature_type = args.feature_type
    run= args.run
    phenotype= args.phenotype
    test_set=args.test_set
    if test_set != "Test":
        test_set = int(test_set)

    seed=40

    folderinput = "Mock_data/"
    folderoutput= "Output"
    directory_data= folderinput
    directory_results, directory_weights = folderoutput+"/Results/", folderoutput+"/Weights/"
    radical = "%s.%s."%(drugname,numbersamples)
    save_variable = "%s_%s_%s_%s_%s_%s_%s"%(drugname,numbersamples,method.replace("_",""),feature_type,test_set,phenotype.replace("_",""),run.replace("_",""))


def define_Y(labelname):
    global Y_full, Y_LOG2MIC_NEW, Y_BINARY, Y_stratified, Y_lineage, Y_split
    Y_full = pd.read_csv(directory_data+radical+"labels.csv")
    Y_LOG2MIC_NEW = Y_full[["LOG2MIC_NEW"]]
    Y_BINARY = Y_full[["BINARY_PHENOTYPE"]]
    Y_stratified = Y_full[["STRATIFICIATION_NEW"]]
    Y_full["LINEAGE_NAME"] = Y_full["LINEAGE_NAME"].str[-1:]#.astype(int)
    Y_lineage = Y_full["LINEAGE_NAME"] 
    Y_split = Y_full["SPLIT"] 
    
    Y = Y_full [[labelname]]
    
    return(Y, Y_full, Y_stratified)
    
def define_X():
    global X
    if feature_type == "MUTATIONS":
        X = pd.read_hdf(directory_data+"%s.%s.matrix.h5"%(drugname,numbersamples)).rename_axis("FEATURE", axis="columns").rename_axis("SAMPLE", axis="index")
    elif feature_type == "KMER":
        X = pd.read_hdf(directory_data+radical+"features.h5").rename_axis("FEATURE", axis="columns").rename_axis("SAMPLE", axis="index")

    X = X.reset_index(drop=False)
    X.set_index("SAMPLE",inplace=True)
    
    return(X)
    

def read_train_test(X,Y,Y_full,test):
    #feed either "Test" or the site to test on as "test"
    global train_index, test_index
    if test == "Test":
        train_index = Y_full.index[Y_full['SPLIT'] == "Train"].tolist()
        test_index = Y_full.index[Y_full['SPLIT'] == "Test"].tolist()
    else:
        train_index = Y_full.index[Y_full['SITEID'] != test].tolist()
        test_index = Y_full.index[Y_full['SITEID'] == test].tolist()
    X_train, X_test, Y_train, Y_test, Y_strat_train, Y_strat_test, Y_full_train, Y_full_test = X.iloc[train_index], X.iloc[test_index], Y.iloc[train_index].values.ravel(), Y.iloc[test_index].values.ravel(),Y_stratified.iloc[train_index].values.ravel(), Y_stratified.iloc[test_index].values.ravel(),Y_full.iloc[train_index], Y_full.iloc[test_index]
    return(X_train, X_test, Y_train, Y_test, Y_strat_train, Y_strat_test,Y_full_train, Y_full_test)

def pick_model(method):#
    
    if method == 'xgbr': #old kmer optimizaton
        model = XGBRegressor(objective="reg:squarederror", reg_lambda=0, min_child_weight= 4, learning_rate=0.05,n_estimators=100,max_depth=6,colsample_bytree=0.5, subsample=0.7,n_jobs=ncores)
    return(model)

def pick_cv_method(method):
    if method == "stratk":
        model=StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    return(model)
    
def fit_model (method, model):
    if method == 'merf':
        model.fit(X_train, pd.DataFrame(data = np.ones([len(Y_train),1])), pd.Series(Y_lineage_train), pd.Series(Y_train))
    else:
        model.fit(X_train,Y_train)

def predict_test (method):
    if method == 'merf':
        Y_pred = model.predict(X_test,pd.DataFrame(data = np.ones([len(Y_test),1])), pd.Series(Y_lineage_test))  
    else:
        Y_pred = model.predict(X_test)
    return(Y_pred)
    


def save_results(Y_pred):
    TestTable = pd.DataFrame({"Drug":drugname,"Method":method,"CV":"NA","Y_truth":Y_test,"Y_pred":Y_pred.round(2)},columns=["Drug","Method","CV","Y_truth","Y_pred"])
    TestTable["UNIQUEID"] = Y_full.iloc[test_index]["UNIQUEID"].tolist()
    TestTable.to_csv(directory_results+"Results__%s__.csv"%(save_variable))

def select_features(X_train, Y_train, X_test, percentage_remaining):
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
    return X_train_fs, X_test_fs, fs
    #get the scores for each feature with "fs.scores_"
    #great source for code


def run_filtering_cv(X, Y, Y_full, Y_stratified, save_variable):
    #df, df_cols, df_lst = pd.DataFrame(), ["k","feat_filter","feat_total","PA","std","EA","std","MSA","std","duration_per_iteration","std"],[]
    cv_methodology = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    cv = cv_methodology.split(X, Y_stratified)
    ConcatTable, scores, k = pd.DataFrame(), [], 1
    for train_index, test_index in cv:
        X_train, X_test, Y_train, Y_test = X.iloc[train_index], X.iloc[test_index], Y.iloc[train_index].values.ravel(), Y.iloc[test_index].values.ravel()
        Y_train_cv, Y_test_cv = Y_train, Y_test
        X_train_cv, X_test_cv, fs = select_features(X_train, Y_train, X_test, n)
        model = pick_model(method)
        model.fit(X_train_cv,Y_train_cv)
        Y_pred_cv = model.predict(X_test_cv)
        TestTable = pd.DataFrame({"Drug":drugname,"Method":method,"CV":k,"Y_truth":Y_test_cv,"Y_pred":Y_pred_cv},columns=["Drug","Method","CV","Y_truth","Y_pred"])
        TestTable["UNIQUEID"] = Y_full.iloc[test_index.tolist()]["UNIQUEID"].tolist()
        ConcatTable = pd.concat([TestTable,ConcatTable]).sort_values(by="UNIQUEID", ascending=True)
        k=k+1 
    ConcatTable.reset_index(drop=True).to_csv(directory_results+"Results_%s.csv"%(save_variable))

#INPUT VARIABLES
args=parse_arguments()

define_variables(args)

#read X and Y, and read train and test##
X = define_X()
Y, Y_full, Y_stratified = define_Y(phenotype)

run_filtering_cv(X, Y, Y_full, Y_stratified, save_variable)

