import numpy as np
import pandas as pd
import statistics as stat
import math, time, itertools, glob, os, argparse

#from merf import MERF

from sklearn import linear_model, svm, ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix, accuracy_score, roc_auc_score, confusion_matrix, roc_curve, auc, f1_score
from sklearn.feature_selection import f_classif,  SelectKBest, chi2, f_regression
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV, LeavePGroupsOut
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
    parser.add_argument("-g","--group",type=str) #ALL500
    parser.add_argument("-a","--abxlist",type=str) #INH
    parser.add_argument("-n","--numbersamples",type=str) #0050
    parser.add_argument("-m","--methodlist",type=str) #"xgbr0"
    parser.add_argument("-c","--ncores", type=int) #4
    parser.add_argument("-f","--feature_type",type=str) #"KMER" or "MUTATIONS"
    parser.add_argument("-r","--run",type=str) #"Run_001"
    parser.add_argument("-t","--test_set",type=str) #"Test" or 4
    parser.add_argument("-p","--phenotype",type=str) # BIN or LOG
    parser.add_argument("-s","--percentage_remaining",type=str) #0030, 4 digits no period
    args = parser.parse_args()
    return(args)

    # return(0)

def define_variables(args):
    global groupname, drugname, numbersamples, method, ncores, feature_type, run, phenotype, test_set, seed, percentage_remaining
    global folderinput, folderoutput, directory_data, directory_weights, directory_results, radical, save_variable

    
    groupname = args.group
    drugname = args.abxlist
    numbersamples = args.numbersamples
    method = args.methodlist
    ncores= args.ncores
    feature_type = args.feature_type
    run= args.run
    phenotype= args.phenotype #BIN or LOG2MIC
    test_set=args.test_set
    percentage_remaining=args.percentage_remaining
    
    seed=40

    folderinput = feature_type+"_WILSON/"
    folderoutput= feature_type+"_RUNS/"+run
    directory_data= folderinput
    directory_results, directory_weights = folderoutput+"/Results/", folderoutput+"/Weights/"
    radical = "%s%s.%s."%(groupname,drugname,numbersamples)
    save_variable = "%s_%s_%s_%s_%s_%s_%s_%s"%(groupname,drugname,numbersamples,method.replace("_",""),feature_type,test_set,phenotype,run.replace("_",""))

def read_already_chewed_train_test_filtered (phenotype):
    #reads existing X_train_fs and X_test_fs files
    X_train_fs = pd.read_hdf(directory_data+radical+"Xtrainfs_{}_{}.h5".format(percentage_remaining,test_set))
    X_test_fs = pd.read_hdf(directory_data+radical+"Xtestfs_{}_{}.h5".format(percentage_remaining,test_set))
    Y_full_train = pd.read_csv(directory_data+radical+"Yfulltrain_{}.csv".format(test_set))
    Y_full_test = pd.read_csv(directory_data+radical+"Yfulltest_{}.csv".format(test_set))
    
    Correct_Y_column_names = {"BIN":drugname,"LOG":"Log"+drugname}
    labelname=Correct_Y_column_names[phenotype]

    Y_train = Y_full_train [[labelname]].values.ravel()
    Y_test = Y_full_test [[labelname]].values.ravel()
    return(X_train_fs, X_test_fs, Y_train, Y_test, Y_full_train, Y_full_test)

def pick_model(method):
    
    if method == 'rf':
        model = ensemble.RandomForestClassifier(n_estimators=100, max_features=None, n_jobs=ncores)
    elif method == 'rfreg':
        model = ensemble.RandomForestRegressor(n_estimators=100, max_features=None, n_jobs=ncores)
    elif method == 'gbt':
        model = ensemble.GradientBoostingClassifier(n_estimators=100, max_features=None)
    elif method == 'rgbt':
        model = ensemble.GradientBoostingRegressor(n_estimators=100, max_features=None)
    elif method == 'ridge': #regression models 
        model = linear_model.RidgeCV(alphas=[0.1,0.5],fit_intercept=True)
    elif method == 'lasso': #regression models 
        model = linear_model.Lasso(alpha=0.1)
    elif method == 'linreg': #regression models 
        model = linear_model.LinearRegression()
    elif method == 'LogAT': #regression models 
        model = LogisticAT() #LogisticIT()
        Y=Y_stratified
    elif method == 'elastic': #regression models 
        model = linear_model.ElasticNetCV(l1_ratio=[0.1,0.3,0.5], alphas=[0.1,0.5], cv=5)
    elif method == 'svr-poly': #regression models 
        model = svm.SVR(kernel='poly', gamma=0.1,C=1)
    elif method == 'xgbr0': #old kmer optimizaton
        model = XGBRegressor(objective="reg:squarederror", reg_lambda=0, min_child_weight= 4, learning_rate=0.05,n_estimators=200,max_depth=6,colsample_bytree=0.6, subsample=0.7,n_jobs=ncores)
    elif method == 'xgbr1':
        model = XGBRegressor(min_child_weight=3, learning_rate=0.01,n_estimators=200,max_depth=6,colsample_bytree=0.7, subsample=0.7,n_jobs=ncores)
    elif method == 'xgbr2':
        model = XGBRegressor(n_jobs=ncores)
    elif method == 'xgbr0mc': #old kmer optimizaton
        model = XGBClassifier(objective="multi:softmax",eval_metric="merror",num_class=7,n_jobs=ncores)
    elif method == 'merf':
        model = MERF(n_estimators=100, max_iterations=10, n_jobs =ncores)
    return(model)
    
def xgb_param_hyperopt():
    
    param1 = {
    "objective":"reg:squarederror", #reg:squarederror, #reg:squaredlogerror, #multi:softmax, multi:softprob, binary:hinge
    #"eval_metric":"rmse", #rmse, #rmsle, mae, error (bin error), 
        #merror (multiclass error, #(wrong cases)/#(all cases)), mlogloss (multiclass logloss)
    #https://xgboost.readthedocs.io/en/latest/parameter.html
    "nthread":ncores,
    "learning_rate":0.05,
    "n_estimators":200,
    #"early_stopping_rounds": 50,
    "max_depth":6, #high = risk overfitting
    "min_child_weight":4, #lower = risk overfitting
    "gamma":0, #minimum loss reducton to make a split; lower = risk overfitting
    "subsample":0.7, #samples to be used for each tree; higher = risk overfitting
    "colsample_bytree":0.6, #features to be used for each tree; higher = risk overfitting
    #"silent":1,
    "seed":seed,
    "reg_alpha": 0, #L1 regularization term on weights; used when very high dimensionality to run faster
    "reg_lambda:": 0 #L2 regularization term on weights; reduces overfitting
    }

    param2 = {
        "objectve":["reg:squarederror","reg:squaredlogerror","reg:pseudohubererror"]
        #"n_estimators":[100,200,300], #learn from 250 that 200 or 300 is best; average of 250 means 200 equal to 300
        #'max_depth': [3, 4], #average best is 3.7, so picking 4; 
        #'min_child_weight': [1,2], #learn from 250 that 1 is best; 
        #"learning_rate":[0.01,0.03,0.05], #learn from 250 that 0.01 is best, confirmed that 0.05 is better than 0.01
        #'colsample_bytree': [0.6,0.8], #average best is 0.65, so picking 0.6
        #'early_stopping_rounds': [10,30,50],
        #'gamma': [0,0.5, 1, 1.5, 2, 5], #average best is 2, so picking 2
        #'subsample': [0.6, 0.8], #average best is 0.75, so picking 0.8 
    }
    
    #param2 = {"n_estimators":list(np.arange(1,300))}
    
    #param2 = {"n_estimators":[50,100,200,300]}
    
    model = XGBRegressor(**param1)
    
    return (model, param2)


def custom_error_func(y_true, y_predicted):
    error = mean_squared_error(y_true, y_predicted)
    return error

def hyperopt(search_type, model, param):
    
    #my_scorer = make_scorer(custom_error_func, greater_is_better=True)
    #print(sorted(sklearn.metrics.SCORERS.keys()))
    #scoring = {'r2': 'r2', 'MSE': 'neg_mean_squared_error'}
    lpgo = LeavePGroupsOut(n_groups=2)
    if search_type == "random_search":
        search_parameters = RandomizedSearchCV(
            estimator=model, 
            param_distributions=param, 
            n_iter=2, 
            n_jobs=ncores,
            scoring="neg_mean_squared_error", 
            #refit='r2',
            #scoring=my_scorer,  
            cv=2,
            verbose=1, 
            random_state=1001,
            return_train_score=True)
    elif search_type == "grid_search":
        search_parameters = GridSearchCV(
            estimator=model, 
            param_grid=param, 
            n_jobs=ncores,
            scoring="neg_mean_squared_error", 
            #refit='r2',
            #scoring=my_scorer, "r2", "neg_mean_squared_error"
            cv=5,
            verbose=1, 
            #random_state=1001,
            return_train_score=True)
    return(search_parameters)

def save_hyperopt(df):
    df = df.sort_values(by="rank_test_score",ascending=True).reset_index(drop=True)
    df.to_csv(directory_results+"Hyperopt_Results__%s__.csv"%(save_variable))  

#INPUT VARIABLES
args=parse_arguments()

define_variables(args)

#read X and Y, and read train and test
X_train_fs, X_test_fs, Y_train, Y_test, Y_full_train, Y_full_test = read_already_chewed_train_test_filtered(phenotype)

hyperopt_model, hyperopt_param = xgb_param_hyperopt()
hyperopt_search = hyperopt("random_search",hyperopt_model,hyperopt_param)
search_result = hyperopt_search.fit(X_train_fs, Y_train)
hyperopt_results = pd.DataFrame(search_result.cv_results_)
save_hyperopt(hyperopt_results)






