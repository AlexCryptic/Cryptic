from __future__ import print_function, division
import numpy as np
import pandas as pd
import math, time, itertools, glob, os, argparse
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, roc_auc_score, confusion_matrix, roc_curve, auc, f1_score
from scipy.special import ndtri
from math import sqrt


import matplotlib.pyplot as plt
import seaborn as sns
plt.switch_backend('agg')

import matplotlib.pyplot as plt
import seaborn as sns
plt.switch_backend('agg')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("-i","--folderinputresults",type=str)
    parser.add_argument("-t","--datatype",type=str) #KMER or #MUTATIONS
    args = parser.parse_args()
    return(args)

def define_variables(args):
    global data_type, folderinputresults, folderoutput, pheno_directory, result_directory, analysis_directory
    global dico, CONCORDANCE, LABELS, LABELS_SPLIT, WALKER_PRED, Ecoff_lim, Ecoff_with_U, Ecoff_no_U, Ecoff
    data_type = args.datatype
    folderinputresults= "Output"
    folderoutput= str(folderinputresults)

    #DEFINE DIRECTORY
    pheno_directory = ""
    result_directory = folderinputresults+"/Results/"
    analysis_directory = folderoutput+"/Analysis/"

    #LOAD FILES
    dico = pd.read_csv("Supporting_Files/CONCORDANCE.csv")
    CONCORDANCE = dico
    #WALKER_PRED = pd.read_csv("WALKER_PRED.csv",usecols = ['uniqueid',"drug","predicted_stratification_new","predicted_binary"]).rename(columns={"uniqueid":"UNIQUEID","drug":"DRUG","predicted_stratification_new":"PRED_WALKER_STRAT","predicted_binary":"PRED_WALKER_BIN"})

    #CLEAN UP LABELS
    LABELS= pd.read_csv("Supporting_Files/LABELS.csv",low_memory=False)
    LABELS=LABELS[['UNIQUEID', 'SITEID', 'DRUG', 'MIC_NEW',
       'LOG2MIC_NEW', 'STRATIFICIATION_NEW', 'ECOFF','MIC',
       'LOG2MIC', 'BINARY_PHENOTYPE', 'PRED_CATAL','LINEAGE_NAME',
       'SUBLINEAGE_NAME','CATALOGUE_NAME']]
    #LABELS=LABELS[LABELS["SPLIT"]=="Test"]
    #LABELS=LABELS.merge(WALKER_PRED,on=["UNIQUEID","DRUG"],how="inner")
    LABELS.to_csv("test.csv")
    
    #LABELS = pd.read_pickle("LABELS.pkl")
    #LABELS= pd.read_csv("LABELS.csv",low_memory=False)
    

    #DEFINE VARIABLES
    Ecoff_lim = {"RIF":[-2,-2],"INH":[-4.32,-3.32],"EMB":[1,2],"AMI":[-1,0],"KAN":[2,3],"LEV":[-1,0],"MXF":[-1,-1],"ETH":[1,2],"RFB":[-2,-2],"LZD":[0,0],"BDQ":[-2,-2],"DLM":[-4.06,-4.06],"CFZ":[-2,-2]}
    Ecoff_with_U = {"RIF":-2,"INH":-4.32,"EMB":1,"AMI":-1,"KAN":2,"LEV":-1,"MXF":-1,"ETH":1,"RFB":-2,"LZD":0,"BDQ":-2,"DLM":-4.0,"CFZ":-26}
    Ecoff_no_U = {"RIF":-2,"INH":-3.32,"EMB":2,"AMI":0,"KAN":3,"LEV":0,"MXF":-1,"ETH":2,"RFB":-2,"LZD":0,"BDQ":-2,"DLM":-4.0,"CFZ":-26}
    Ecoff = Ecoff_no_U

def _proportion_confidence_interval(r, n):
    """Compute confidence interval for a proportion.
    References
    ----------
    [2] E. B. Wilson, Probable inference, the law of succession, and statistical inference,
    J Am Stat Assoc 22:209-12, 1927
    Follows notation described on pages 46--47 of [1].[1] R. G. Newcombe and D. G. Altman, Proportions and their differences, in Statisics
    with Confidence: Confidence intervals and statisctical guidelines, 2nd Ed., D. G. Altman, 
    D. Machin, T. N. Bryant and M. J. Gardner (Eds.), pp. 45-57, BMJ Books, 2000. 
    """

    #source: https://gist.github.com/maidens/29939b3383a5e57935491303cf0d8e0b
    
    alpha = 0.95
    z = -ndtri((1.0-alpha)/2)

    A = 2*r + z**2
    B = z*sqrt(z**2 + 4*r*(1 - r/n))
    C = 2*(n + z**2)
    #return ((A-B)/C, (A+B)/C)
    return ((A+B)/C-(A-B)/C)/2

def analysis_scores(approach, metric):
    Column = "Acc"+approach
    TP,TN,FP,FN = analysis[(analysis[Column]=="TP")].shape[0],analysis[(analysis[Column]=="TN")].shape[0],analysis[(analysis[Column]=="FP")].shape[0],analysis[(analysis[Column]=="FN")].shape[0]
    if TP == 0 or TN == 0:
        value, CI = '-','-'
    else:
        if metric == "Acc":
            value, CI = (TP+TN)/(TP+TN+FN+FP), _proportion_confidence_interval((TP+TN),(TP+TN+FN+FP))
        elif metric == "Sens":
            value, CI = TP/(TP+FN), _proportion_confidence_interval(TP, TP + FN)
        elif metric == "Spec":
            value, CI = TN/(TN+FP), _proportion_confidence_interval(TN, TN + FP)
        elif metric == "PPV":
            value, CI =  TP/(TP+FP), _proportion_confidence_interval(TP, TP + FP)
        elif metric == "NPV":
            value, CI = TN/(FN+TN), _proportion_confidence_interval(TN, FN + TN)
        elif metric == "F1":
            value, CI = TP/(TP+((FP+FN)/2)), _proportion_confidence_interval(TP,(TP+((FP+FN)/2)))
    return(value,CI)


def plot_confusion_matrix(cm, classes, xlabel, ylabel, name, normalize=False, cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(name + '.jpg', format='jpg', dpi=1000)
    plt.close()

def list_all_result_files():
    allFiles = glob.glob(result_directory+"Results*.csv") #Load list of all files to include in tables
    df=pd.DataFrame()
    df_cols = ["DRUG","SAMPLES","METHOD","DATA","TEST_SET","LABEL","RUN","PATH","SAVE_PREFIX"]
    df_lst = []
    for file in allFiles:
        save_prefix = file.split("Results_")[1].split(".csv")[0] #keep between "Results_" and ".csv"
        file_variables = save_prefix.split(sep="_") #create list of all terms of interest
        file_variables.append(file)#add path to list, to read_csv
        file_variables.append(save_prefix)
        df_lst.append(file_variables)
    df = pd.DataFrame(df_lst, columns=df_cols)
    replacements = {"LOG2MICNEW":"LOG2MIC_NEW","MICNEW":"MIC_NEW","ECOFF":"ECOFF"} #add back "_" in labels
    df["LABEL"] = df["LABEL"].map(replacements) #add back "_" in labels
    df["RUN"] = df["RUN"].str.replace("Run","Run_") #add back "_" in runs
    return(df)

def generate_and_save_analysis_table(index, row):
    time_start = time.time()

    wgs_pred_group  = [["S","R","U"],["S","R"],["U"]][j]
    drug, method, samples, labelname = row["DRUG"], row["METHOD"],row["SAMPLES"],row["LABEL"]
    save_prefix = row["SAVE_PREFIX"]+"_"+str(j)
    pheno = LABELS[LABELS["DRUG"]==drug]
    results = pd.read_csv(row["PATH"])

    analysis = results.merge(pheno,on="UNIQUEID",how="inner")

    if labelname == "ECOFF":
         #0 to 1
        analysis["Y_probs"] = analysis["Y_probs"]
        analysis["Y_truth_bin"] = analysis["Y_truth_bin"] #1 or 0

        analysis["Y_pred_bin"] = (analysis["Y_pred"] =='R')*1 # 1 or 0
        analysis["Y_truth"] = analysis["Y_truth"] #S or R
        analysis["Y_pred"] = analysis["Y_pred"] #S or R

        analysis["Bin_true"] = analysis["Y_truth"]
        analysis["Bin_pred"] = analysis["Y_pred"]
        analysis["Bin_catal"] = analysis["PRED_CATAL"]

        #if len(analysis["Bin_catal"]) != 0: #skip over times where no U 
        analysis["Acc_ml"] = np.where((analysis["Bin_pred"]=="R")&(analysis["Bin_true"]=="R"), 'TP', np.where((analysis["Bin_pred"]=="S")&(analysis["Bin_true"]=="S"), 'TN', np.where((analysis["Bin_pred"]=="S")&(analysis["Bin_true"]=="R"),'FN', np.where((analysis["Bin_pred"]=="R")&(analysis["Bin_true"]=="S"),'FP', '-'))))
        analysis["Acc_cat"] = np.where((analysis["Bin_catal"]=="R")&(analysis["Bin_true"]=="R"), 'TP', np.where((analysis["Bin_catal"]=="S")&(analysis["Bin_true"]=="S"), 'TN', np.where((analysis["Bin_catal"].isin(["S","U","F"]))&(analysis["Bin_true"]=="R"),'FN', np.where((analysis["Bin_catal"].isin(["R","U","F"]))&(analysis["Bin_true"]=="S"),'FP', '-'))))
       
        #define ordero f columns
        left_column = ["UNIQUEID"]
        acc_columns = [ col for col in df.columns if col.startswith("Acc")]
        bin_columns =  [ col for col in df.columns if col.startswith("Bin")]
        analysis = analysis [left_columns+acc_columns+bin_columns]

    elif labelname == "LOG2MIC_NEW":
        #analysis = pd.merge(analysis,CONCORDANCE,how="inner",left_on=["DRUG","LOG2MIC_NEW"],right_on=["DRUG","LOG2MIC_NEW"])
        

        #DEFINE Ys
        analysis["Y_label"] = analysis[labelname].round(2)
        analysis["Y_pred"] = analysis["Y_pred"].round(2)
        analysis["Y_delta"] = (analysis["Y_pred"]-analysis["Y_label"]).round(2)
        analysis["LOG2_true"] = analysis["LOG2MIC_NEW"]

        #FIND MIC_pred, Strat_Pred and Bin_pred using LOG2Pred
        #Also Fnd Bin_true using MIC_true
        dico_drug = dico[dico["DRUG"]==drug]
        a = dico_drug["LOG2MIC_NEW"].tolist()[::-1]
        b = dico_drug["MIC_NEW"].tolist()[::-1]
        c = dico_drug["STRATIFICIATION_NEW"].tolist()[::-1]
        d = dico_drug["ECOFF"].tolist()[::-1]
        
        LOG2_pred = []
        MIC_pred = []
        Strat_pred = []
        Ecoff_pred = []
        Ecoff_true = []
        for i in analysis['Y_pred'].tolist():
            closest_match = min(a, key=lambda x:abs(x-i))
            LOG2_pred.append(a[a.index(closest_match)])
            MIC_pred.append(b[a.index(closest_match)])
            Strat_pred.append(c[a.index(closest_match)])
            Ecoff_pred.append(d[a.index(closest_match)])
        for i in analysis['LOG2_true'].tolist():
            closest_match = min(a, key=lambda x:abs(x-i))
            Ecoff_true.append(d[a.index(closest_match)])
        
        analysis["LOG2_pred"] = LOG2_pred
        analysis["LOG2_delta"] = (analysis["LOG2_pred"]-analysis["LOG2_true"]).round(2)
        analysis["MIC_true"] = analysis["MIC_NEW"]
        analysis["MIC_pred"] = MIC_pred
        analysis["Strat_true"] = analysis["STRATIFICIATION_NEW"]
        analysis["Strat_pred"] = Strat_pred
        analysis["Strat_delta"] = abs(analysis["Strat_pred"]-analysis["Strat_true"])
    
        #analysis["Bin_pred"] = np.where((analysis["LOG2_pred"]<=(Ecoff_lim[drug][0])),"S",np.where((analysis["LOG2_pred"]>Ecoff_lim[drug][1]),"R",np.where((analysis["LOG2_pred"]<=(Ecoff_lim[drug][0]+0.3)),"S","R")))
        
        #analysis["Bin_true"] = np.where((analysis["LOG2_true"]<=Ecoff_lim[drug][0]),"S",np.where((analysis["LOG2_true"]>Ecoff_lim[drug][1]),"R","U"))
        
        
        ##########################JUICY STUFF UNDER THIS LINE ##########

        analysis["Bin_true"] = analysis["ECOFF"]
        analysis["Bin_pred"] = Ecoff_pred
        analysis["Bin_catal"] = analysis["PRED_CATAL"]

        if plot_MIC=="ok":
            MIC_CM = confusion_matrix(analysis["MIC_true"],analysis["MIC_pred"],labels=b)
            plot_confusion_matrix(MIC_CM,b,'Predicted MIC (mg/ml)','Actual MIC (mg/ml)',analysis_directory+"CM_MIC_"+save_prefix+".csv")    

            Bin_CM = confusion_matrix(analysis["Bin_true"],analysis["Bin_pred"],labels=["S","R"])
            plot_confusion_matrix(Bin_CM,["S","R"],'Predicted','Actual',analysis_directory+"CM_Bin_"+save_prefix+".csv")  
      

        #if len(analysis["Bin_catal"]) != 0: #skip over times where no U 
        analysis["Acc_ml"] = np.where((analysis["Bin_pred"]=="R")&(analysis["Bin_true"]=="R"), 'TP', np.where((analysis["Bin_pred"]=="S")&(analysis["Bin_true"]=="S"), 'TN', np.where((analysis["Bin_pred"]=="S")&(analysis["Bin_true"]=="R"),'FN', np.where((analysis["Bin_pred"]=="R")&(analysis["Bin_true"]=="S"),'FP', '-'))))
        analysis["Acc_cat"] = np.where((analysis["Bin_catal"]=="R")&(analysis["Bin_true"]=="R"), 'TP', np.where((analysis["Bin_catal"]=="S")&(analysis["Bin_true"]=="S"), 'TN', np.where((analysis["Bin_catal"].isin(["S","U","F"]))&(analysis["Bin_true"]=="R"),'FN', np.where((analysis["Bin_catal"].isin(["R","U","F"]))&(analysis["Bin_true"]=="S"),'FP', '-'))))

        ##########################JUICY STUFF ABOVE THIS LINE ##########

        analysis["PA_ml"] = ((analysis["Strat_delta"]==0))*1 
        analysis["EA_ml"] = ((analysis["Strat_delta"]<=1))*1 
        analysis["CA_ml"] = ((analysis["Acc_ml"]!="FN")&(analysis["Acc_ml"]!="FP"))*1


        analysis = analysis[["UNIQUEID","Acc_ml","Acc_cat",\
        "PA_ml","EA_ml","CA_ml",\
        "Y_label","Y_pred","Y_delta",\
        "LOG2_true","LOG2_pred","LOG2_delta",\
        "MIC_true","MIC_pred",\
        "Strat_true","Strat_pred","Strat_delta",\
        "Bin_true","Bin_pred","Bin_catal"]]  
    
    analysis.to_csv(analysis_directory+"Analysis_"+save_prefix+".csv")
    return(analysis)



def fill_summary_table(analysis, df):
   
    df.loc[index,'Samples'] = format(int(analysis.shape[0]),".0f") 
    
    if "CA_ml" in analysis.columns: #check to make sure it's not binary
        for metric in ["EA","PA","CA"]:
            for approach in ["_ml",]: #first is ml, second s walker
                metric_name = metric+approach #add ending if ml(nothing) or walker ("_w")
                if analysis.shape[0] == 0:
                    df.loc[index,metric_name] = "-"
                    df.loc[index,str(metric_name+"CI")] = "-"
                elif analysis.shape[0] != 0:
                    metric_value = analysis[(analysis[metric_name]==1)].shape[0]/analysis.shape[0]
                    CI = _proportion_confidence_interval(analysis[(analysis[metric_name]==1)].shape[0], analysis.shape[0])
                    df.loc[index,metric_name] = format(metric_value,".3f")
                    df.loc[index,str(metric_name+"CI")] = format(CI,".3f")
    for metric in ["Sens","Spec","PPV","NPV","Acc","F1"]:
        for approach in ["_ml","_cat"]:
            metric_name = metric+approach
            metric_value, CI = analysis_scores(approach, metric)
            if metric_value == "-":
                df.loc[index,metric_name] = '-'
                df.loc[index,str(metric_name+"CI")] = '-'
            elif metric_value != "-":
                df.loc[index,metric_name] = format(metric_value,".3f")
                df.loc[index,str(metric_name+"CI")] = format(CI,".3f")

    df.loc[index,'S_true'] = int(analysis[analysis["Bin_true"]=="S"].shape[0])
    df.loc[index,'R_true'] = int(analysis[analysis["Bin_true"]=="R"].shape[0])
    df.loc[index,'S_ml'] = int(analysis[analysis["Bin_pred"]=="S"].shape[0])
    df.loc[index,'R_ml'] = int(analysis[analysis["Bin_pred"]=="R"].shape[0])
    
    return(df)

def save_summary_table(df):
    #define order of columns
    df["TEST_SET"]=df["TEST_SET"].apply(str).str.zfill(2)
    left_columns = ["DRUG","METHOD","TEST_SET","Samples"]
    right_columns = ["S_true","R_true","S_ml","R_ml","LABEL","SAMPLES","DATA","RUN"]
    middle_columns_CI = [ col for col in df.columns if "CI" in col ]
    remove_columns = ["SAVE_PREFIX","PATH"]
    middle_columns_values = [item for item in list(df) if item not in left_columns+right_columns+middle_columns_CI+remove_columns]
    
    #reorder columnss
    df = df[left_columns+middle_columns_values+middle_columns_CI+right_columns]
    
    #order rows by drug
    #df["DRUG"] = pd.Categorical(df["DRUG"],["INH","RIF","RFB","EMB","MXF","LEV","AMI","KAN","ETH","BDQ","CFZ","DLM","LZD"])

    #order rows after drug if needed
    #df = df.sort_values(by=["METHOD",'TEST_SET',"SAMPLES","EA_ml",'DRUG']).reset_index(drop=True)
    df = df.sort_values("DRUG").reset_index(drop=True)
    df.to_csv(analysis_directory+"Summary_table.csv")


args=parse_arguments()

define_variables(args)

plot_MIC="notok"#"ok"
       
for j in [0]:
    df = list_all_result_files()
    for index, row in df.iterrows():
        analysis=generate_and_save_analysis_table(index,row)
        df=fill_summary_table(analysis,df)
    save_summary_table(df)
        
    
#