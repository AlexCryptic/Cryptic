import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

drug_list = {"CRY2":["INH","RIF","EMB","LEV","MXF","AMI","KAN","ETH","RFB","BDQ","DLM","CFZ","LZD"],"CSEQ":["INH","RIF","EMB","LEV","MXF","AMI","KAN","ETH"]}
min_walker = {"AMI":1,"DLM":1,"RIF":1,"RFB":1,"INH":2,"KAN":2,"EMB":3,"ETH":3,"LEV":3,"MXF":3,"LZD":4,"BDQ":2,"CFZ":1}
min_ML = {"INH":2,"RIF":1,"EMB":3,"LEV":3,"MXF":3,"AMI":1,"KAN":2,"ETH":3,"RFB":1,"BDQ":2,"DLM":1,"CFZ":1,"LZD":4}

group_name = {"CRY2":"CRYPTIC"}

summary_list = []
for save_group in ["CRY2"]:
    final2 = pd.read_csv("FINAL/Final2_{}.csv".format(save_group)).fillna("-")
    for drug in drug_list[save_group]:
        df=final2[(final2.DRUG==drug)][["DRUG","PHENO_BIN",'STR_ML_f','STR_W','BIN_J_OR']]
        df_R=df[(df.PHENO_BIN=="R")]
        df_FP=df[(df.BIN_J_OR=="R")&(df.PHENO_BIN=="S")]
        df_TP=df[(df.BIN_J_OR=="R")&(df.PHENO_BIN=="R")]
        df_FN=df[(df.BIN_J_OR=="S")&(df.PHENO_BIN=="R")]
        df_TN=df[(df.BIN_J_OR=="S")&(df.PHENO_BIN=="S")]
        min_str = {}
        mode_str = {}
        a = df_FN[(df_FN["STR_ML_f"]>min_ML[drug])|(df_FN["STR_W"]>min_walker[drug])]
        b = df_FN[(df_FN["STR_ML_f"]>min_ML[drug])&(df_FN["STR_W"]>min_walker[drug])]
        c = df_TN[(df_TN["STR_ML_f"]>min_ML[drug])|(df_TN["STR_W"]>min_walker[drug])]
        print(drug,df_FN.shape[0],a.shape[0],round(a.shape[0]/df_FN.shape[0],2) )
        print(drug,df_TN.shape[0],c.shape[0],round(c.shape[0]/df_TN.shape[0],2) )

        newdata = {"GROUP":group_name[save_group],"DRUG":drug,"n":df.shape[0],"TP":df_TP.shape[0],"FP":df_FP.shape[0],"TN":df_TN.shape[0],"FN":df_FN.shape[0],"TN_MIC_increase":c.shape[0],"TN_Pct":round(c.shape[0]/df_TN.shape[0],3),"FN_MIC_increase":a.shape[0],"FN_Pct":round(a.shape[0]/df_FN.shape[0],3)}
        summary_list.append(newdata)
            
discrepancy = pd.DataFrame.from_dict(summary_list,orient="columns")
discrepancy.to_csv("FINAL/DISCREPANCY_MIC.csv")       
