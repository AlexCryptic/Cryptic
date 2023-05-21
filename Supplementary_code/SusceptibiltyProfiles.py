import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import itertools

method_name = {"BIN_J_OR":"SYSTEM"}
group_name = {"CRY2":"CRYPTIC"}

group_A1 = ["LEV","BDQ","LZD"]
group_A2 = ["MXF","BDQ","LZD"]
group_B = ["CFZ"]
group_C = ["EMB","DLM","AMI","ETH"]
n = 2
m = 1
o = 3

regimen_list_default = [["MXF","LEV","AMI"],["MXF","BDQ","LZD"],["MXF","BDQ","LZD","CFZ"]]

number_list = []
regimen_list = []
breakdown_list = []
a=1
for n in [3,2,]:
    for m in [1,0]:
        if n == 3 and m == 1:
            o = 0
        elif n == 3 and m == 0:
            o = 1
        elif n == 2 and m == 1:
            o = 1
        elif n == 2 and m == 0:
            o = 3
        elif n == 1 and m == 1:
            o = 3
        elif n == 1 and m == 0:
            o = 4
        elif n == 0 and m == 1:
            o = 4
        elif n == 0 and m == 0:
            o = 4
        permut_A1 = list(itertools.combinations(group_A1, n))
        permut_A2 = list(itertools.combinations(group_A2, n))
        permut_B = list(itertools.combinations(group_B, m))
        permut_C = list(itertools.combinations(group_C, o))
        
        
        for i in [permut_A2]:
            for j in i:
                for k in permut_B:
                    for l in permut_C:
                        regimen = [x for x in j]+[x for x in k]+[x for x in l]
                        regimen_list.append(regimen)
                        breakdown_list.append("{}A, {}B, {}C".format(n,m,o))
                        number_list.append(a)
                        a=a+1
                        
                        if "MXF" in regimen:
                            regimen = ["LEV" if x=="MXF" else x for x in regimen]
                            regimen_list.append(regimen)
                            breakdown_list.append("{}A, {}B, {}C".format(n,m,o))
                            number_list.append(a)
                            a=a+1
                        
regimen_df = pd.DataFrame(
    {'Number': number_list,
     'Regimen': regimen_list,
     'Breakdown': breakdown_list
    })

regimen_df.to_csv("FINAL/Regimens_list.csv", index=False)
regimen_list = list(regimen_df["Regimen"])

##############################

for save_group in ["CRY2"]:
    for method in ["BIN_J_OR"]:

       
        final = pd.read_csv("FINAL/Final2_{}.csv".format(save_group)).fillna("-").rename(columns={method: "G", "PHENO_BIN": "P"})
        n = final.drop_duplicates(subset="UNIQUEID").shape[0]
        #final = final[final.RR==1] #orgnal
        final = final.merge(final[(final.DRUG=="RIF")&(final["G"]=="R")][["UNIQUEID"]], on="UNIQUEID", how="inner")
        print(final.drop_duplicates(subset="UNIQUEID").shape[0])
        df =pd.pivot_table(final, values=["G","P"], index=['UNIQUEID'],columns=['DRUG'], aggfunc=np.sum,fill_value="-")
        n_RR = df.shape[0]
        pct_RR = round(n_RR/n,3)
        col_name = []
        for i in df.columns:
            if i[0]=="G":
                name = i[0]+"_"+i[1]
            elif i[0]=="P":
                name = i[0]+"_"+i[1]
            col_name.append(name)
        df.columns = col_name
        df_dupl = df.copy()

        for regimen in regimen_list:
            
            regimen_number = regimen_list.index(regimen)+1
            regimen_breakdown = regimen_df.loc[regimen_df['Number'] == regimen_number, 'Breakdown'].iloc[0]
    
            y=pd.DataFrame()
            
            #add columns for phenotypic and genotypic strings (P and G)
            String_P_regimen_name = "P"+str(regimen_number) #Pheno string
            String_G_regimen_name = "G"+str(regimen_number) #Geno string
            String_a_regimen_name = "a"+str(regimen_number) #are they the same (accuracy)
            String_s_regimen_name = "s"+str(regimen_number) #if G string is only S, are they the same
            String_w_regimen_name = "w"+str(regimen_number) #if P string is only S (to check if some are just impossible)
            String_x_regimen_name = "x"+str(regimen_number) #if P string is only S (to check if some are just impossible)
            
            for i in ["P_","G_",]:
                x = pd.DataFrame()
                for drug in regimen:
                    x[i+drug] = df[i+drug]
                x['string'] = x.apply(''.join, axis=1)
                dico = {"P_":String_P_regimen_name,"G_":String_G_regimen_name}
                y[dico[i]]=x['string']
                df[dico[i]]=x['string']
            
            #add column for accuracy of genotypc strng
            df[String_a_regimen_name] = np.where(y[String_G_regimen_name].str.contains("-"),"-",np.where(y[String_G_regimen_name]==y[String_P_regimen_name],1,0))
            df[String_s_regimen_name] = np.where(y[String_G_regimen_name]!="S"*(len(regimen)),"-",np.where(y[String_G_regimen_name]==y[String_P_regimen_name],1,0))
            df[String_w_regimen_name] = np.where(y[String_P_regimen_name]=="S"*(len(regimen)),"1","-")
            df[String_x_regimen_name] = np.where(~y[String_P_regimen_name].str.contains("-"),"1","-")


##CHECK PAN S PHENO
col_filter = list([i for i in df.columns if "s" in i])
dfz = df[col_filter]
summary_list = []
for index, row in dfz.iterrows():
    PanSRegimens = list(row[row.isin(["0","1"])].index)
    PanSAccuracy = list(row[row.isin(["0","1"])])
    if len(PanSRegimens)>0:
        BestSRegimen,BestSAccuracy = PanSRegimens[0],PanSAccuracy[0]
        BestSDrugs = regimen_df.loc[regimen_df['Number'] == int(BestSRegimen.replace("s","")), 'Regimen'].iloc[0]
        BestSBreakdown = regimen_df.loc[regimen_df['Number'] == int(BestSRegimen.replace("s","")), 'Breakdown'].iloc[0]
    else:
        BestSRegimen,BestSAccuracy,BestSDrugs,BestSBreakdown = "-","-","-","-"
    newdata = {"UNIQUEID":index,
               "BestGRegimen":BestSRegimen,
               "BestGAccuracy":BestSAccuracy,
               "BestGDrugs":BestSDrugs ,
               "BestGBreakdown":BestSBreakdown ,
               "AllGRegimens": PanSRegimens,
               "AllGAccuracy": PanSAccuracy}
    summary_list.append(newdata)
summary = pd.DataFrame.from_dict(summary_list,orient="columns")

###CHECK REAL PHENO - PAN S
col_filter = list([i for i in df.columns if "w" in i])
dfz = df[col_filter]
summary_list2 = []
for index, row in dfz.iterrows():
    PanSRegimens = list(row[row.isin(["0","1"])].index)
    PanSAccuracy = list(row[row.isin(["0","1"])])
    if len(PanSRegimens)>0:
        BestSRegimen,BestSAccuracy = PanSRegimens[0],PanSAccuracy[0]
        BestSDrugs = regimen_df.loc[regimen_df['Number'] == int(BestSRegimen.replace("w","")), 'Regimen'].iloc[0]
        BestSBreakdown = regimen_df.loc[regimen_df['Number'] == int(BestSRegimen.replace("w","")), 'Breakdown'].iloc[0]
        
    else:
        BestSRegimen,BestSAccuracy,BestSDrugs,BestSBreakdown = "-","-","-","-"
    newdata = {"UNIQUEID":index,
               "BestPRegimen":BestSRegimen,
               "BestPDrugs":BestSDrugs ,
               }
    summary_list2.append(newdata)
summary2 = pd.DataFrame.from_dict(summary_list2,orient="columns")

###CHECK REAL PHENO - SOME S
col_filter = list([i for i in df.columns if "x" in i])
dfz = df[col_filter]
summary_list3 = []
for index, row in dfz.iterrows():
    FullPhenoRegimens = list(row[row.isin(["0","1"])].index)
    newdata = {"UNIQUEID":index,
               "FullPhenoRegimens":FullPhenoRegimens,
               }
    summary_list3.append(newdata)
summary3 = pd.DataFrame.from_dict(summary_list3,orient="columns")


final = pd.merge(summary, summary2, on="UNIQUEID", how="inner").merge(summary3, on="UNIQUEID", how="inner").merge(df_dupl,on="UNIQUEID", how="inner")
final.to_csv("FINAL/Regimens_bysample.csv")


#######################
df_allRR = final
df_haspheno = final[final["FullPhenoRegimens"].str.len() != 0]
df = df_haspheno[["BestPRegimen","BestGRegimen","BestPDrugs","BestGDrugs","BestGAccuracy",]]
df.to_csv("FINAL/Regimens_bysample_simple.csv")
df_pred_true = df[(df["BestGRegimen"]!="-")&(df["BestPRegimen"]!="-")]
df_pred_false = df[(df["BestGRegimen"]!="-")&(df["BestPRegimen"]=="-")]
df_nopred_true = df[(df["BestGRegimen"]=="-")&(df["BestPRegimen"]!="-")]
df_nopred_false = df[(df["BestGRegimen"]=="-")&(df["BestPRegimen"]=="-")]
df_accurate_panS = df[(df["BestGAccuracy"]=="1")]
df_inaccurate_pansS = df[(df["BestGAccuracy"]=="0")]

newdata = {"RR":df_allRR.shape[0],"RR_pheno_atleast1":df_haspheno.shape[0]}
newdata.update({"g1p1":df_pred_true.shape[0],"g1p0":df_pred_false.shape[0],"g0p1":df_nopred_true.shape[0],"g0p0":df_nopred_false.shape[0]})
newdata.update({"panSright":df_accurate_panS.shape[0],"panSwrong":df_inaccurate_pansS.shape[0]})
newdata.update({"%PredRegHasReg":round(df_pred_true.shape[0]/(df_pred_true.shape[0]+df_pred_false.shape[0]),3)})
newdata.update({"%PredSRegCorrect":round(df_accurate_panS.shape[0]/(df_pred_true.shape[0]+df_pred_false.shape[0]),3)})
summary = pd.DataFrame(newdata, index=[0]).T

pd.options.display.float_format = '{:,.0f}'.format
summary.to_csv("FINAL/Regimens_stats.csv")


