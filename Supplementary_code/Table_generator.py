import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

final_group_list = ["CRY"]
method_list = ["SYSTEM","CAT","XPERT"]
drug_order = ["INH","RIF","EMB","LEV","MXF","AMI","KAN","ETH","RFB","BDQ","DLM","CFZ","LZD"]
drug_whitelist = {"SYSTEM":["INH","RIF","EMB","LEV","MXF","AMI","KAN","ETH","RFB","BDQ","DLM","CFZ","LZD"],"CAT":["INH","RIF","EMB","LEV","MXF","AMI","KAN","ETH"],"XPERT":["INH","RIF","LEV","MXF","AMI","KAN","ETH"]}
drug_blacklist = {"SYSTEM":[],"CAT":["RFB","BDQ","DLM","CFZ","LZD"],"XPERT":["EMB","RFB","BDQ","DLM","CFZ","LZD"]}
U_status_list = ["All","Only_U","No_U"]
RR_status_list = ["All","RR"]

for final_group in final_group_list:
    for U_status in U_status_list:
        for RR_status in RR_status_list:
            df = pd.read_csv("FINAL/Final4_{}.csv".format(final_group), index_col=0)
            df = df[(df.METHOD.isin(method_list))&(df.U_STATUS==U_status)&(df.RR_STATUS==RR_status)]
            for i in ["Sens","Spec","PPV","NPV"]:
                df[i+"+CI"]=df[i].round(3).astype(str)+" "+df[i+"_CI_range"].astype(str)
            for index, row in df.iterrows():
                if row["DRUG"] not in drug_whitelist[row["METHOD"]]:
                    df.drop(index, inplace=True)
            df = df.rename(columns={"METHOD":"Method","DRUG":"Drug","N":"All"})
            df["S"] = df["All"]-df["R"]
            cat_drug = CategoricalDtype(drug_order, ordered=True,)
            cat_method = CategoricalDtype(method_list, ordered=True,)
            df["Drug"]=df["Drug"].astype(cat_drug)
            df["Method"]=df["Method"].astype(cat_method)

            df = df.set_index(["Method","Drug"]).sort_values(by=["Method","Drug"])

            df = df[["All","R","S","SITE","N_PER_SITE","PREV_PER_SITE","Sens+CI","Spec+CI","PPV+CI","NPV+CI", ]]
            df["PREV_PER_SITE"] = df["PREV_PER_SITE"].round(3)

            df = df.rename(
                columns={"All":"Isolates","R":"Resistant",
                "S":"Susceptible",
                "Sens+CI":"Sensitivity(95%CI)","Spec+CI":"Specificity(95%CI)","PPV+CI":"PPV(95%CI)","NPV+CI":"NPV(95%CI)",
                "SITE":"Number_of_sites","N_PER_SITE":"Samples_per_site","PREV_PER_SITE":"Prev_per_site"},
                ).rename(
                index={"INH":"Isoniazid","RIF":"Rifampicin","EMB":"Ethambutol","LEV":"Levofloxacin","MXF":"Moxifloxacin","AMI":"Amikacin","KAN":"Kanamycin","ETH":"Ethionamd","RFB":"Rifabutin","BDQ":"Bedaquilin","DLM":"Delamanid","CFZ":"Clofazimine","LZD":"Linezolid"},
                ).rename(
                index={"XPERT":"GeneXpert","SYSTEM":"Machine learning","CAT":"Catalogue"},
                )

            df.to_csv("FINAL/Table2_{}_{}_{}.csv".format(final_group, U_status,RR_status))

            