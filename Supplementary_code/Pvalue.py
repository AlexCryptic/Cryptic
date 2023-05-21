from statsmodels.stats.contingency_tables import mcnemar
import pandas as pd

df = pd.read_csv("Downloads/CRY_ML_CAT.csv")

for drug in ["INH","RIF","EMB","MXF","LEV","AMI","KAN","ETH"]:
    for subset in ["Pheno_R"]:
    
        dfo=df[(df.Subset==subset)&(df.Drug==drug)]
        pp, pn, np, nn = int(dfo["++"]),int(dfo["+-"]),int(dfo["-+"]),int(dfo["--"])
        table = [[pp, pn],[np, nn]]
        result = mcnemar(table, exact=False)
        
        print(subset+"_"+drug+'_p-value=%.5f' % (result.pvalue))
        print(pp, pn, np, nn)
        
        