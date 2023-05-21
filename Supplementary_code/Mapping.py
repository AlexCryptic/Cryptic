import numpy as np
import pandas as pd
import statistics as stat
import math, time, itertools, glob, os, re, argparse

import Bio, pysam
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio.Blast import NCBIWWW
from Bio.Blast import NCBIXML

import copy
import gumpy
from gumpy import Genome
import gzip, pickle

import matplotlib.pyplot as plt
import seaborn as sns
plt.switch_backend('agg')

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("-i","--folderinputresults",type=str)
parser.add_argument("-t","--datatype",type=str) #KMER or #MUTATIONS
args = parser.parse_args()
data_type = args.datatype
run = str(args.folderinputresults)
weight_directory= "{}_RUNS/{}/Weights/".format(data_type,run)



catalogue_loc = "Mapping/CATALOGUE_CRyPTICv1.csv"
catalogue = pd.read_csv(catalogue_loc)

with gzip.open("Mapping/H37rV_v3.pkl.gz",'rb')as INPUT:
    reference_genome=pickle.load(INPUT)






allFiles = glob.glob(weight_directory+"TopWeightsKmers_*.csv") #Load list of all files to include in tables
df=pd.DataFrame()
df_cols = ["EPIC","DRUG","FULL","METHOD","DATA","TEST_SET","LABEL","RUN","PATH","SAVE_PREFIX"]
df_lst = []
for file in allFiles:
    save_prefix = file.split("TopWeightsKmers_")[1].split("_Run")[0] #keep between "Results_" and ".csv"
    file_variables = save_prefix.split(sep="_") #create list of all terms of interest
    #epic, drug, method,data,test_set, label = file_variables[0],file_variables[1],file_variables[3],file_variables[4],file_variables[5],file_variables[6]
    save_prefix = save_prefix+"_"+run
    df_lst.append(file_variables+[run,file, save_prefix])
df = pd.DataFrame(df_lst, columns=df_cols)



for index, row in df.iterrows():

    save_prefix, drug = row["SAVE_PREFIX"], row["DRUG"]
    print(save_prefix)

    KmerRanked = pd.read_csv(weight_directory+"TopWeightsKmers_{}.csv".format(save_prefix))


    StringList = KmerRanked["Kmer"].tolist()

    #convert strings to seqs
    AlphabetList = []
    for String in StringList:
        My_seq = SeqRecord(Seq(String), name = str(String), id=str(String), description="randomdescripton")
        AlphabetList.append(My_seq)

    #convert seqs to fasta
    FastALoc = "Mapping/Gottarun.fasta"
    FastAGenerator = SeqIO.write(AlphabetList,FastALoc,"fasta")

    #map fasta to reference genome and generate sam file
    SamLoc = "Mapping/Frodoand.sam"
    ReferenceLoc = "Mapping/TBreference"
    bwa_command = ("bowtie2 -D 20 -R 3 -N 0 -L 20 -i S,1,0.5 -x Mapping/TBreference -f %s -S %s" %(FastALoc, SamLoc))
    os.system (bwa_command)

    #generate table of mapped results
    mappingresults_cols = ["Kmer","Mapping_Sequence","Same?","Mapped?","Position","NM","MD"]
    mappingresults_lst = []

    samfile = pysam.AlignmentFile(SamLoc, "rb", check_sq=False)
    count = 0
    for read in samfile.fetch():
        count = count+1
        try:
            mappingresults_lst.append([read.qname,read.seq,(read.seq==read.qname),int(not read.is_unmapped),read.pos,read.get_tag("NM"),read.get_tag("MD")])
        except KeyError:
            mappingresults_lst.append([read.qname,read.seq,(read.seq==read.qname),int(not read.is_unmapped),read.pos,"0","0"])
    mappingresults = pd.DataFrame(mappingresults_lst,columns=mappingresults_cols)
    FinalList = KmerRanked.merge(mappingresults, how='outer', on="Kmer").sort_values(by=['Rank', 'Position'],ascending=[True,True]).reset_index(drop=True)

    FinalList.to_csv(weight_directory+"TopLocations_%s.csv"%(save_prefix))  

    dff = FinalList[(FinalList["Mapped?"]==1)].copy().reset_index(drop=True)
    dff['MD'] = dff['MD'].str.replace('^', '')
    dff["Shift"] = dff["Shift2"] = dff["Shift3"] = "-" 
    dff["PositionM1"] = dff["PositionM2"] = dff["PositionM3"] = "-" 
    dff["BP_M1"] = dff["BP_M2"] = dff["BP_M3"] = "-" 
    dff["BP_W1"] = dff["BP_W2"] = dff["BP_W3"] = "-" 
    dff["M1"] = dff["M2"] = dff["M3"] = "-" 
    dff["Type"]= np.where((dff["NM"]==0),"Wild","Mutation")

    dff["Mutation"] = "-"

    #CALCULATE ALL SHIFTS, POSITIONS AND MUTATIONS
    for index, row in dff.iterrows():

        if row["NM"] >= 1:
            try: 
                row["Shift"] = int(re.split('[a-z]+',row["MD"], flags=re.IGNORECASE)[0])
                dff.loc[index,'Shift'] = int(row["Shift"])
                row["PositionM1"] = int(row["Position"])+int(row["Shift"])
                dff.loc[index,'PositionM1'] = row["PositionM1"]
                row["BP_M1"] = str(''.join([i for i in row["MD"] if not i.isdigit()]).lower())[0]
                dff.loc[index,'BP_M1'] = row["BP_M1"]
                dff.loc[index,'BP_W1'] = reference_genome.genome_sequence[reference_genome.genome_index==row["PositionM1"]] 
            except (IndexError,AttributeError,TypeError):
                    None
            if row["NM"] >= 2:
                try: 
                    row["Shift2"] = int(re.split('[a-z]+',row["MD"], flags=re.IGNORECASE)[1])+1
                    dff.loc[index,'Shift2'] = int(row["Shift2"])
                    row["PositionM2"] = int(row["PositionM1"])+int(row["Shift2"])
                    dff.loc[index,'PositionM2'] = row["PositionM2"]
                    row["BP_M2"] = str(''.join([i for i in row["MD"] if not i.isdigit()]).lower())[1]
                    dff.loc[index,'BP_M2'] = row["BP_M2"]
                    dff.loc[index,'BP_W2'] = reference_genome.genome_sequence[reference_genome.genome_index==row["PositionM2"]]
                except (IndexError,AttributeError,TypeError):
                        None
                if row["NM"] >= 3:
                    try: 
                        row["Shift3"] = int(re.split('[a-z]+',row["MD"], flags=re.IGNORECASE)[2])+1
                        dff.loc[index,'Shift3'] = int(row["Shift3"])
                        row["PositionM3"] = int(row["PositionM2"])+int(row["Shift3"])
                        dff.loc[index,'PositionM3'] = row["PositionM3"]
                        row["BP_M3"] = str(''.join([i for i in row["MD"] if not i.isdigit()]).lower())[1]
                        dff.loc[index,'BP_M3'] = row["BP_M3"]
                        dff.loc[index,'BP_W3'] = reference_genome.genome_sequence[reference_genome.genome_index==row["PositionM3"]]
                    except (IndexError,AttributeError,TypeError):
                        None

    def position_to_mutation (position,bp_mutated):
        try:
            gene_name = reference_genome.at_index(position)
            reference_gene=reference_genome.genes[gene_name]
            sample=copy.deepcopy(reference_gene)
            sample.sequence[sample.index==position]=bp_mutated
            sample._translate_sequence()
            mutation = sample.list_mutations_wrt(reference_gene)
            full_mutation = gene_name+"_"+mutation[0]
            return(full_mutation)
        except (IndexError,AttributeError,TypeError,KeyError):
            return("fail")

    def position_to_boundaries (position_boundary,depth):
        try:
            left_boundary = position_boundary+depth
            right_boundary = position_boundary+31

            mask=reference_genome.genome_index==left_boundary
            gene_name = reference_genome.genome_feature_name[mask]
            gene = reference_genome.genes[gene_name[0]]
            position = gene.numbering[gene.index==left_boundary]
            aminoacid = gene.amino_acid_sequence[gene.amino_acid_numbering==position[0]]

            mask_r=reference_genome.genome_index==right_boundary 
            gene_name_r = reference_genome.genome_feature_name[mask_r]
            gene_r = reference_genome.genes[gene_name_r[0]]
            position_r = gene_r.numbering[gene_r.index==right_boundary]
            aminoacid_r = gene_r.amino_acid_sequence[gene_r.amino_acid_numbering==position[0]]

            return([gene_name[0],(position[0]),(position_r[0])])
        except (IndexError,AttributeError,TypeError):
            return("-")


    for index, row in dff.iterrows():
        if row["NM"] >= 0:
            None
        if row["NM"] >= 1:
            try:
                dff.loc[index,'M1'] = position_to_mutation (row["PositionM1"],row["BP_M1"])
            except (IndexError,AttributeError,TypeError,KeyError):
                None
            if row["NM"] >= 2:
                try:
                    dff.loc[index,'M2'] = position_to_mutation (row["PositionM2"],row["BP_M2"])
                except (IndexError,AttributeError,TypeError,KeyError):
                    None
                if row["NM"] >= 3 :
                    try:
                        dff.loc[index,'M3'] = position_to_mutation (row["PositionM3"],row["BP_M3"])
                    except (IndexError,AttributeError,TypeError,KeyError):
                        None
        


    dff = dff.sort_values(by=['Rank', 'Position'],ascending=[True,True])
    dff.to_csv(weight_directory+"TopMutations_%s.csv"%(save_prefix))

    merger=dff[["Rank","Position", "Weight"]].drop_duplicates(subset=["Rank"],keep="first").reset_index(drop=True)

    dff2 =pd.DataFrame({'Depth' : dff.groupby(["Rank","Type","M1","M2","M3"]).size()}).sort_values(by=['Rank', 'Depth'],ascending=[True,False]).reset_index()
    dff3=pd.merge(dff2,merger,on="Rank",how="left")
    dff3["Limits"] = dff3["Gene"] = dff3["Change"] = dff3["Cat_gene"] = dff3["Cat_mut"] = dff3["Cat_loc"] = dff3["Cat_gene_other"] = dff3["Cat_mut_other"]= "-"


    for index, row in dff3.iterrows():
        if row["Type"]=="Wild":
            try:
                boundary_list = position_to_boundaries(row["Position"],row["Depth"])
                wild_stretch = str(boundary_list[0])+"_"+str(boundary_list[1])+"-->"+str(boundary_list[2])
                dff3.loc[index,'Limits'] = wild_stretch
                dff3.loc[index,'Change'] = wild_stretch
            except (IndexError,AttributeError,TypeError):
                None
        if row["Type"]=="Mutation" and row["M2"]=="-" and row["M3"]=="-":
            dff3.loc[index,'Change'] = row["M1"]
        elif row["Type"]=="Mutation" and row["M3"]=="-":
            dff3.loc[index,'Change'] = row["M1"] + ", " + row["M2"]
        elif row["Type"]=="Mutation":
            dff3.loc[index,'Change'] = row["M1"] + ", " + row["M2"] + ", " + row["M3"]
        dff3.loc[index,'Gene'] = reference_genome.at_index(row["Position"])

    #for index, row in dff3.iterrows():
        #dff3.loc[index,"Cat_gene"] = np.where(row["Gene"].isin(catalogue[catalogue["DRUG"]==drug]["GENE"]),1,0)
    dff3["Cat_gene"] = np.where(dff3["Gene"].isin(catalogue[catalogue["DRUG"]==drug]["GENE"]),1,0)
    dff3["Cat_gene_other"] = np.where(dff3["Gene"].isin(catalogue["GENE"]),1,0)
    dff3["Cat_mut"] = np.where(dff3["M1"].isin(catalogue[catalogue["DRUG"]==drug]["GENE_MUTATION"]),1,0)
    dff3["Cat_mut_other"] = np.where(dff3["M1"].isin(catalogue["GENE_MUTATION"]),1,0)

    for index, row in dff3.iterrows():
        if row["Cat_gene"] == 1:
            if row["Type"]=="Wild":
                try:
                    boundary_list = position_to_boundaries(row["Position"],row["Depth"])
                    list_truth = list(map(int,catalogue[(catalogue["GENE"]==row["Gene"])&(catalogue["POSITION"]!="*")]["POSITION"].tolist()))
                    list_pred = list(range(boundary_list[1], boundary_list[2]))
                    if any(elem in list_truth  for elem in list_pred) == True:
                        dff3.loc[index,'Cat_loc'] = 1
                    else:
                        dff3.loc[index,'Cat_loc'] = 0
                except (IndexError,AttributeError,TypeError):
                    None




    dff3.to_csv(weight_directory+"FinalMutations_%s.csv"%(save_prefix))

    final = dff3[["Rank","Gene","Type","Change","Depth","Weight","Cat_gene","Cat_mut","Cat_loc","Cat_gene_other"]].set_index("Rank")
    final["Weight"] = final["Weight"].round(6)
    final.to_csv(weight_directory+"0FinalTable_%s.csv"%(save_prefix))






