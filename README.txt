0. Introduction

This zipped folder contains sample code supporting the manuscript "A generalisable approach to drug susceptibility prediction for M. tuberculosis using machine learning and whole-genome sequencing".

1. Data availability

All CRyPTIC data used in this manuscript were made publicly available at the time of its submission (http://ftp.ebi.ac.uk/pub/databases/cryptic/). Samples and their unique identifiers used for this analysis are presented in the supplementary appendix included in the submission.

2. System requirements

All code was written in python (v3.8.5) and run on a CentOS Linux 7 (Core) system. Environments were managed using conda v4.8.5. The minimum required dependencies to generate and run the code are available in the "Dependencies.txt" file. 


3. Installation 

Instructions to install required dependencies are provided in the "Dependencies.txt" file.
 
4. Instructions and Demo data

A demo is provided to train, test and assess the performance of the machine learning model. Given the huge size of the dataset, we provide a simple example with 50 isolates and phenotypes for all thirteen drugs (isoanizid, rifampicin, ethambutol, levofloxacin, moxifloxacin, amikacin, kanamycin, ethionamide, bedaquiline, linezolid, delamanid, clofazimine). 

The "Mock_data" folder contains two files for each drug: (1) a feature matrix in HDF5 format, saved as "DRUG.NUMBERSAMPLES.features.h5", with one line per isolate and one column per kmer pattern, and (2) a labels table in csv format, saved as "DRUG.NUMBERSAMPLES.labels.csv", with one line per isolate and one column per phenotype.

The "Supporting_files" folder contains two files: (1) a table outlining the equivalence between each MIC, LOG2MIC and binary phenotypes using the method described by Fowler et al. (2021, see references) saved as "CONCORDANCE.csv", and (2) a table outlining all available lables and isolates.

The "Output" folder contains two empty subfolders: (1) a "Results" subfolders where raw MIC predictons for each model are saved, and (2) an "Analysis" subfolder where MIC predictons are converted to binary predictions and all relevant analysis are computed. 

The "Model.py" script requires 8 variables to run. -a is the acronym of the antibiotic to be studied (ex: INH for isoniazid). -n is the number of samples included in the set using a four digit format (ex: 0050 for 50 isolates). -m is the machine learning model to be tested (ex: xgbr for XGBoost regressor). -c is the number of cores to use in parallel for computation (ex: 1 for 1 core on 1 node). -f is the featype type (ex: KMER for kmer patterns). -r is the run number (ex: Run_001 for the first run). -t is the test set (ex: Test for a simple cross-validation). -p is the phenotype (ex: LOG2MIC_NEW for the MIC in LOG2 format). The output of the script is a result file in "Output/Results". For the demo, the script computed a simple five-fold cross-validation of the data set. In the manuscript, the script computes a train-test data split, and a leave-one-site-out cross-validation (see Methods).

The "StatisticalAnalysis" requires 2 variables to run. -i is the the Run number mentioned in "Model.py" (ex: Run_001 for the first run). -t is the feature type (ex: KMER for kmer patterns). The output of the script is an analysis file and summary table n "Output/Analysis".

The demo can be run from the main folder once all dependencies have been installed using the following commands:
---> python Model.py -a INH -n 0050 -m xgbr -c 1 -f KMER -r Run_001 -t Test -p LOG2MIC_NEW
---> python StatisticalAnalysis.py -i Run_001 -t KMER
The commands can be repeated for amikacin by replacing INH with the acronym for the drug of interest.

5. Supplementary code

We provide several additional sample scripts used for all steps of this study. We note that these were conducted on different clusters using different folder and naming conventions. These include:
- Weights_to_Kmers.py: finding the kmers associated with the most impactful pattern features
- Mapping.py: identifying the gene mutations associated with each kmer described by the previous script ("Weights_to_Kmers.py") 
- SusceptibilityProfiles.py: predict entire drug regmens in line with 2021 WHO guidelines
- Pvalue.py: Generate p-values for sensitivity and specificity comparisons using McNemar's test
- Table_generator: Generate results tables
- Model_hyperopt.py: Perform hyperparameter optimization to determine the best training hyperparameters.
- Model_filter.py: Perform filtering to keep only most relevant features.

6. Supplementary files

Additional information and files to support analysis are presented in the "Supplementary files" folder.

6. Questions and details

We remain available should you have any questions about the code used to generate results, or require further details.