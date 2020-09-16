# PreAntiCOV
## Glimpse

An implementation for prediction and analysis of Anti-CoV functional peptide.

## How to Use

* `analysis.ipynb`: contains basic feature description of AAC, PHYCs, as well as classification between Anti-CoV and the other from three different sets
* `classify.py`: Make classification for different functional peptides set.
* `ArgsClassify.py`: Parameters of how to perform classification. 

## Requirements

We have already integrate the environment in `env.py`. execute `conda create -f env.yml` to install packages required in a new created `AMPrediction` conda env.

Enter the enviornment with `conda activate PreAntiCoV`.

## Steps for constructing classifiers

1. To extract features for given .fasta files, execute `python feature_extract.py`.
2. To establish predictors, execute `classify.py`. You can adjust any parameters at `ArgsClassify.py`.

