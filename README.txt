1. Dataset: Train/Dataset
All datasets related to reactions between linear symmetric ethers and CH₃, H, OH, HO₂ radicals are stored in the "Dataset" folder.
train__H/CH3/OH/HO2-M062-0_1 (1-10).txt: Training set for reaction rate constants of linear symmetric ethers with CH₃, H, OH, HO₂ radicals.
test__H/CH3/OH/HO2-M062-0_1 (1-10).txt: Validation set for reaction rate constants of linear symmetric ethers with CH₃, H, OH, HO₂ radicals.
H/CH3/OH/HO2__M062-test.txt: Test set for reaction rate constants of linear symmetric ethers with CH₃, H,  OH, HO₂ radicals.
Data format: similarity, four Mordred topological descriptors, temperature and log k (logarithmic rate constant).

2. Neural network training module: train/
This folder contains Python codes and shell scripts for model training.
XGB-FNN.py: Training code of the XGB-FNN hybrid model.

1.Download Anaconda3-2019.07-Linux-x86_64.sh from the Anaconda official website (https://repo.anaconda.com/archive/). In the Linux terminal, run the following command:
$ sh Anaconda3-2019.07-Linux-x86_64.sh
Follow the prompts to complete the installation.

2.Create and activate the virtual environment. In the Linux terminal, run the following commands:
$ conda create -n ML python=3.7.3
ML is the name of the virtual environment. All packages will be installed within this environment.
Activate environment:
$ conda activate ML
Exit environment:
$ conda deactivate

3.Install required packages by the following commands:
$ conda activate ML
$ conda install pytorch==1.4.0 cpuonly -c pytorch
$ conda install scikit-learn==0.23.2
$ conda install matplotlib==3.3.2
$ conda install numpy==1.15.4
$ conda install pandas==1.1.5
$ conda install dill

After environment configuration, open the Linux terminal under the "train" folder and run the following commands to start model training:
$ conda activate ML
$ sh note.sh
$ sh all.work.sh

3. Prediction module: predict/
The "predict" folder stores trained models, input datasets and prediction codes.
30020.fnn / 30022.fnn /30030.fnn: Trained CH3-CatBoost-FNN model files for CH₃.
30023.fnn / 30031.fnn / 30070.fnn: Trained H-CatBoost-FNN model files for H.
32020.fnn / 32091.fnn / 32100.fnn: Trained OH-XGB-FNN model files for OH.
31060.fnn / 31072.fnn / 31092.fnn: Trained HO2-LGB-FNN model files for HO₂.
W3-NN.py: Code for model inference