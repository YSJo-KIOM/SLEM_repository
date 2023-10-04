# SLEM Code Repository
Welcome to the code repository for Step-wise deep LEarning technique with Multi-precision data **(SLEM)**.

This repository provides **source codes for model training and model analysis.** Datasets for training and testing models, and example output files are also available.

**Citation**: Yousang Jo, Maree J Webster, Sanghyeon Kim, Doheon Lee, Interpretation of SNP combination effects on schizophrenia etiology based on stepwise deep learning with multi-precision data, Briefings in Functional Genomics, 2023.<br>
https://doi.org/10.1093/bfgp/elad041

### Model training code (SLEM_model_training.py)
```
python SLEM_model_training.py -i (input_dataset) -o (output_path) -m (save|CV_eval|tuning) --pr (connection_threshold,batch_size,epoch,learning_rate => ex: 0.1,100,750,0.01)
```
#### Save mode (save)
This mode is used to save models for further analysis. Input dataset (-i), output path (-o), and learning hyperparameters (--pr) should be provided. Initial weights are specified by files in 'weights_file' directory. <br>
The model is provided by a set of layer and bias matrices for four layers.
(example: files in model_save)

#### Cross-validation mode (CV_eval)
This mode is used to evaluate performances between SLEM and fully connected model. Input dataset (-i), output path (-o), and learning hyperparameters (--pr) should be provided. Initial weights are specified by files in 'weights_file' directory. <br>
The result is provided by a file with the performance of every model and the average of each method.<br>
(example: CV_eval.txt)

#### Hyperparameter tuning mode (tuning)
This mode is used to determine optimal hyperparameters. Input dataset (-i) and output path (-o) should be provided. Initial weights are specified by files in 'weights_file' directory. <br>
The range of hyperparameters can be set by modifying lines 584-587 in the code.
The result is provided by a file with the performance of every model that responds to the hyperparameter set.<br>
(example: param_tuning_xv_0.1.txt)
