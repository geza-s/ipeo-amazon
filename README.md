 A deep learning project by Geza Soldati, Shadya Gamal and Anna Halloy
# Planet: Understanding the Amazon from Space

## Topic
This project is focused on the set of 40479 images of the Amazon basin. 
They come from Planet’s Flock 2 satellites and were collected between January 1, 2016 and February 1, 2017. 
Each image has a size of 256x256 pixels and is defined only by the RGB band. 
Each image is associated with a set of labels describing it.
There are in total 17 possible labels that characterise the ground and atmospheric conditions.

## Goals
The main goal of this project is to create a model that can predict which label is associated to an image.

The additional goal is to compare the performance of two approaches: 
- The first approach, here often referred as "multilabel model", is the training of a CNN model with, as output, 
a binary array of size 17, corresponding to the prediction for each label. 
- The second approach, here often referred as "multi-model", is the training of two CNN models where one is related 
to 3 atmospheric condition labels, and the second is related to the 14 other ground condition labels.

## Files structure
The files structures consist mainly in 2 sets of 3 files: A main python jupyter notebook, 
and two related python files ("engine" and "module").
Each set is related to one of the two approaches: "Multilabel" for the full multilabel classifier, 
and "Multi_Model" for the separated model approach (a multi-class classifier and a multilabel classifier).

First, "IPEO_data_Pre_Processing.ipynb" is the file associated to the pre-processes we implemented before 
the training and testing. It is not needed to be re-run to be able to train or test our models. 
This pre-process mainly created the 3 csv which are given: "test.csv", "training.csv" and "validation.csv"

The main files, here "Multilabel_classification_Amazon.ipynb" and "MultiModel_classification_Amazon.ipynb",
are the files to run to train, validate and test the models.
The two other associated python files integrate all the necessary functions and classes.
All the important functions are grouped in the "engine" called files, 
while all the classes are in the "module" called files. 

Finally, "IPEO_Post_Processing.ipynb" is an additionnal file regrouping all the analysis and comparison we did. 
It is mainly creating figures for the report. 

## Dependencies (to install beforehand)
To be able to run without issues, the following python modules should be installed:
- numpy, matplotlib, pandas, pytorch, scikit-image, scikit-learn, datetime, json, os
- seaborn, tqdm, torchinfo

## Instruction to run
The main files are all jupyter notebooks in python. 

The order in which to run the notebooks is the following : 
1) IPEO_data_Pre_Processing.ipynb (Optional)(To Explore and Split the data)
2) Multilabel_classification_Amazon.ipynb (To train and test the first model)
3) MultiModel_classification_Amazon.ipynb (To train and test the second model)
4) IPEO_Post_Processing.ipynb (Optional)(To plot the results)

