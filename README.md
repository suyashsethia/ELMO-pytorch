# ELMo-using-pytorch
------
## There are two tasks: 
### 1. Sentiment Analysis([ SST.ipynb ](https://github.com/JainitBITW/ELMo-using-pytorch/blob/main/SST.ipynb))
### 2. Natural Language Inference([MULTI_NLI.ipynb](https://github.com/JainitBITW/ELMo-using-pytorch/blob/main/MULTI_NLI.ipynb))
------
## 1. Sentiment Analysis
### 1.1. Data
The data used is [sst](https://huggingface.co/datasets/sst) from huggingface datasets. 
The dataset is already imported in the code.
### 1.2. Model
The model used is ElMO which is head and a classifier is attached to it.
#### 1.2.1. Elmo
The ElMO model is made using pytorch's BiLSTM. Two layers BiLSTM are stacked. The output of the first layer is fed to the second layer. The output of the second layer is fed to the classifier. 
The ElMO is first pretrained on language modelling task and then fine tuned on the sentiment analysis task.
#### 1.2.2. Classifier
WHen the classifier is attached the ElMO model, the ElMO model is freezed and only the classifier is trained except the lambda parameters. The classifier is a simple MLP.
### 1.3. Training
The Training is done using optimisers which I thoought are fine one can change them as per their wish. 

------
## 2. Natural Language Inference
### 2.1. Data
The data used is [multi-nli](https://huggingface.co/datasets/multi_nli) from huggingface datasets.
The dataset is already imported in the code.
### 2.2. Model
The model used is ElMO which is head and a classifier is attached to it.
#### 2.2.1. Elmo
The ElMO model is made using pytorch's BiLSTM. Two layers BiLSTM are stacked. The output of the first layer is fed to the second layer. The output of the second layer is fed to the classifier.
The ElMO is first pretrained on language modelling task and then fine tuned on the sentiment analysis task.
#### 2.2.2. Classifier
WHen the classifier is attached the ElMO model, the ElMO model is freezed and only the classifier is trained except the lambda parameters. The classifier is a simple MLP.
### 2.3. Training
The Training is done using optimisers which I thoought are fine one can change them as per their wish.


##### The code is already implemented in jupyter book and now being tranformed into modular code.