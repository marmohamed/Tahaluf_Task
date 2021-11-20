
Run the following line to know the arguments
```
python Main.py --help
```

Run the following line to train different models with different hyperparameters.
```
bash scripts/train.sh
```

The `environment.yml` file is exported from my machine which has a different os than the docker machine, therefore, running conda create on docker makes errors and conflicts in the versions of some libraries.

------------
receptive field code is from [here](https://github.com/Fangyh09/pytorch-receptive-field.git), but adjusted to handle new layers. However, not all models are supported, any new model has new components other than conv, therefore, we have to modify the code for any new model.

macs and params is from [here](https://github.com/Lyken17/pytorch-OpCounter)

------------
I did not train a model till the end as it needs time and resources. Therefore, I made a script of what should be run. 

------------

* Dataset: 

The dataset is from [Kaggle](https://www.kaggle.com/agrigorev/clothing-dataset-full)

The dataset is not clean, therefore, we clean it by removing the noisy labels such as "Not sure".

The images contain 4-channel images. This is also handled.
Augmentation is used from albumentation, however, we can use pytorch transforms library.

* Loss function:

CrossEntropyLoss, but since the data is imbalanced, we add the option to weigh the loss based on the number of instances per class, using the argument `--weight_loss`.

* Metrics:

We track many metrics in tensorboard, the accuracy, f1-score (macro-micro) and accuracy per class. The data is imbalanced therefore, we need to know the f1 score and track the performnace of each class to check if there is a conflict between them. Maybe improvng one class decreases the accuracy of the other because the clothes are very similar to each other and we may have some classes under one category and one may have more instances than the other.

* Image size:

The dataset has high resolution images, which is very good and allows us to use deeper models. Therefore, we make the width and the height of the image hyperparameters to change and increase/decrease our receptive field depending on the alllowed resources.

* Models:

We have multiple model to try from torchvision, efficientnet and timm libraries, however we can add more in models/model.py.

* Distributed Training:

It is supported and you can check the arguments for enabling multi gpu/node training.

* Minimal model:

Regarding the model deployment and the reequirements for minimal model size, we can train any model we want then use model distillation to get a smaller model for deployment.

It is implemented , add the argument `--use_kd` and specify the teacher model path and the student model type.

Note: it does not work with mixed precision.


