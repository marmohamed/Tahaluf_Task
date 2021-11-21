* The notebook attached is just a run of a pretrained model (yolo-s) using kaggle environment.

* Questions on the paper:

1. Is the increase in performance from the decoupled head is because the new architecture or the increased number of parameters? In other words, Is the decouped head better than a bigger coupled head? especially the improvement from yolo to yolox decreased wehen moving from yolo-s/yolox-s to yolo-x/yolox-x.
2. We should test the effect of the decoubled head on models other than YOLO, such as any network doing more than one task using one head then converting that head to multiple heads.
3. Does the decoupled head work on different tasks such as classification and regression and also classification of different objects not sharing the same semantics? Meaning, can we use the decoupled head in classification only and make a heirarchy of heads depending on the category of classes?


* I liked the idea of multi-positive. It is similar to MIL Tracker. They share the idea of making use of output that are not 100% correct to handle the imbalance problem. 

* YOLOX may not outperform YOLO with a large model size.

* Suggestions to imrpove:
1. Try Attention.
2. For each detected object, we can add another layer to get that part from the original image and finetune the prediction. But, this will convert it to two-stage detector.
3. Extending the idea of multi-positives, we can detect parts of the objects to increase the number of positive samples and making use of the gradient from small parts of objects. For example, we can detect the center and the corners of the object and we may add other parameters to predict to enhance the model features.
4. Try self-supervised training during the original training.