
* Questions on the paper:

1. Is the increase in performance from the decoupled head is because the new architecture or the increased number of parameters? In other words, Is the decouped head better than a bigger coupled head? especially the improvement from yolo to yolox decreased wehen moving from yolo-s/yolox-s to yolo-x/yolox-x.
2. We should test the effect of the decoubled head on models other than YOLO, such as any network doing more than one task using one head then converting that head to multiple heads.
3. Does the decoupled head work on different tasks such as classification and regression and also classification of different objects not sharing the same semantics? Meaning, can we use the decoupled head in classification only and make a heirarchy of heads depending on the category of classes?


* I liked the idea of multi-positive. It is similar to MIL Tracker. They share the idea of making use of output that are not 100% correct to handle the imbalance problem. 

* YOLOX may not outperform YOLO with a large model size.
