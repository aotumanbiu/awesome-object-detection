# Object detection reference training scripts

This folder contains reference training scripts for object detection.
They serve as a log of how to train specific models, to provide baseline
training and evaluation scripts to quickly bootstrap research.

To execute the example commands below you must install the following:

```
cython
pycocotools
matplotlib
```

You must modify the following flags:

`--data-path=/path/to/coco/dataset`

`--nproc_per_node=<number_of_gpus_available>`

Except otherwise noted, all models have been trained on 8x V100 GPUs. 

### Faster R-CNN ResNet-50 FPN
```
torchrun --nproc_per_node=8 train.py\
    --dataset coco --model fasterrcnn_resnet50_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3 --weights-backbone ResNet50_Weights.IMAGENET1K_V1
```

### Faster R-CNN MobileNetV3-Large FPN
```
torchrun --nproc_per_node=8 train.py\
    --dataset coco --model fasterrcnn_mobilenet_v3_large_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3 --weights-backbone MobileNet_V3_Large_Weights.IMAGENET1K_V1
```

### Faster R-CNN MobileNetV3-Large 320 FPN
```
torchrun --nproc_per_node=8 train.py\
    --dataset coco --model fasterrcnn_mobilenet_v3_large_320_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3 --weights-backbone MobileNet_V3_Large_Weights.IMAGENET1K_V1
```



**==========================================================分割线=============================================================**



本文为了方便学习整理，单独把RetinaNet文件简单提取出来，因此环境需要满足以下条件：

```python
torch >= 1.11.0
torchvision >= 0.12.0
```

### 网络结构图

<img src="../files/fasterrcnn.svg" style="zoom:100%;" />



#### 1. RPN

```python
# RPN的检测头权重共享 conv+ logits/reg_bbox
objectness, pred_bbox_deltas = self.head(features)
        anchors = self.anchor_generator(images, features)  # 先验框 B x [32205, 4]
    
        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]  # FPN各个输出层中的先验框数量
        # 将预测结果按单位铺平 如[2, 15, 25, 38] -> [28500, 1], [2, 60, 25, 38] -> [28500, 4]
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
        
        # 先验框解码生成建议框(注: 在这里截断反向传播)
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        # 最终筛选后预测框及其对应置信度分数, RPN最后输出的建议框就是boxes
        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level) 

        losses = {}
        if self.training:
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            # 先验框与其匹配真实框进行编码
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors) 
            loss_objectness, loss_rpn_box_reg = self.compute_loss(objectness, pred_bbox_deltas, labels, regression_targets)
```



#### 2. MultiRoIAlign

在pytorch中的Faster R-CNN实验中, 采用的是RoIAlign而不是原论文中的RoIPooling。

##### a. RoIPooling

* $feat\_stride=32$，原图 $800*800$，最后一层特征图 feature  map 大小 $25*25$。

* 假定原图中有一region proposal，大小为 $665*665$，这样，映射到特征图中的大小：$665/32=20.78$， 即 $20.78*20.78$。在C++源码中，计算的时候会进行取整操作，于是，进行所谓的第一次量化，即映射的特征图大小为$20*20$。
* 假定 $pooled\_w=7$，$pooled\_h=7$，即 $pooling$ 后固定成 $7*7$ 大小的特征图，所以，将上面在 feature map上映射的 $20*20$ 的 region proposal划分成49个同等大小的小区域，每个小区域的大小 $20/7=2.86$，即 $2.86*2.86$，此时，进行第二次量化，故小区域大小变成 $2*2$。
* 每个 $2*2$ 的小区域里，取出其中最大的像素值，作为这一个区域的‘代表’，这样，49个小区域就输出49个像素值，组成 $7*7$ 大小的feature map。

​		经过这两次量化，候选区域已经出现了较明显的偏差。更重要的是，该层特征图上0.1个像素的偏差，缩放到原图就是3.2个像素。那么0.8的偏差，在原图上就是接近30个像素点的差别，这一差别不容小觑。



##### b. RoIAlign

​		ROI Align的思路很简单：取消量化操作，使用双线性内插的方法获得坐标为浮点数的像素点上的图像数值，从而将整个特征聚集过程转化为一个连续的操作。值得注意的是，在具体的算法操作上，ROI Align并不是简单地补充出候选区域边界上的坐标点，然后将这些坐标点进行池化，而是重新设计了一套比较优雅的流程：

* 遍历每一个候选区域，保持浮点数边界不做量化。
* 将候选区域分割成 $k * k$ 个单元，每个单元的边界也不做量化。
* 在每个单元中计算固定四个坐标位置，用双线性内插的方法计算出这四个位置的值，然后进行最大池化操作。



#### 3. Head

注意：在Faster R-CNN中检测头输出维度与SDD，RetinaNet，yolo不同。

