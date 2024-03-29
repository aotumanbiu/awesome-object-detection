# awesome-object-detection
#####  为方便理解不同目标检测网络原理，整理了一些自己debug学习使用的代码，代码可以直接运行。
##### 同时，便于后续的温习和回顾。



##### Anchor-based和Anchor-free的主要区别在于：```正样本定义方式的区别，ATSS为消除两者区别的桥梁！！！```



* anchor-based

  * [x] [RetinaNet: Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002v2) 
  * [x] [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325v5)
  * [x] [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://proceedings.neurips.cc/paper/2015/file/14bfa6bb14875e45bba028a21ed38046-Paper.pdf)
  * [ ] [Mask R-CNN ](https://openaccess.thecvf.com/content_ICCV_2017/papers/He_Mask_R-CNN_ICCV_2017_paper.pdf)
  * [ ] ........
  
* anchor-free

  * [x] [FCOS: Fully Convolutional One-Stage Object Detection](https://arxiv.org/abs/1904.01355v5)
    
  * [x] [ATSS:](https://arxiv.org/abs/1912.02424v4)
  
  * [x] [Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection](https://proceedings.neurips.cc//paper/2020/file/f0bda020d2470f2e74990a07a607ebd9-Paper.pdf)
  
  * [ ] [Generalized Focal Loss V2: Learning Reliable Localization Quality Estimation for Dense Object Detection](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Generalized_Focal_Loss_V2_Learning_Reliable_Localization_Quality_Estimation_for_CVPR_2021_paper.pdf)
  
  * [x] [LD: Localization Distillation for Dense Object Detection](https://arxiv.org/abs/2102.12252)
    
  * [ ] [CenterNet: Keypoint Triplets for Object Detection](https://arxiv.org/abs/1904.08189v3)
    
  * [ ] ........


* mmdetection-mini  

  对于上述的部代码, 直接将其流程注释在mmdetection-mini中

