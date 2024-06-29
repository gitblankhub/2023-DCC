# 2023 Data Creator Camp 

SEP 2023 - DEC 2023

데이터셋 출처 : https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=79

분석 목적 : 

## 0. Data
- kfood_train : 학습데이터, 33593 images, 42 kinds of food classes(갈비구이, 갈치구이, .. 황태구이, 훈제오리) 
- kfood_val : 평가데이터, 4198 images 
- kfood_health_train : 학습데이터, 14115 images, 13 kinds of fodd classes(갈비찜, 된장찌개, ... 부대찌개, 순대) 
- kfood_health_val : 평가데이터, 1764 images 

## Mission 1
**kfood_train**, **kfood_val** dataset     
label 별 count & see each image   

8:2 cross validation        

- ResNet18      
epoch = 50, loss = cross entropy, optimizer = SGD(Stochastic gradient descent), learning rate = 0.001
batch size = 32, image resize = 224 x 224
(can't use pretrained weight)         

validation acc = 0.91, test acc = 0.6065 -> *overfitting problem*  

> (Self Study) ResNet    
> [Paper] Deep Residual Learning for Image Recognition https://arxiv.org/pdf/1512.03385v1     
> As depth of NN increases, dedgradation problem(not overfitted but error is larger) might occurs.     
> Residual learning framework makes deep network train easier. Network do easier residual mapping $F(x)=H(x)-x$ instead of directing mapping $H(x)$.      
> A residual block forms a building blocks of ResNets. Each block has a few stacked layers and identity shortcut connection that **skips** layers. Output of block is sum of original input and out put of stacked layer. $Output = F(x)+x$  

## Mission 2
**kfood_train**, **kfood_val** dataset  


(We've tried resnet34, 50 models and various kinds of augmentation combination, learning rate, epoch tuning!)     

- Resnet50     
epoch = 50, loss = cross entropy, optimizer = Adam(Adaptive Moment Estimation), learning rate = 0.005       
batch size = 32, image resize = 224 x 224

- To improve accuracy        
Normalization : image들의 RGB 채널별 정규화           
Image Augmentation for train data : RandomRotation(회전), CenterCrop         

validation acc = 0.71, test acc = 0.745

[본선]

- Resnet 101     
  epoch = 70, batch size = 64, optimizer = Adam, learning rate = 0.001      
  (We've tried Resnet50 Resnet101 Resnet154 architectures.. learning rate and epoch tuning.. lots of models!)        

test acc = 0.7918 
 

## Mission 3
**kfood_health_train**, **kfood_health_val** dataset  
Mission2에서 학습한 모델을 활용하여 건강관리를 위한 음식 이미지 데이터를 13개의 클래스로 분류하는 모델 학습.      

- Transfer learning 

Mission2에서 학습한 checkpoint(mission2.pt)를 불러와 resnet50 모델에 적용  
Full Connected Layer 13개의 classes로 수정 

- Fine tuning
  method1) train the entire model. Freezing 없이 모든 layer의 가중치를 다시 training 하기. (slightly better perf)
  method2) 초기 일부 layers는 freezing 하고 후반 layers들만 training 하기.

- Resnet101
  batchsize = 64 

> [SelfStudy] Transfer learning & fine tuning       
> pytorch tutorials : (https://tutorials.pytorch.kr/beginner/transfer_learning_tutorial.html)       
> Transfer learning : ML technique where pretrained model is used as starting point of new task. Instead of training model from scratch, use model that has already trained on other dataset.        
> Fine tuning : Replace the final layers to match the number of classes in new task. Then, optionally freeze initial layers to prevent updtaing.       


test acc = 0.95


[본선] 

epoch=50 , optimizer = Adam, learning rate = 0.001 

test acc = 0.980





