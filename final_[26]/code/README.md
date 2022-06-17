#VFX Final Project
## I. Train a MaskRCNN 
### 1. 標註
1.1 透過Blender生成Dataset

### 2. 訓練資料紀錄
2.1 yaml檔案中範例
```yaml=
train :
  discription: 'Training dataset'
  location: 'Blender'
  path: 'C:/Users/MMD/Desktop/Xiang//MaskRCNN/joint/train'
  image_count: 27850
  crop:
    xmin: 0    # xmin (left column)
    xmax: 500  # xmax (right column)
    ymin: 0     # ymin (top row)
    ymax: 500    # ymax (bottom row)
```
2.2 訓練資料
```
[dataset]/
  └──[joint]/
     ├──[rgb]/
     |  ├──[rgb_1.png]
     |  ├──......
     |  └──[rgb_27850.png]
     └──[depth]/
     |  ├──[depth_1.png]
     |  ├──......
     |  └──[depth_27850.png]
     └── via_export_json.json   
      
```
### 3. 訓練
```shell=
python train.py --weights coco --yaml train.yaml --epochs 1500
```

3.2 模型儲存路徑

預設路徑: **MaskRCNN/logs** 

### 4. 預測
```shell=
python eval.py --gpu 0 --cfg eval --vis_depth --dataset_path ./datasets --weight_path ../logs/latefusion.tar 
```


## II. Image-to-Image Translation
### 1. 引用
1.1 引用於[Jun-Yan Zhu](https://github.com/junyanz) ,[Taesung Park](https://github.com/taesungp) 與 [Tongzhou Wang](https://github.com/SsnL) 的 CycleGAN
### 2. 訓練資料
```
[datasets]/
└──[smi2real]/
  └──[joint]/
     ├──[trainA]/
     |  ├──[real0.jpg]
     |  ├──......
     |  └──[real400.jpg]
     └──[trainB]/
        ├──[smi0.jpg]
        ├──......
        └──[smi400.jpg]
```
### 3. 訓練
```
python train.py --dataroot ./datasets/smi2real --name joint_cyclegan --model cycle_gan
```
### 4. 轉換
```
python test.py --dataroot ./datasets/smi2real --name joint_cyclegan --model cycle_gan
```