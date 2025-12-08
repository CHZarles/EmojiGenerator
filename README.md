# Emoji Face 😄 Image Generator

##  Dataset


一个专为生成任务整理的小型高质量 emoji 脸部数据集。
> 图片摘取自全量的emoji数据集  https://www.kaggle.com/code/subinium/emoji-full-emoji-dataset
- Emoji face只收录了人脸形式的 emoji （对应emoji序号1~94）
- 已将透明背景的图片转成白色背景,方便模型拟合
- 每个emoji表情包含多个平台的版本 (Apple、Google ...) 

数据集已上传到[kaggle - emoji_face](https://www.kaggle.com/datasets/chzarles/emoji-face)

### Structure

数据集里包含一些图片和一个自制的标签文件`labels.json`。

图片按照emoji序号分类存放，不同emoji对应的序号可参考 [emoji_number.sjon](./emoji_number.json)

具体形式组织如下，详见 [EMOJI_FACE](./EMOJI_FACE/)
```
EMOJI_FACE/
├── 1
│   ├── Apple_1.png
│   ├── Facebook_1.png
...
│   └── Windows_1.png
├── 2 
│   ├── Apple_1.png
│   ├── Facebook_1.png
...
│   └── Windows_1.png
....
├── 94
│   ├── Apple_94.png
│   ├── Facebook_94.png
...
│   └── Windows_94.png
└── labels.json

```

`labels.json` 是一个标签文件，把每个 emoji（用数字 ID 和字符）按视觉特征分组。
- 顶层字段是若干类别（如 skin_color、eye_shape、mouth_shape 等）；
- 每个类别下面是若干子类（如 smiling_curve、frown_curve 等）。
- 每个子类的值是一个列表，元素形如 [id, "emoji"]，表示编号为 id 的表情属于该子类。

>在进行无条件生成训练时，我发现模型总是习惯于生成某一类特定的 Emoji，这就忽略了数据集中大量丰富的特征信息。由于 Emoji 中的组合特征（如眼型、嘴型、肤色）非常丰富，因此我计划先对图像进行一些细粒度的**特征打标**，在训练过程中用这些标签转化为**条件向量Embedding** 输入模型。这样不仅能强制模型学习所有数据的分布，防止信息丢失，还能在生成时实现对特征的自由组合。


## Reference
