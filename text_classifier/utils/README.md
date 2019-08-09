`base_model`：定义了模型的基础类，根据这个基础类派生专用类，包含一些初始化方法，保存和还原模型的方法。



`base_trainer`：定义了模型的训练类，定义了单步`train_step`，单个epoch`train_epoch`以及单次`train_all`的方法。



`data_gen`：定义了生成数据的类，在模型的训练过程中逐步生成数据。



`data_loader`：定义了数据准备的类，包括把单词转换为index，添加`UNK, PAD`等标记，为模型训练准备数据。



`dirs`：创建不存在的文件夹。



`logger`：定义训练过程中的记录文件，用于`tensorboard`可视化。



`metric`：定义一些评价标准，包含`accuracy`, `precision`等。