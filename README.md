# cas_classification
A high-precision protein classification model

不同分类精度
cas1:0.9992181391712275, 
cas2:0.9989821882951654, 
cas3:1.0, 
cas4:1.0, 
cas5:0.9932659932659933, 
cas6:1.0, 
cas7:1.0, 
cas8:0.9970238095238095, 
cas9:1.0, 
cas10:1.0, 
cas12:1.0, 
cas13:0.9344262295081968

# 项目结构
data/
    train/
    val/
    test/
experiments/
model/
    *.py
build_dataset.py
train.py
search_hyperparams.py
synthesize_results.py
evaluate.py

data/：将包含项目的所有数据，具有明确的训练/开发/测试分割
experiments：包含不同的实验（主要是不同ESM 650M\3B\15B以及不同learning_rate）
model/：定义训练或评估中使用的模型和函数的模块
collect_dataset.py：从NCBI获取数据
build_dataset.py：创建或转换数据集，将分割构建为train/val/test
train.py：在输入数据上训练模型，并在开发集上评估每个时期
search_hyperparams.py：使用不同的超参数运行train.py多次
synthesize_results.py：探索目录中的不同实验并显示结果的漂亮表格
evaluate.py：在测试集上评估模型

params.json：超参数列表，json格式
train.log：训练日志（我们打印到控制台的所有内容）
last_weights：最后 5 个 epoch 保存的权重
best_weights：最佳权重（基于开发准确性）

# 运行
python train.py --model_dir experiments/esm_650M
