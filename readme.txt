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

data/：将包含项目的所有数据（通常不存储在 github 上），具有明确的训练/开发/测试分割
experiments：包含不同的实验（将在下一节中解释）
model/：定义训练或评估中使用的模型和函数的模块。我们的 PyTorch 和 TensorFlow 示例有所不同
build_dataset.py：创建或转换数据集，将分割构建为train/val/test
train.py：在输入数据上训练模型，并在开发集上评估每个时期
search_hyperparams.py：使用不同的超参数运行train.py多次
synthesize_results.py：探索目录中的不同实验并显示结果的漂亮表格
evaluate.py：在测试集上评估模型（应在项目结束时运行一次）

experiments运行几个不同模型后的结构可能如下所示（尝试根据您正在运行的实验为目录指定有意义的名称）：
experiments/
    base_model/
        params.json
        ...
    learning_rate/
        lr_0.1/
            params.json
        lr_0.01/
            params.json
    batch_norm/
        params.json

训练后的每个目录将包含多个内容：

params.json：超参数列表，json格式
train.log：训练日志（我们打印到控制台的所有内容）
train_summaries：TensorBoard 的训练摘要（仅限 TensorFlow）
eval_summaries：TensorBoard 的评估摘要（仅限 TensorFlow）
last_weights：最后 5 个 epoch 保存的权重
best_weights：最佳权重（基于开发准确性）

python train.py --model_dir experiments/base_model
python train.py --model_dir experiments/esm_650M