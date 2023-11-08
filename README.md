# cas_classification
A high-precision protein classification model

不同Crispr cas蛋白分类精度
|  cas类别   | 精度  |
|  ----  | ----  |
| cas1  | 0.99 |
| cas2  | 0.99 |
| cas3  | 1.0 |
| cas4  | 1.0 |
| cas5  | 0.99 |
| cas6  | 1.0 |
| cas7  | 1.0 |
| cas8  | 0.99 |
| cas9  | 1.0 |
| cas10  | 1.0 |
| cas12  | 1.0 |
| cas13  | 0.93 |

不同模型的精度
|                      |   accuracy |       loss |
|:---------------------|-----------:|-----------:|
| experiments/esm_650M |   0.998067 | 0.00729729 |
| experiments/esm_15B  |   0.9988   | 0.00428334 |
| experiments/esm_3B   |   0.9992   | 0.00305734 |

# 项目结构
* data/
    *    train/
    *    val/
    *    test/
* experiments/
* model/
    *    *.py
* collect_dataset.py
* build_dataset.py
* train.py
* search_hyperparams.py
* synthesize_results.py
* evaluate.py

1. data/：将包含项目的所有数据，具有明确的训练/验证/测试分割
2. experiments：包含不同的实验（主要是不同ESM 650M\3B\15B以及不同learning_rate）
3. model/：定义训练或评估中使用的模型和函数
4. collect_dataset.py：从NCBI获取数据
5. build_dataset.py：创建或转换数据集，分割为train/val/test
6. train.py：在train数据集上训练模型，并在val数据集上评估每个epoch
7. search_hyperparams.py：使用不同的超参数运行train.py
8. synthesize_results.py：搜索目录中的不同实验并显示结果表格
9. evaluate.py：在test数据集上评估模型

## 一些重要文件
1. params.json：超参数列表，json格式
2. train.log：训练日志（我们打印到控制台的所有内容）
3. last_weights：最后 5 个 epoch 保存的权重
4. best_weights：最佳权重（基于预测准确性）

# 运行
1. collect_dataset.py (注意修改成你自己的NCBI账号)
2. build_dataset.py （train：val：test=8:1:1）
3. python train.py --model_dir experiments/esm_650M
