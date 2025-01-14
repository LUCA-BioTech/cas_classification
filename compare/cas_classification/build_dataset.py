import csv
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
import torch

def lable_number(row):
    castype = row['type']
    if castype =='./cas/cas1':
        return 0
    elif castype =='./cas/cas2':
        return 1
    elif castype =='./cas/cas3':
        return 2
    elif castype =='./cas/cas4':
        return 3
    elif castype =='./cas/cas5':
        return 4
    elif castype =='./cas/cas6':
        return 5
    elif castype =='./cas/cas7':
        return 6
    elif castype =='./cas/cas8':
        return 7
    elif castype =='./cas/cas9':
        return 8
    elif castype =='./cas/cas10':
        return 9
    elif castype =='./cas/cas12':
        return 10
    elif castype =='./cas/cas13':
        return 11

def load_dataset(path_csv):
    df = pd.read_csv(path_csv)
    df['label'] = df.apply(lable_number, axis=1)
    data = df[["id", "seq","label"]]
    
    # for test
    # data= data[0:1000]
    
    unique_labels = data['label'].unique()
    train_data = pd.DataFrame()    
    valid_data = pd.DataFrame()    
    test_data = pd.DataFrame()

    for label in unique_labels:
        # 获取特定类别的数据
        class_data = data[data['label'] == label]
        
        # 将该类别的数据分为训练集、验证集和测试集
        train_class_data, temp_data = train_test_split(class_data, test_size=0.2, random_state=42)
        valid_class_data, test_class_data = train_test_split(temp_data, test_size=0.5, random_state=42)
        
        # 将训练集、验证集和测试集的数据合并到相应的DataFrame中
        train_data = pd.concat([train_data, train_class_data])
        valid_data = pd.concat([valid_data, valid_class_data])
        test_data = pd.concat([test_data, test_class_data])

    
    return train_data,valid_data,test_data

def save_dataset(dataset, save_dir_file):
    dataset.to_csv(save_dir_file)

if __name__ == "__main__":
    path_dataset = 'data/all_cas.csv'
    msg = "{} file not found. Make sure you have downloaded the right dataset".format(path_dataset)
    assert os.path.isfile(path_dataset), msg

    print("Loading dataset and split train val test dataset ...")
    train_dataset,val_dataset,test_dataset = load_dataset(path_dataset)
    print("- done.")


    save_dataset(train_dataset, 'data/train/train.csv')
    save_dataset(val_dataset, 'data/val/val.csv')
    save_dataset(test_dataset, 'data/test/test.csv')
    print('end')