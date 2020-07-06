
update to transformer 2.3.0
### 如何将bert model 的Tensorflow模型 转换为pytorch模型

convert_bert_original_tf_checkpoint_to_pytorch.py

运行脚本run.sh

后生成对应pytorch_model.bin

---
 chinese bert   
 
 https://github.com/ymcui/Chinese-BERT-wwm/blob/master/README_EN.md

下载 BERT-wwm-ext, Chinese 或者 BERT-wwm, Chinese pytorch模型


-------

transformer 2.1.1
### 如何将bert model 的Tensorflow模型 转换为pytorch模型


下载Tensorflow模型文件
解压缩到文件夹下

应该有

- bert_config.json

- bert_model.ckpt.data-00000-of-00001

- bert_model.ckpt.index

- bert_model.ckpt.meta

- vocab.txt

这几个文件
运行run.sh
后生成对应pytorch_model.bin

具体代码

```
export BERT_BASE_DIR=./

transformers bert \
  $BERT_BASE_DIR/bert_model.ckpt \
  $BERT_BASE_DIR/bert_config.json \
  $BERT_BASE_DIR/pytorch_model.bin

```


原来convert_tf_checkpoint_to_pytorch.py被新版本废除