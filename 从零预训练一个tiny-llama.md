# 从零预训练一个tiny-llama



## 环境准备

1. 创建环境

```
conda create -n tiny-llm python=3.10
conda activate tiny-llm
```



2. 安装依赖

```
pip install numpy==1.23.5 Requests==2.31.0 sentencepiece==0.1.99 tqdm==4.64.1
// 安装适配本机cuda版本的torch，这里本人的cuda版本为12.6
 pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```



## 训练Tokenizer

​		需要为文本处理训练一个Tokenizer。Tokenizer的作用是将文本转换为数字序列，以便模型能够理解和处理。使用的数据集是 TinyStory ，它是一个由GPT-3.5和GPT-4生成的小型故事数据集，包含简短的故事，且词汇量有限。在这个任务中，我们采用字符级Tokenizer，将文本中的每个字符映射为对应的数字。训练完成后，我们得到的 Tokenizer 能够将文本转换为数字序列，也可以将数字序列还原为文本。

```
python train_vocab.py --download True --vocab_size 4096
```

自动生成两个文件：`tok4096.model` 和 `tok4096.vocab`，其中 `tok4096.model` 是我们训练好的模型文件，位于 `data` 目录下。这个文件可以用于将文本数据转换为 `Token` 序列，也可以将 `Token` 序列还原为文本。

为了更便捷地使用这个 `Tokenizer`，我们还在 `tokenizer.py` 文件中定义了一个 `Tokenizer` 类。这个类封装了 `Tokenizer` 的常用操作，例如文本编码和解码功能，并支持加载我们训练好的模型文件。通过这个类，我们可以轻松地将文本转换为模型可接受的数字序列，或将预测结果转化为可读的文本。



## 数据预处理

在训练模型之前，首先需要对数据进行预处理。这一步的核心任务是将文本数据转换为模型能够理解的数字序列。具体来说，文本中的每个字符、单词或子词都需要被映射为一个唯一的数字 ID，这样模型才能处理这些数据。

```
python preprocess.py
```

在这部分中，首先定义了 `process_shard` 函数，用于处理数据分片。该函数的主要功能是将文本数据分词后，转换为更高效的二进制文件格式，以便后续更快速地加载和处理数据。

接下来，我们定义了 `pretokenize` 函数，用于批量处理多个数据分片。通过这一函数，所有数据可以并行处理，进一步加快预处理的速度。

然后，我们设计了一个 `PretokDataset` 类，用于加载已预处理好的数据集。我们继承了 `torch.utils.data.IterableDataset` 来定义该数据集，这使得我们可以更灵活、高效地处理数据。在这个类中，核心是 `__iter__` 方法，它负责生成用于训练的数据批次。

最后，我们还定义了一个 `Task` 类，专门用于迭代数据集，并生成模型所需的输入和目标输出。这一部分的设计确保了数据流的顺畅对接，为模型训练提供了标准化的数据输入。可以通过以下代码来测试预处理后的数据集。



## 训练模型

在完成数据预处理后，我们就可以开始训练模型了。我们使用的模型是一个与 LLaMA2 结构相同的 Decoder-only Transformer 模型，采用 PyTorch 实现。具体的实现细节已经包含在 `model.py` 文件中，在此不再赘述。该源码中包含详细的中文注释，此外我们在之前的文章中也对模型架构进行了深入介绍。

在模型部分，建议重点关注生成式模型如何生成 token 的过程。可以参考 `model.py` 文件中的 `Transformer` 类，尤其是 `generate` 方法的实现，它展示了模型如何基于已有的上下文生成后续 token 的机制。

在 generate 方法中，我们首先获取序列中最后一个位置的 logits，然后基于这些 logits 生成新的 token。接着，生成的新 token 会被添加到序列中，模型随后会继续生成下一个 token。通过这种迭代过程，我们能够生成完整的文本。接下来，您可以使用以下命令开始训练模型。

```
python train.py
```

![image-20240923204722698](../image/image-20240923204722698.png)



## 使用模型生成文本

模型还在跑~~~