> Build a Large Language Model (From Scratch)

## 第二章：处理文本数据
### 不懂的词
- 神经元之间的连接权重和偏置在训练过程中不断更新。
- 对于那些熟悉独热编码的人来说，上述嵌入层方法本质上只是实现独热编码后再进行矩阵乘法的一种更高效的方式
#### 连续向量
- 向量值是连续的：向量的每个元素都是一个实数值（如0.2, -0.5），而不是离散的整数0或1。
    - 这个ok
- 语义相近的单词（如“猫”和“狗”）它们的向量在空间中的**距离会很近**（余弦相似度高）。vector(“国王”) - vector(“男人”) + vector(“女人”) ≈ vector(“女王”)
    - 这个还没体会
### 不懂的逻辑
- 嵌入层是随机生成的，
```
torch.manual_seed(123)
// 行（词表大小），列（维度）
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
// 取第3行（从0开始）
embedding_layer(torch.tensor([3]))
// token_id -> input_ids行的嵌入向量
// 但embedding_layer是随机的，所以这里看不到token之间的关系？也就是说意思相近的token，嵌入向量不一定近
embedding_layer(input_ids)
```
### 背景知识
- 对计算中向量和张量不熟悉的读者，可以在附录 A 的 A2.2 节中了解更多关于张量的内容。
- 如果你不熟悉神经网络是如何通过反向传播进行训练的，请参阅附录 A 中的 A.4 节，《自动微分简易教程》。