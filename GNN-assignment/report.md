Node classification with GNN
==========

本次实验使用GNN进行节点分类。

## Framework

GNN可以用来在图数据上进行特征的抽取。本次节点分类的大致框架设计如下：
1. 前`N-1`层使用GNN，将输入的feature由`input_dim`映射到`hidden_dim`的空间中
2. 最后一层使用MLP，将GNN的输出由`hidden_dim`映射到`output_dim`，经过softmax函数后即得到分类结果。

其中`N`,`hidden_dim`为可以调整的Hyper parameter。

GNN的类型多种多样，本次实验中采用了
1. Graph Convolution Network(GCN)
2. Graph Attention Network(GAT)
3. GraphSAGE(`SAGE` for `SA`mple and aggre`G`at`E`)

这三种比较经典的结构。

本次实验使用[Torch Geometric](https://github.com/rusty1s/pytorch_geometric)来实现所用到的GNN。在Torch Geometric中图卷积操作使用Message Passing的方式完成，其中比较重要的三个函数分别是
1. Message
2. Aggregate
3. Update


Message Passing的Pipeline如下：

$$
 \mathbf{x}_i^{(k)} = \gamma^{(k)} \left( \mathbf{x}_i^{(k-1)}, \square_{j \in \mathcal{N}(i)} \, \phi^{(k)}\left(\mathbf{x}_i^{(k-1)}, \mathbf{x}_j^{(k-1)},\mathbf{e}_{j,i}\right) \right)

$$


其中
- $\mathcal{N}(i)$ 表示当前节点$i$的所有邻居的集合
- $\phi^{(k)}$为Message函数，对当前节点$i$,邻居$j$和可能的边权$e_{j,i}$，给当前节点$i$生成一个新的变量，记作 $z_i^{k-1}$

- $\square$代表Aggregate函数，它接受当前节点的邻居们$j$的隐式表示$z_{j}^{k-1}$，对他们进行聚合操作，得到一个新的变量，记作$z_{j'}^{k-1}$  
- $\gamma^{(k)}$代表Update函数，他接受当前节点$i$在前一层的feature $x_i^{(k-1)}$和Aggregate函数的输出，为当前节点$i$生成下一层的feature $x_i^{k}$

定义好三个函数后，在`forward`中调用`self.propagate`即可由Pytorch Geometric自动调用我们写好的这三个函数。训练和评估的过程与普通的神经网络区别不大。在下一节中，我将用这套Message Passing的方式来介绍所使用到的算法

## Algorithm

### GCN

GCN对首先当前节点$i$周围所有邻居节点$j$的feature按照度的平方根做标准化，按照平均值来聚合。
最后在得到的结果上进行一个线性的变换从而得到当前节点$i$的表示

Aggregate: 
$$z_i^{l-1}\lArr \sum_{j \in N(i) \cup i} \frac{1}{\sqrt{d_id_j}}x_j^{(l-1)}$$  
Update: 

$$x_i^{l}\lArr \mathrm{ReLU}\left(z_{i}^{l-1}W^{l-1} \right)$$

### GAT
GAT在Message中对每一个邻居节点去计算一个`P` head的Attention score。在Aggregate操作中将这些得到的$z_i^{(l-1,p)}$按照`P`组拼接起来，得到$z_i^{(l-1)}$。最后经过一个全连接层得到下一层中节点$i$的表示$x_i^{l}$

Message:
$$z_i^{(l-1,p)}\lArr \sum_{j \in N(i) \cup i} \alpha_{i,j}^px_j^{(l-1)} $$
Aggregate:
$$z_i^{(l-1)} \lArr \mathrm{Concat}(z_i^{(l-1,p)})$$

Update: 

$$x_i^{l}\lArr \mathrm{ReLU}\left(z_{i}^{(l-1)}W^{l-1} \right)$$
### GraphSAGE
GraphSAGE中，首先使用当前节点$i$周围所有邻居节点$j$的feature按照度的平方根做标准化，按照平均值来聚合，得到$z_i^{(l-1)}$。然后将$z_i^{(l-1)}$和现有的feature $x_{i-1}$进行拼接，最后经过一个全连接层得到节点最终的表示$x_{i}$


Message: 

$$z_i^{l-1}\lArr \sum_{j \in N(i) \cup i} \frac{1}{\sqrt{d_id_j}}x_j^{(l-1)}$$  

Aggregate:
$$z_i^{(l-1)} \lArr \mathrm{Concat}(z_i^{l},x_i^{(l-1)})$$

Update: 

$$x_i^{l}\lArr \mathrm{ReLU}\left(z_{i}^{l-1}W^{l-1} \right)$$

## Result

在Cora和CiteSeer上分别进行模型的评估。
采用的参数如下：
- num_layers=2
- batch_size=32
- hidden_dim=32
- dropout=0.0
- epochs=200  
  
使用的Optimizer为`adam`,lr=0.01
得到最好的结果由GraphSAGE得到。

下表是`score()`函数运行中每一次的Acc:
|Round|Dataset|Acc|
|:---:|:---:|:---:|
|1|Cora |0.755|
|2|Cora |0.768|
|3|Cora |0.762|
|4|Cora |0.751|
|5|Cora |0.763|
|1|CITESEER |0.659|
|2|CITESEER |0.633|
|3|CITESEER |0.641|
|4|CITESEER |0.609|
|5|CITESEER |0.645|
  
其中平均准确率`MeanACC`:为`0.6986000000000001`


## 踩坑
在使用Pytorch geometric的过程中遇到一个BUG，在调用
``` python
self.propagate(edge_index,size=(num_node,num_node),x=out)
```
的时候，无论无论如何修改参数，一直报错`Required parameter size is empty`。


在网上检索时发现已经有相关[issue](https://github.com/rusty1s/pytorch_geometric/issues/1760)，但作者宣称已经在`dev`分支上修复，但因为众所周知的原因，无法使用`pip install git+https://github.com/rusty1s/pytorch_geometric.git`安装最新版。

在重装、降级Pytorch geometric均无果后，我将`size`参数改名为`size1`后，代码终于可以正常执行。