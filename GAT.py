
# gat.py

# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_c, out_c):
        super(GraphAttentionLayer, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.F = F.softmax

        self.W = nn.Linear(in_c, out_c, bias=False)  # y = W * x
        self.b = nn.Parameter(torch.Tensor(out_c))

        nn.init.normal_(self.W.weight)
        nn.init.normal_(self.b)

    def forward(self, inputs, graph):
        """
        :param inputs: input features, [B, N, C].
        :param graph: graph structure, [N, N].
        :return:
            output features, [B, N, D].
        """

        h = self.W(inputs) 
        outputs = torch.bmm(h, h.transpose(1, 2)) * graph.unsqueeze(
            0)  
        outputs.data.masked_fill_(torch.eq(outputs, 0), -float(1e16))

        attention = self.F(outputs, dim=2) 
        return torch.bmm(attention, h) + self.b 

class GATSubNet(nn.Module):
    def __init__(self, in_c, hid_c, out_c, n_heads):
        super(GATSubNet, self).__init__()
        self.attention_module = nn.ModuleList(
            [GraphAttentionLayer(in_c, hid_c) for _ in range(n_heads)])  # in_c为
        self.out_att = GraphAttentionLayer(hid_c * n_heads, out_c)
        self.act = nn.LeakyReLU()


    def forward(self, inputs, graph):
        """
        :param inputs: [B, N, C]
        :param graph: [N, N]
        :return:
        """
        outputs = torch.cat([attn(inputs, graph) for attn in self.attention_module], dim=-1)  # [B, N, hid_c * h_head]
        outputs = self.act(outputs)

        outputs = self.out_att(outputs, graph)

        return self.act(outputs)


class GATNet(nn.Module):
    def __init__(self, in_c, hid_c, out_c, n_heads):
        super(GATNet, self).__init__()
        self.subnet = GATSubNet(in_c, hid_c, out_c, n_heads)

    def forward(self, data, device):
        graph = torch.tensor(data["graph"]).to(device)  
        flow = data["flow_x"] 
        flow = flow.to(device) 

        B, N = flow.size(0), flow.size(1)
        flow = flow.view(B, N, -1) 
        """
       上面是将这一段的时间的特征数据摊平做为特征，这种做法实际上忽略了时序上的连续性
       这种做法可行，但是比较粗糙，当然也可以这么做：
       flow[:, :, 0] ... flow[:, :, T-1]   则就有T个[B, N, C]这样的张量，也就是 [B, N, C]*T
       每一个张量都用一个SubNet来表示，则一共有T个SubNet，初始化定义　self.subnet = [GATSubNet(...) for _ in range(T)]
       然后用nn.ModuleList将SubNet分别拎出来处理，参考多头注意力的处理，同理

       """
        prediction = self.subnet(flow, graph) 

        return prediction




