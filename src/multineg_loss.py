import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict
import torch.nn.functional as F


class MultipleNegativesRankingLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MultipleNegativesRankingLoss, self).__init__()

        self.reduction = reduction

    def forward(self, questions: Tensor, docs: Tensor, device: torch.device):

        """
        Compute the loss over a batch with two embeddings per example.

        Each pair is a positive example. The negative examples are all other embeddings in embeddings_b with each embedding
        in embedding_a.

        See the paper for more information: https://arxiv.org/pdf/1705.00652.pdf
        (Efficient Natural Language Response Suggestion for Smart Reply, Section 4.4)

        :param embeddings_a:
            Tensor of shape (batch_size, embedding_dim)
        :param embeddings_b:
            Tensor of shape (batch_size, embedding_dim)
        :return:
            The scalar loss
        """

        loss = torch.zeros(()).to(device)

        for i in range(len(questions)):
            for doc in docs[i]:
                similarity = torch.matmul(questions[i].mean(dim=0), doc['doc'].mean(dim=0).t())
                # similarity = F.cosine_similarity(questions[i].mean(dim=0).unsqueeze(0), doc['doc'].mean(dim=0).unsqueeze(0))
                if doc['is_ans']:
                    loss = torch.sub(loss, similarity)
                else:
                    loss = torch.add(loss, similarity)

        # if self.reduction == 'mean':
        #     embeddings_a = torch.mean(embeddings_a, dim=1)
        #     embeddings_b = torch.mean(embeddings_b, dim=1)

        # scores = torch.matmul(embeddings_a, embeddings_b.t())
        # diagonal_mean = torch.mean(torch.diag(scores))
        # mean_log_row_sum_exp = torch.mean(torch.logsumexp(scores, dim=1))
        # return -diagonal_mean + mean_log_row_sum_exp
        return loss

