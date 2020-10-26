import heapq
import json
import logging
import os
import numpy as np
from collections import defaultdict

import functools
import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from typing import List, Dict, Optional, Callable, Iterable
import torch.nn.functional as F

from data_processor import QaInputFeature
from torch.utils.data import TensorDataset


def evalRank(predictions, rankat, total_count):

    best_matches = []

    counts = defaultdict(int)
    reciprocal_rank = 0

    # total_count = predictions.shape[0]

    for idx in range(total_count):

        pred = np.argsort(-predictions[idx, :]).tolist()

        best_matches.append(pred[0])

        for rank in rankat:
            if idx in pred[: rank]:
                counts[rank] += 1

        reciprocal_rank += 1 / (pred.index(idx) + 1)

    mean_reciprocal_rank = reciprocal_rank / total_count

    hit_ratios = [v/total_count for k, v in sorted(counts.items(), key=lambda item: item[0])]

    return hit_ratios, mean_reciprocal_rank



def predict_output(device, dataset: str, eval_features: List[QaInputFeature], eval_dataset: TensorDataset,
                   model: nn.Module, model_type: str, output_dir: str, predict_batch_size: int, epoch: Optional[int] = -1):

    logging.info("**** Starting Predict ****")
    logging.info(" \tNum examples = %d", len(eval_dataset))
    logging.info(" \tNum Epochs = %d", epoch)
    logging.info('\tpredict batch size = %d' % predict_batch_size)
    logging.info("\tmodel type = %s" % model_type)
    model.eval()

    progress_bar = tqdm(total=len(eval_dataset), desc="eval")

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=predict_batch_size)

    questions, responses = [], []

    for batch in eval_dataloader:

        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():

            question_inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
            response_inputs = {'input_ids': batch[2], 'attention_mask': batch[3]}

            question_outputs = model(**question_inputs)[0]
            response_outputs = model(**response_inputs)[0]

            reduced_question = F.normalize(torch.mean(question_outputs, dim=1), p=2, dim=1)
            reduced_response = F.normalize(torch.mean(response_outputs, dim=1), p=2, dim=1)

            question_embeddings = reduced_question.to("cpu").numpy()
            response_embeddings = reduced_response.to("cpu").numpy()

            questions.extend(question_embeddings)
            responses.extend(response_embeddings)
            assert len(questions) == len(responses)

            progress_bar.update(n=len(questions))

    eval_instances = len(questions)

    cosine_scores = np.inner(np.array(questions), np.array(responses))
    hit_ratios, mean_reciprocal_rank = evalRank(cosine_scores, [1, 5], total_count=eval_instances)

    logging.info("Epoch {}: Hit@K during one interval is: {}".format(epoch, hit_ratios))
    logging.info("Epoch {}: MRR during one interval is: {}".format(epoch, mean_reciprocal_rank))

    # with open('further_real_time_out_{}_4.5.txt'.format(dataset), 'a') as fw:
        # fw.write('{}\t{}\n'.format(hit_ratios, mean_reciprocal_rank))
    # fw.close()