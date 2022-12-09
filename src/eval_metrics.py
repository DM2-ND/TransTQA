# import heapq
import json
import logging
# import os
# import numpy as np
from collections import defaultdict

# import functools
import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from typing import List, Dict, Optional, Callable, Iterable
# import torch.nn.functional as F

from data_processor import QaInputFeature
from torch.utils.data import TensorDataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration
)

from data_processor import get_tech_qa_features
from torch.utils.data import Dataset, TensorDataset
import pickle


def load_data(location, corpus):
    with open(location, encoding='utf-8') as f:
        queries = json.load(f)

    with open(corpus, encoding='utf-8') as f:
        corpus = json.load(f)

    features = []
    for _, query in enumerate(tqdm(queries, desc='Featurizing queries')):
        question, doc_ids = f'{query["QUESTION_TITLE"]}.{query["QUESTION_TEXT"]}', query['DOC_IDS']
        documents = [{
            'text': corpus[doc]['text'],
            'id': doc,
            'title': corpus[doc]['title']
        } for doc in doc_ids]
        query.update(
            {
                'question': question,
                'candidate_docs': documents,
            }
        )
        features += [query]
    
    return features


def predict_output(
    device, dataset: str, eval_features: List[QaInputFeature], eval_dataset: TensorDataset, tokenizer,
    model: nn.Module, model_type: str, output_dir: str, predict_batch_size: int, epoch: Optional[int] = -1,
    rankat: Optional[List] = [1, 5, 10], fp16: bool = True, top_k: int = 10
):

    eval_dataset = load_data('data/techqa/dev_answers_is_correct.json', 'data/techqa/training_dev_technotes.json')

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=predict_batch_size, collate_fn=lambda x: x)

    reciprocal_rank = 0
    total_count = [True if f['ANSWERABLE'] == 'Y' and f['is_correct'] == 'Y' else False for f in eval_dataset].count(True)
    counts = defaultdict(int)

    progress_bar = tqdm(total=len(eval_dataset), desc="eval")

    final_ranking = []
    span_to_gen_doc = []

    for batch in eval_dataloader:
        for b in batch:
            if b['ANSWERABLE'] != 'Y' or b['is_correct'] != 'Y':
                del b['candidate_docs']
                # b['DOC_IDS'] = b['DOC_IDS'][:top_k]
                final_ranking += [b]
                progress_bar.update(n=1)
                continue
            question = b['question']
            candidate_docs = b['candidate_docs']
            answer = b['ANSWER']
        
            # target_encoding = tokenizer(question,max_length=512,truncation=True,return_tensors='pt').input_ids

            def split_on_window(sequence, limit, stride):
                split_sequence = sequence.split()
                if len(split_sequence) < limit:
                    return iter([sequence])
                iterators = [iter(split_sequence[index::stride]) for index in range(limit)]
                return zip(*iterators)

            all_ids = []
            span_doc_id = {}
            all_questions = []
            for doc in candidate_docs:
                # question_encoding = tokenizer(f'{question}. {doc["title"]}',max_length=512,truncation=True,return_tensors='pt').input_ids
                split = doc['text'].split()
                # for span in [ doc['text'][i:i+2000] for i in range(0, len(doc['text']), 2000) ]:
                for span in split_on_window(doc['text'], 100, 100):
                    # all_ids += [f'Passage: {doc["title"]}. {" ".join(span)}. What is a technical issue that this passage answers.']
                    all_ids += [
                        f'Passage: {" ".join(span)}. Please generate a how-to problem statement for this passage.',
                        f'Cause: {" ".join(span)}. Please generate a problem statement for the above cause.'
                    ]
                    span_doc_id[len(all_ids)-1] = doc['id']
                    span_doc_id[len(all_ids)-2] = doc['id']
                    all_questions += [f'{question}', f'{question}']
                # all_ids.append(f'Please write a question based on this passage: {doc["title"]}. {doc["text"]}')
            # answer_idx = [i for i, string in enumerate(all_ids) if answer in string]

            input_encoding = tokenizer(all_ids,padding='longest',max_length=512,pad_to_multiple_of=8,truncation=True,return_tensors='pt')
            target_encoding = tokenizer(all_questions, padding='longest', max_length=512,truncation=True,pad_to_multiple_of=8,return_tensors='pt').input_ids

            context_tensor, attention_mask = input_encoding.input_ids, input_encoding.attention_mask
            # target_encoding = torch.repeat_interleave(target_encoding, len(context_tensor), dim=0)
            # target_encoding = torch.stack(target_encoding)

            context_tensor = context_tensor.to(device)
            attention_mask = attention_mask.to(device)
            target_encoding = target_encoding.to(device)

            shard_size = 64
            sharded_nll_list = []

            with torch.no_grad():
                span_to_gen_doc.append({
                    'span': all_ids[0],
                    'generated': tokenizer.decode(t5model.generate(context_tensor[:1])[0], skip_special_tokens=True)
                })
            
            shard_size = 4
            for i in range(0, len(context_tensor), shard_size):
                encoder_tensor_view = context_tensor[i: i + shard_size]
                attention_mask_view = attention_mask[i: i + shard_size]
                decoder_tensor_view = target_encoding[i: i + shard_size]
                with torch.no_grad():
                    logits = model(input_ids=encoder_tensor_view,attention_mask=attention_mask_view,labels=decoder_tensor_view).logits
                log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)
                nll = -log_softmax.gather(2, decoder_tensor_view.unsqueeze(2)).squeeze(2)

                avg_nll = torch.sum(nll, dim=1)
                sharded_nll_list.append(avg_nll)

            scores, indexes = torch.topk(-torch.cat(sharded_nll_list), k=len(context_tensor))
            
            indexes = indexes.tolist()

            doc_id_scores = [{'doc': span_doc_id[index], 'score': scores[index]} for index in indexes]
            doc_ids = {}
            for dict in doc_id_scores:
                doc_ids[dict['doc']] = max(doc_ids.get(dict['doc'], 0), dict['score'])
            doc_id_indexes = [k for k, v in sorted(doc_ids.items(), key=lambda item: item[1], reverse=True)]
            # doc_id_indexes = [span_doc_id[index] for index in indexes]

            # doc_id_indexes = list(dict.fromkeys(doc_id_indexes))

            idx = doc_id_indexes.index(b['DOCUMENT'])
            # idxs = [indexes.index(idx) for idx in answer_idx]
            # idxs.sort()
            # idx = idxs[0] if len(idxs) > 0 else 100000000
            for rank in rankat:
                if idx < rank:
                    counts[rank] += 1
            reciprocal_rank += 1/(idx + 1)

            # b['DOC_IDS'] = list(dict.fromkeys(doc_id_indexes))[:top_k]
            b['DOC_IDS'] = list(dict.fromkeys(doc_id_indexes))
            del b['candidate_docs']
            final_ranking += [b]
            progress_bar.update(n=1)
    mean_reciprocal_rank = reciprocal_rank / total_count
    hit_ratios = [v/total_count for k, v in sorted(counts.items(), key=lambda item: item[0])]

    print("Epoch {}: Hit@K during one interval is: {}".format(epoch, hit_ratios))
    print("Epoch {}: MRR during one interval is: {}".format(epoch, mean_reciprocal_rank))

    # with open('data/techqa/generated_without_duplicates.json', 'w') as f:
    #     f.write(json.dumps(span_to_gen_doc))
    with open('data/techqa/rerank.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(final_ranking))


if __name__ == '__main__':
    model_name = "google/flan-t5-xl"
    # model_name = 't5-small'
    # device=torch.device('cpu')
    device=torch.device('cuda')
    t5model = T5ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float32)
    t5tokenizer = T5Tokenizer.from_pretrained(model_name)

    for param in t5model.parameters():
        param.requires_grad = False
    
    t5model = t5model.to(device)

    t5model = t5model.eval()

    predict_output(
        model=t5model,
        tokenizer=t5tokenizer,
        device=device,
        dataset='techqa',
        model_type='',
        eval_features=None,
        eval_dataset=None,
        predict_batch_size=50,
        output_dir=''
    )
