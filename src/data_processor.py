import json
import logging
from copy import deepcopy
from os import path
from random import random
import re
from datetime import datetime
from tqdm import tqdm
import torch
from typing import Optional, Iterable, List, Tuple, Dict

from transformers import PreTrainedTokenizer, RobertaTokenizer
from torch.utils.data import TensorDataset


class QaInputFeature(object):
    """A single input vector for the model."""

    __slots__ = ['qid', 'question_masks', 'question_ids', 'answer_masks', 'answer_ids']

    def __init__(self, qid: str, question_ids: List[int], question_masks: List[int],
                answer_ids: List[int], answer_masks: List[int]):

        self.qid = qid
        self.question_ids = question_ids
        self.question_masks = question_masks
        self.answer_ids = answer_ids
        self.answer_masks = answer_masks

    def __repr__(self):
        return self.__str__()

    def __str__(self) -> str:
        return "%s(%s)" % (
            self.__class__.__name__,
            ",".join('%s=%s' %
                     (attribute, getattr(self, attribute)) for attribute in self.__slots__))


def _prepare_tokens(input_tokens: str, tokenizer:PreTrainedTokenizer, max_lens: int):

    indexed_tokens_ids = tokenizer.encode(input_tokens, add_special_tokens=False)[: max_lens-2]
    indexed_tokens_ids = [tokenizer.cls_token_id] + indexed_tokens_ids + [tokenizer.sep_token_id]
    indexed_masks = [1] * len(indexed_tokens_ids)

    while len(indexed_tokens_ids) < max_lens:
        # Mask is 0 for padding tokens
        indexed_tokens_ids.append(tokenizer.pad_token_id)
        indexed_masks.append(tokenizer.pad_token_id)

    return indexed_tokens_ids, indexed_masks


def create_techqa_features(input_query_file: str, max_seq_length: int, tokenizer: PreTrainedTokenizer):

    with open(input_query_file, encoding='utf-8') as infile:
        logging.info('Loading queries and annotations from %s' % infile)
        queries = json.load(infile)
        logging.info('Loaded %d queries from %s' % (len(queries), infile))

    num_generated_features = 0

    features = []

    for i, query in enumerate(tqdm(queries, desc='Featurizing queries')):

        qid = query['QUESTION_ID']
        if 'ANSWERABLE' in query and query['ANSWERABLE'] == 'Y':

            question, answer = query['QUESTION_TEXT'], query['ANSWER']

            # print(tokenizer.cls_token_id, tokenizer.sep_token_id)

            indexed_qids, indexed_qmasks = _prepare_tokens(question, tokenizer, max_seq_length)
            indexed_aids, indexed_amasks = _prepare_tokens(answer, tokenizer, max_seq_length)

            features.append(QaInputFeature(
                qid=qid, question_ids=indexed_qids, question_masks=indexed_qmasks,
                answer_ids=indexed_aids, answer_masks=indexed_amasks,
            ))

            num_generated_features += 1

    logging.info('Generated %d features from %d queries' %
                 (num_generated_features, len(queries)))

    return features


def create_others_features(input_query_file: str, max_seq_length: int, tokenizer: PreTrainedTokenizer):

    with open(input_query_file, encoding='utf-8') as infile:
        logging.info('Loading queries and annotations from %s' % infile)
        queries = infile.readlines()
        logging.info('Loaded %d queries from %s' % (len(queries), infile))

    num_generated_features = 0

    features = []

    for qid, query in enumerate(tqdm(queries, desc='Featurizing queries')):

            question, answer = query.strip('\n').split('\t')

            # print(tokenizer.cls_token_id, tokenizer.sep_token_id)

            indexed_qids, indexed_qmasks = _prepare_tokens(question, tokenizer, max_seq_length)
            indexed_aids, indexed_amasks = _prepare_tokens(answer, tokenizer, max_seq_length)

            features.append(QaInputFeature(
                qid=qid, question_ids=indexed_qids, question_masks=indexed_qmasks,
                answer_ids=indexed_aids, answer_masks=indexed_amasks,
            ))

            num_generated_features += 1

    logging.info('Generated %d features from %d queries' %
                 (num_generated_features, len(queries)))

    return features


def _derive_feature_cache_name(
        input_query_file: str, max_seq_len: int, tokenizer: PreTrainedTokenizer) -> str:

    cache_filename = '{0}_features_with_{1}_max_seq_len_{2}'.format(
        path.splitext(input_query_file)[0],
        tokenizer.__class__.__name__,
        max_seq_len)

    if hasattr(tokenizer, 'basic_tokenizer') and hasattr(tokenizer.basic_tokenizer, 'do_lower_case'):
        cache_filename += '_lower_case_{0}'.format(tokenizer.basic_tokenizer.do_lower_case)

    return cache_filename

''' Load and cache tokenized datas!!! '''
def load_and_cache_examples(args, dataset, tokenizer, evaluate=False, output_examples=False):

    if evaluate:
        input_query_file = args.predict_file

    else:
        input_query_file = args.train_file

    if dataset == 'techqa':
        return  get_tech_qa_features(
                                input_query_file=input_query_file,
                                max_seq_len=args.max_seq_length,
                                tokenizer=tokenizer,
                                output_examples=output_examples
                            )

    else:
        return  get_others_features(
                                input_query_file=input_query_file,
                                max_seq_len=args.max_seq_length,
                                tokenizer=tokenizer,
                                output_examples=output_examples
                            )

def get_dataset_from_features(features):

    all_question_ids = torch.tensor([f.question_ids for f in features], dtype=torch.long)
    all_question_masks = torch.tensor([f.question_masks for f in features], dtype=torch.long)
    all_answer_ids = torch.tensor([f.answer_ids for f in features], dtype=torch.long)
    all_answer_masks = torch.tensor([f.answer_masks for f in features], dtype=torch.long)

    dataset = TensorDataset(
        all_question_ids, all_question_masks, all_answer_ids, all_answer_masks)

    return dataset


def get_tech_qa_features(
        input_query_file: str, max_seq_len: int, tokenizer: PreTrainedTokenizer,
        output_examples: bool = False) -> Iterable[QaInputFeature]:

    logging.info('**** Generating feature caches for examples from %s ****\n' %
                 input_query_file)
    logging.info('Using tokenizer: %s' % tokenizer.__class__.__name__)
    logging.info('Using max sequence length: %s' % max_seq_len)

    cache_file = _derive_feature_cache_name(
        max_seq_len=max_seq_len, tokenizer=tokenizer, input_query_file=input_query_file,)

    if not path.isfile(cache_file):
        logging.info('Did not find previously cached features (%s) for %s, generating them now' %
                     (cache_file, input_query_file))

        features = create_techqa_features(input_query_file=input_query_file,
                tokenizer=tokenizer, max_seq_length=max_seq_len,)

        dataset = get_dataset_from_features(features)

        with open(input_query_file, encoding='utf-8') as infile:
            gold_dict = {q['QUESTION_ID']: q for q in json.load(infile)}

        logging.info("Saving features into cached file %s", cache_file)
        torch.save({"features": features, "dataset": dataset, "examples": gold_dict}, cache_file)

    else:
        logging.info('Skipping featurization and loading features from cache: %s' % cache_file)
        features_and_dataset = torch.load(cache_file)
        dataset = features_and_dataset["dataset"]
        if output_examples:
            gold_dict = features_and_dataset["examples"]
            features = features_and_dataset["features"]

    if output_examples:
        return dataset, features, gold_dict

    return dataset


def get_others_features(
        input_query_file: str, max_seq_len: int, tokenizer: PreTrainedTokenizer,
        output_examples: bool = False) -> Iterable[QaInputFeature]:

    logging.info('**** Generating feature caches for examples from %s ****\n' %
                 input_query_file)
    logging.info('Using tokenizer: %s' % tokenizer.__class__.__name__)
    logging.info('Using max sequence length: %s' % max_seq_len)

    cache_file = _derive_feature_cache_name(
        max_seq_len=max_seq_len, tokenizer=tokenizer, input_query_file=input_query_file,)

    if not path.isfile(cache_file):
        logging.info('Did not find previously cached features (%s) for %s, generating them now' %
                     (cache_file, input_query_file))

        features = create_others_features(input_query_file=input_query_file,
                tokenizer=tokenizer, max_seq_length=max_seq_len,)

        dataset = get_dataset_from_features(features)

        with open(input_query_file, encoding='utf-8') as infile:
            gold_dict = {i: q for i, q in enumerate(infile.readlines())}

        logging.info("Saving features into cached file %s", cache_file)
        torch.save({"features": features, "dataset": dataset, "examples": gold_dict}, cache_file)

    else:
        logging.info('Skipping featurization and loading features from cache: %s' % cache_file)
        features_and_dataset = torch.load(cache_file)
        dataset = features_and_dataset["dataset"]
        if output_examples:
            gold_dict = features_and_dataset["examples"]
            features = features_and_dataset["features"]

    if output_examples:
        return dataset, features, None

    return dataset