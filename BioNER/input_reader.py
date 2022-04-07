# -*- coding: utf-8 -*-
# @Time    : 2021/12/8 下午3:54
# @Author  : cp
# @File    : input_reader.py
import os
import json
from tqdm import tqdm
import numpy as np
from abc import abstractmethod, ABC
from collections import OrderedDict
from logging import Logger
from typing import Iterable, List
from transformers import BertTokenizer
from BioNER.entities import Dataset, EntityType, Entity, Document


class BaseInputReader(ABC):
    def __init__(self, types_path: str, tokenizer: BertTokenizer, logger: Logger = None):

        types = json.load(open(types_path), object_pairs_hook=OrderedDict)

        self._entity_types = OrderedDict()
        self._idx2entity_type = OrderedDict()

        # entities
        for i, (key, v) in enumerate(types['entities'].items()):
            entity_type = EntityType(key, i, v['short'], v['verbose'])
            self._entity_types[key] = entity_type
            self._idx2entity_type[i] = entity_type

        self._datasets = dict()
        self._tokenizer = tokenizer
        self._logger = logger
        self._vocabulary_size = tokenizer.vocab_size
        self._context_size = -1

    @abstractmethod
    def read(self, datasets):
        pass

    def get_dataset(self, label) -> Dataset:
        return self._datasets[label]

    def get_entity_type(self, idx) -> EntityType:
        entity = self._idx2entity_type[idx]
        return entity

    def _calc_context_size(self, datasets: Iterable[Dataset]):
        sizes = []

        for dataset in datasets:
            for doc in dataset.documents:
                sizes.append(len(doc.encoding))

        context_size = max(sizes)
        return context_size

    def _log(self, text):
        if self._logger is not None:
            self._logger.info(text)

    @property
    def datasets(self):
        return self._datasets

    @property
    def entity_types(self):
        return self._entity_types

    @property
    def entity_type_count(self):
        return len(self._entity_types)

    @property
    def vocabulary_size(self):
        return self._vocabulary_size

    @property
    def context_size(self):
        return self._context_size

    def __str__(self):
        string = ""
        for dataset in self._datasets.values():
            string += "Dataset: %s\n" % dataset
            string += str(dataset)

        return string

    def __repr__(self):
        return self.__str__()


class JsonInputReader(BaseInputReader):
    def __init__(self, types_path: str, tokenizer: BertTokenizer, logger: Logger = None):
        super().__init__(types_path, tokenizer, logger)

        self.word_embedding = np.load(os.path.dirname(types_path) + "/embeddings.npy")
        self.word2inx = json.load(open(os.path.dirname(types_path) + "/vocab.json", "r"))
        self.knowledge_map = json.load(open(os.path.dirname(types_path) + "/Knowledge_Vocabulary.json", "r"))
        self.knowledge_embedding = np.load(os.path.dirname(types_path) + "/Knowledge_Embeddings.npy")
        self.concept_embedding = np.load(os.path.dirname(types_path) + "/Concept_Embeddings.npy")

    def read(self, dataset_paths):
        for dataset_label, dataset_path in dataset_paths.items():
            dataset = Dataset(dataset_label, self._entity_types)
            self._parse_dataset(dataset_path, dataset)
            self._datasets[dataset_label] = dataset

        self._context_size = self._calc_context_size(self._datasets.values())

    def _parse_dataset(self, dataset_path, dataset):
        input_data = json.load(open(dataset_path, 'r')) ## todo test 1000 samples

        for document in tqdm(input_data, desc="Parse dataset '%s'" % dataset.label):
            self._parse_document(document, dataset)

    def _parse_document(self, document, dataset) -> Document:


        sent = document['context'].split(" ")
        evidence = document['evidence']# description //evidence
        cui = document['CUI'].split(" ")
        Semantic_Type = document['SemanticType'].split(" ")
        query = document['query']
        span_position = document["span_position"]
        ner_category = document['entity_label']


        doc_tokens, doc_encoding,query_len,evidence_len = self._parse_tokens(sent, evidence, query, dataset)
        entities = self._parse_entities(span_position, ner_category,doc_tokens, dataset)
        doc_cui, doc_sem_type = self._parse_know(cui, Semantic_Type)

        document = dataset.create_document(doc_tokens, entities, doc_encoding,doc_cui, doc_sem_type,query_len,evidence_len)

        return document

    def _parse_tokens(self, jtokens, evidence, query, dataset):
        """
        jtokens:a list
        evidence: string
        dataset:
        """
        doc_tokens = []
        doc_encoding = [self._tokenizer.convert_tokens_to_ids('[CLS]')]

        for i, token_phrase in enumerate(jtokens):
            token_encoding = self._tokenizer.encode(token_phrase, add_special_tokens=False)
            span_start, span_end = (len(doc_encoding), len(doc_encoding) + len(token_encoding))
            if token_phrase in self.word2inx:
                word_index = self.word2inx[token_phrase]
            else:
                word_index = self.word2inx["<unk>"]
            token = dataset.create_token(i, span_start, span_end, token_phrase, word_index)
            doc_tokens.append(token)
            doc_encoding += token_encoding
        ## todo truncation sequence and evidence
        evidence_len = 0
        if (len(doc_encoding) < 450) and len(evidence)>0:
            evidence_id = self._tokenizer.encode(evidence, add_special_tokens=False)
            evidence_len = len(evidence_id)
            doc_encoding += evidence_id
            if len(doc_encoding) > 450:
                evidence_len = evidence_len - (len(doc_encoding)-450)
                evidence_len = evidence_len if evidence_len>0 else 0
                doc_encoding=doc_encoding[:450]
        else:
            doc_encoding = doc_encoding[:450]

        doc_encoding += [self._tokenizer.convert_tokens_to_ids('[SEP]')]
        query_id = self._tokenizer.encode(query, add_special_tokens=False)
        doc_encoding += query_id
        doc_encoding += [self._tokenizer.convert_tokens_to_ids('[SEP]')]

        return doc_tokens, doc_encoding,len(query_id),evidence_len

    def _parse_entities(self, jentities_position, entity_category, doc_tokens, dataset) -> List[Entity]:
        """
        if jentities_position is null list and return null entities list
        """
        entities = []

        for span_item in jentities_position:
            start, end = span_item.split(";")
            entity_type = self._entity_types[entity_category]
            tokens = doc_tokens[int(start):int(end)]
            phrase = " ".join([t.phrase for t in tokens])
            entity = dataset.create_entity(entity_type, tokens, phrase)
            entities.append(entity)

        return entities

    def _parse_know(self, CUI, semType):

        know_cui, know_semType= [], []
        for cui, sem in zip(CUI, semType):
            know_cui.append(self.knowledge_map[cui] if cui in self.knowledge_map else self.knowledge_map['<unk>'])
            know_semType.append(self.knowledge_map[sem] if sem in self.knowledge_map else self.knowledge_map['<unk>'])

        return know_cui, know_semType
