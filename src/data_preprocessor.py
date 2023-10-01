from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

import pandas as pd
import nltk

from typing import Tuple

from config import Config


class DataPreprocessor:
    def __init__(self, config: Config) -> None:
        self.config = config
        
        
    def tokenize_messages(self, filename: str) -> Tuple[pd.DataFrame, list]:
        data = pd.read_csv(filename)
        data['tokens'] = data['formatted_message'].apply(word_tokenize)
    
        tokenized_messages = [TaggedDocument(word_tokenize(message.lower()), [i])
            for i, message in enumerate(data['formatted_message'])]
    
        return data, tokenized_messages


    def preprocess_data(self, filename: str) -> pd.DataFrame:
        data, tokenized_messages = self.tokenize_messages(filename)
        
        model = Doc2Vec(vector_size=20)
        model.build_vocab(tokenized_messages)
        model.train(tokenized_messages, total_examples=model.corpus_count, epochs=model.epochs)
        
        data['message_vector'] = data['tokens'].apply(lambda x: model.infer_vector(x))

        return data
