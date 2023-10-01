from src.reddit_scrapper import RedditScrapper
from src.data_preprocessor import DataPreprocessor
from src.message_assembler import MessageAssembler


import shutil
import os

from config import Config


if __name__ == '__main__':
    # Configuration Initiation
    time_limit_in_seconds = input('Please enter the time duration (in integer seconds) you want this system to collect reddit post data: ')
    
    while not time_limit_in_seconds.isdigit():
        time_limit_in_seconds = input('Invalid input. Please enter the time duration (in integer seconds) you want this system to collect reddit post data: ')
    
    time_limit_in_seconds = int(time_limit_in_seconds)
    
    print()
    
    n_clusters = input('Please enter the (integer) number of clusters by which you want this system to assemble collected reddit posts into: ')
    
    while not n_clusters.isdigit():
        n_clusters = input('Invalid input. Please enter the (integer) number of clusters by which you want this system to assemble collected reddit posts into: ')
    
    n_clusters = int(n_clusters)
    
    print()
    
    n_keywords = input('Please enter the (integer) number of top keywords by which you want this system to assemble collected reddit posts based on: ')
    
    while not n_keywords.isdigit():
        n_keywords = input('Invalid input. Please enter the (integer) number of keywords by which you want this system to assemble collected reddit posts based on: ')
    
    n_keywords = int(n_keywords)
    
    print()
    
    config = Config(time_limit_in_seconds, n_clusters, n_keywords)
    
    if os.path.isdir(config.post_data_dir):
        print('Removing existing post data ...')
        shutil.rmtree(config.post_data_dir)
        
        print('Existing post data removed.')
    
    os.mkdir(config.post_data_dir)
        
    if os.path.isdir(config.clustering_results_dir):
        print('Removing existing clustering results ...')
        shutil.rmtree(config.clustering_results_dir)
        
        print('Existing clustering results removed.')
    
    os.mkdir(config.clustering_results_dir)
    
    reddit_scrapper = RedditScrapper(config)
    data_preprocessor = DataPreprocessor(config)
    message_assembler = MessageAssembler(config)
    
    reddit_scrapper.get_messages()
    data = data_preprocessor.preprocess_data(f'{config.post_data_dir}/messages.csv')
    message_assembler.summarize_modeling_results(data)
