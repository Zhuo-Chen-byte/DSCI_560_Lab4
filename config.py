import os


class Config:
    def __init__(self, time_limit_in_seconds: int=60, n_clusters: int=5, n_keywords: int=10) -> None:
        self.scroll_js = 'window.scrollTo(0, document.body.scrollHeight);'
        self.scroll_interval = 1
        
        self.time_limit_in_seconds = time_limit_in_seconds
        self.n_clusters = n_clusters
        self.n_keywords = n_keywords
        
        self.post_data_dir = 'post_data'
        self.clustering_results_dir = 'clustering_results'
        
