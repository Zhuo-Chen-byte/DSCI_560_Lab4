from selenium import webdriver
from bs4 import BeautifulSoup

import time
import nltk
import csv

from rake_nltk import Rake
from config import Config


class RedditScrapper:
    def __init__(self, config: Config) -> None:
        self.config = config
    
    
    def get_post_ids_and_names(self):
        driver = webdriver.Chrome()  # You can configure the driver as needed

        # URL of the webpage
        url = 'https://www.reddit.com/r/tech/'

        # Open the webpage with Selenium
        driver.get(url)

        # Get the start time
        start_time = time.time()

        # Scroll down the webpage until the time limit is reached
        while True:
            driver.execute_script(self.config.scroll_js)
            time.sleep(self.config.scroll_interval)
    
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
    
            # Check if the time limit is reached
            if elapsed_time >= self.config.time_limit_in_seconds:
                break

        # Get the updated page source after scrolling
        page_source = driver.page_source

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(page_source, 'html.parser')

        # Initialize a list to store post names
        post_ids= []
        post_names = []
        
        for a in soup.find_all('a', href=True):
            href = a['href']
        
            if href.startswith('/r/tech/comments/'):
                # Split the URL by "/" and get the last part
                url_parts = href.split("/")
            
                if len(url_parts) > 3:
                    post_ids.append(url_parts[4])
                    post_names.append(url_parts[5])
        
        # Close the WebDriver
        driver.quit()
        
        return soup, post_names, post_ids
    
    
    def get_timestamps(self, soup) -> list:
        timestamps = soup.find_all(attrs={'created-timestamp': True})

        return [element['created-timestamp'] for element in timestamps]
    
    
    def get_formatted_messages(self, soup) -> list:
        messages = soup.find_all('div', class_='font-bold text-neutral-content-strong m-0 text-18 mb-2xs xs:mb-xs')
        formatted_messages = []
        
        for message in messages:
            formatted_messages.append(message.text.strip())
            
        return formatted_messages
        
    
    def get_keywords_list(self, formatted_messages) -> list:
        nltk.download('stopwords')
        nltk.download('punkt')
        
        rake_nltk_var = Rake()
        
        keywords_list = []
        
        for message in formatted_messages:
            rake_nltk_var.extract_keywords_from_text(message)
            keywords_list.append(rake_nltk_var.get_ranked_phrases())
        
        return keywords_list
    
    
    def get_messages(self):
        soup, post_ids, post_names = self.get_post_ids_and_names()
        formatted_timestamps = self.get_timestamps(soup)
        formatted_messages = self.get_formatted_messages(soup)
        keywords_list = self.get_keywords_list(formatted_messages)
        post_links = [f'https://www.reddit.com/r/tech/comments/{post_id}/{name}/' for post_id, name in zip(post_ids, post_names)]
                
        data = [{'post_id': id, 'post_name': name, 'formatted_timestamp': timestamp, 'formatted_message': message, 'keywords': keywords, 'post_link': link}
            for id, name, timestamp, message, keywords, link in zip(post_ids, post_names, formatted_timestamps, formatted_messages, keywords_list, post_links)]

        # Write data to CSV file
        with open(f'{self.config.post_data_dir}/messages.csv', 'w', newline='', encoding='utf-8') as file:
            fieldnames = ['post_id', 'post_name', 'formatted_timestamp', 'formatted_message', 'keywords', 'post_link']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        
        print('CSV file message.csv has been created.')
