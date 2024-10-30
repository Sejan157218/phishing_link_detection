import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

import re
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import whois
from datetime import datetime
import time
from googlesearch import search
import requests
from urllib.parse import urlparse, urljoin

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("archive","K_model.pkl")
            preprocessor_path=os.path.join('archive','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            print("preds", preds)
            return preds[0]
        
        except Exception as e:
            raise CustomException(e,sys)

#_______data_____#

def domain_registration_length(domain):
    try:
        res = whois.whois(domain)
        expiration_date = res.expiration_date
        today = time.strftime('%Y-%m-%d')
        today = datetime.strptime(today, '%Y-%m-%d')
        # Some domains do not have expiration dates. The application should not raise an error if this is the case.
        if expiration_date:
            if type(expiration_date) == list:
                expiration_date = min(expiration_date)
            return abs((expiration_date - today).days)
        else:
            return 0
    except:
        return -1

def domain_registration_length1(domain):
    v1 = -1
    v2 = -1
    try:
        host = whois.whois(domain)
        hostname = host.domain_name
        expiration_date = host.expiration_date
        today = time.strftime('%Y-%m-%d')
        today = datetime.strptime(today, '%Y-%m-%d')
        if type(hostname) == list:
            for host in hostname:
                if re.search(host.lower(), domain):
                    v1 = 0
            v1= 1
        else:
            if re.search(hostname.lower(), domain):
                v1 = 0
            else:
                v1= 1  
        if expiration_date:
            if type(expiration_date) == list:
                expiration_date = min(expiration_date)
            return abs((expiration_date - today).days)
        else:
            v2= 0
    except:
        v1 = 1
        v2 = -1
        return v1, v2
    return v1, v2

#################################################################################################################################
#               Domain recognized by WHOIS
#################################################################################################################################

 
def whois_registered_domain(domain):
    try:
        hostname = whois.whois(domain).domain_name
        if type(hostname) == list:
            for host in hostname:
                if re.search(host.lower(), domain):
                    return 0
            return 1
        else:
            if re.search(hostname.lower(), domain):
                return 0
            else:
                return 1     
    except:
        return 1

#################################################################################################################################
#               Unable to get web traffic (Page Rank)
#################################################################################################################################
import urllib

def web_traffic(short_url):
        try:
            rank = BeautifulSoup(urllib.request.urlopen("http://data.alexa.com/data?cli=10&dat=s&url=" + short_url).read(), "xml").find("REACH")['RANK']
        except:
            return 0
        return int(rank)


#################################################################################################################################
#               Domain age of a url
#################################################################################################################################

import json

def domain_age(domain):

    url = domain.split("//")[-1].split("/")[0].split('?')[0]
    show = "https://input.payapi.io/v1/api/fraud/domain/age/" + url
    r = requests.get(show)

    if r.status_code == 200:
        data = r.text
        jsonToPython = json.loads(data)
        result = jsonToPython['result']
        if result == None:
            return -2
        else:
            return result
    else:       
        return -1


#################################################################################################################################
#               Global rank
#################################################################################################################################

def global_rank(domain):
    rank_checker_response = requests.post("https://www.checkpagerank.net/index.php", {
        "name": domain
    })
    
    try:
        return int(re.findall(r"Global Rank: ([0-9]+)", rank_checker_response.text)[0])
    except:
        return -1


#################################################################################################################################
#               Google index
#################################################################################################################################


from urllib.parse import urlencode

def google_index(url):
    #time.sleep(.6)
    user_agent =  'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/48.0.2564.116 Safari/537.36'
    headers = {'User-Agent' : user_agent}
    query = {'q': 'site:' + url}
    google = "https://www.google.com/search?" + urlencode(query)
    data = requests.get(google, headers=headers)
    data.encoding = 'ISO-8859-1'
    soup = BeautifulSoup(str(data.content), "html.parser")
    try:
        if 'Our systems have detected unusual traffic from your computer network.' in str(soup):
            return -1
        check = soup.find(id="rso").find("div").find("div").find("a")
        #print(check)
        if check and check['href']:
            return 0
        else:
            return 1
        
    except AttributeError:
        return 1

#print(google_index('http://www.google.com'))
#################################################################################################################################
#               DNSRecord  expiration length
#################################################################################################################################

import dns.resolver

def dns_record(domain):
    try:
        nameservers = dns.resolver.query(domain,'NS')
        if len(nameservers)>0:
            return 0
        else:
            return 1
    except:
        return 1

#################################################################################################################################
#               Page Rank from OPR
#################################################################################################################################


def page_rank(key, domain):
    url = 'https://openpagerank.com/api/v1.0/getPageRank?domains%5B0%5D=' + domain
    try:
        request = requests.get(url, headers={'API-OPR':key})
        result = request.json()
        result = result['response'][0]['page_rank_integer']
        if result:
            return result
        else:
            return 0
    except:
        return -1




#_______data_____#


# def check_empty_title(url):
#     try:
#         response = requests.get(url, timeout=5)
#         soup = BeautifulSoup(response.text, 'html.parser')
#         title = soup.title.string if soup.title else ""
#         return 1 if not title.strip() else 0  # 1 if title is empty, otherwise 0
#     except requests.RequestException:
#         return 0  # Handle connection issues
    

# def check_domain_in_title(url):
#     try:
#         response = requests.get(url, timeout=5)
#         soup = BeautifulSoup(response.text, 'html.parser')
#         title = soup.title.string if soup.title else ""
#         domain = url.split("//")[-1].split("/")[0]
#         return 1 if domain.lower() in title.lower() else 0
#     except requests.RequestException:
#         return 0


# def get_domain_age(domain):
#     try:
#         domain_info = whois.whois(domain)
#         creation_date = domain_info.creation_date
#         if isinstance(creation_date, list):
#             creation_date = creation_date[0]
#         if creation_date:
#             age = (datetime.now() - creation_date).days
#             return age
#         else:
#             return 0
#     except Exception:
#         return 0
    


# def check_google_index(url):
#     try:
#         results = list(search(url, num_results=1))
#         return 1 if results else 0
#     except Exception:
#         return 0

def get_hyperlink_features(url):
    # Make a request to the webpage
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Get all hyperlink tags
    links = soup.find_all('a', href=True)
    nb_hyperlinks = len(links)  # Total number of hyperlinks
    
    # Parse the base domain for internal link comparison
    domain = urlparse(url).netloc
    
    # Count internal links
    internal_links = 0
    for link in links:
        href = link.get('href')
        if href:
            # Normalize URL to ensure it can be correctly classified as internal or external
            href = urljoin(url, href)
            link_domain = urlparse(href).netloc
            
            # Check if link domain matches the base domain (internal link)
            if link_domain == domain:
                internal_links += 1

    # Calculate ratio of internal hyperlinks to total hyperlinks
    ratio_intHyperlinks = internal_links / nb_hyperlinks if nb_hyperlinks > 0 else 0
    
    return {
        'nb_hyperlinks': nb_hyperlinks,
        'ratio_intHyperlinks': ratio_intHyperlinks
    }

class CustomData:

    def get_data_as_data_frame(self, url):
        try:
            # Parsed URL
            parsed_url = urlparse(url if url.startswith("http") else "http://" + url)
            
            # Initialize features dictionary
            features = {}
            
            # Basic features
            features['length_url'] = len(url)
            features['length_hostname'] = len(parsed_url.hostname) if parsed_url.hostname else 0
            
            # IP address presence
            features['ip'] = 1 if re.match(r'^\d{1,3}(\.\d{1,3}){3}$', parsed_url.hostname) else 0
            
            # Character-based features
            features['nb_dots'] = url.count('.')
            features['nb_qm'] = url.count('?')
            features['nb_eq'] = url.count('=')
            features['nb_slash'] = url.count('/')
            features['nb_www'] = 1 if 'www' in parsed_url.hostname else 0
            
            # Digit ratio in URL and hostname
            features['ratio_digits_url'] = sum(c.isdigit() for c in url) / len(url)
            features['ratio_digits_host'] = sum(c.isdigit() for c in parsed_url.hostname) / len(parsed_url.hostname) if parsed_url.hostname else 0
            
            # TLD in subdomain (e.g., .com in subdomain)
            features['tld_in_subdomain'] = 1 if re.search(r'\.(com|net|org|edu|gov)', parsed_url.hostname.split('.')[0]) else 0
            
            # Prefix-suffix presence
            features['prefix_suffix'] = 1 if '-' in parsed_url.hostname else 0
            
            # Word length characteristics
            subdomains = parsed_url.hostname.split('.') if parsed_url.hostname else []
            features['shortest_word_host'] = min([len(word) for word in subdomains]) if subdomains else 0
            features['longest_words_raw'] = max([len(word) for word in url.split('/')]) if '/' in url else len(url)
            features['longest_word_path'] = max([len(word) for word in parsed_url.path.split('/')]) if parsed_url.path else 0
            
            # Dummy placeholders for external features
            hyperlink_features = get_hyperlink_features(url)
            features['phish_hints'] = 1 if 'phish' in url.lower() else 0
            features['nb_hyperlinks'] = hyperlink_features['nb_hyperlinks']  # You'd need to scrape the page to get the actual count
            features['ratio_intHyperlinks'] = hyperlink_features['ratio_intHyperlinks']  # Also requires scraping
            
            # Placeholder for other features that require external sources or more complex processing
            features['empty_title'] = check_empty_title(url)
            features['domain_in_title'] = check_domain_in_title(url)
            features['domain_age'] = domain_age(url)
            features['google_index'] = google_index(url)
            features['page_rank'] = 0
            print("features", features.values())
            return pd.DataFrame(features, index=[0])

        except Exception as e:
            raise CustomException(e, sys)