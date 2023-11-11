import os
import requests
import json
#from apiKey import apikey
#from apiKey import serpapi
import yfinance as yf
from yahooquery import Ticker
import openai
from dotenv import load_dotenv
from serpapi import GoogleSearch
import json 
from typing import List
import spacy 
from textblob import TextBlob
import emoji 
import streamlit as st 



# Load environment variables from the .env file
load_dotenv()

# Retrieve the API keys from the environment
serpapi_api_key = os.getenv("SERPAPI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

@st.cache_data
def get_company_news(company_name):
    params = {
        "engine": "google",
        "tbm": "nws",
        "q": company_name,
        "api_key": serpapi_api_key,
    }

    response = requests.get('https://serpapi.com/search', params=params)
    data = response.json()

    return data.get('news_results')

@st.cache_data
def get_tweet_data(company_name):
    
    search_params = {
        "q": company_name,
        "location": "Austin, Texas, United States",
        "hl": "en",
        "gl": "us",
        "api_key": serpapi_api_key, 
    }
    search = GoogleSearch(search_params)
    results = search.get_dict()
    if "twitter_results" in results:
        tweet = results["twitter_results"]
    else:
        tweet = None
    #news_results = data.get('news_results')
    return tweet


@st.cache_data
def write_news_to_file(news, filename):
    with open(filename, 'w') as file:
        for news_item in news:
            if news_item is not None:
                title = news_item.get('title', 'No title')
                link = news_item.get('link', 'No link')
                date = news_item.get('date', 'No date')
                file.write(f"Title: {title}\n")
                file.write(f"Link: {link}\n")
                file.write(f"Date: {date}\n\n")

@st.cache_data
def write_knowledge_to_file(knowledge, filename):
    with open(filename, 'a') as file:
        file.write("\nKnowledge Graph\n")
        if 'title' in knowledge:
            file.write(f"Title: {knowledge['title']}\n")
        if 'description' in knowledge:
            file.write(f"Description: {knowledge['description']}\n")
        if 'knowledge_graph_search_link' in knowledge:
            file.write(f"Knowledge Graph Search Link: {knowledge['knowledge_graph_search_link']}\n")



def replace_emojis(text):
    replaced_text = emoji.demojize(text)
    return replaced_text

@st.cache_data
def write_tweet_to_file(tweet, filename):
    with open(filename, 'a', encoding="utf-8") as file:
        file.write("\nTwitter results\n")
        if 'title' in tweet:
            file.write(f"Title: {tweet['title']}\n")
        if 'link' in tweet:
            file.write(f"Link: {tweet['link']}\n")
        if 'tweets' in tweet:
            for index, t in enumerate(tweet['tweets']):
                replaced_snippet = replace_emojis(t['snippet'])
                file.write(f"Snippet {index + 1}: {replaced_snippet}\n")
                file.write(f"Link {index + 1}: {t['link']}\n")
                if 'published_date' in t:
                    file.write(f"Published Date {index + 1}: {t['published_date']}\n")
                file.write("\n")


@st.cache_data
def get_stock_evolution(company_name, period="1y"):
    # Get the stock information
    stock = yf.Ticker(company_name)

    # Get historical market data
    hist = stock.history(period=period)

    # Convert the DataFrame to a string with a specific format
    data_string = hist.to_string()

    # Append the string to the "investment.txt" file
    with open("investment.txt", "a") as file:
        file.write(f"\nStock Evolution for {company_name}:\n")
        file.write(data_string)
        file.write("\n")

    # Return the DataFrame
    return hist

@st.cache_data
def get_financial_statements(ticker):
    # Create a Ticker object
    company = Ticker(ticker)

    # Get financial data
    balance_sheet = company.balance_sheet().to_string()
    cash_flow = company.cash_flow(trailing=False).to_string()
    income_statement = company.income_statement().to_string()
    valuation_measures = str(company.valuation_measures)  # This one might already be a dictionary or string

    # Write data to file
    with open("investment.txt", "a") as file:
        file.write("\nBalance Sheet\n")
        file.write(balance_sheet)
        file.write("\nCash Flow\n")
        file.write(cash_flow)
        file.write("\nIncome Statement\n")
        file.write(income_statement)
        file.write("\nValuation Measures\n")
        file.write(valuation_measures)

@st.cache_data
def get_data(company_name, company_ticker, period="1y", filename="investment.txt"):
    news_results = get_company_news(company_name)
    tweet = get_tweet_data(company_name)

    if news_results:
        write_news_to_file(news_results, filename)

    if tweet:
        #pass
        write_tweet_to_file(tweet, filename)  # Use the correct function for writing the knowledge graph
    else:
        print("No tweets found.")

    hist = get_stock_evolution(company_ticker)
    get_financial_statements(company_ticker)

    return hist, tweet 

@st.cache_data
def chunk_text(text: str, max_tokens: int = 512, overlap: int = 10) -> List[str]:
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    
    chunks = []
    current_chunk = []
    current_token_count = 0

    for sentence in sentences:
        sentence_doc = nlp(sentence)
        sentence_tokens = [token.text for token in sentence_doc]
        num_tokens_in_sentence = len(sentence_tokens)
        
        if num_tokens_in_sentence > max_tokens:
            start = 0
            end = max_tokens
            while start < num_tokens_in_sentence:
                chunk = " ".join(sentence_tokens[start:end])
                #print(f"Chunk: {chunk}")
                #print(f"Chunk size: {len(chunk.split())}")
                chunks.append(chunk)
                start += max_tokens - overlap
                end = min(start + max_tokens, num_tokens_in_sentence)
            current_chunk = []
            current_token_count = 0
            continue

        if current_token_count + num_tokens_in_sentence > max_tokens:
            chunks.append(" ".join(current_chunk))
            #print(f"Chunk: {' '.join(current_chunk)}")
            #print(f"Chunk size: {len(' '.join(current_chunk).split())}")
            current_chunk = []
            current_token_count = 0

        current_chunk.append(sentence)
        current_token_count += num_tokens_in_sentence

    if current_chunk:
        chunks.append(" ".join(current_chunk))
        #print(f"Chunk: {' '.join(current_chunk)}")
        #print(f"Chunk size: {len(' '.join(current_chunk).split())}")
        
    return chunks

# Process each chunk separately
@st.cache_data
def process_chunks(chunks: List[str]) -> str:
    results = []
    for chunk in chunks:
        chunk_size = 4000
        start_index = 0
        #print(f"Processing chunk: {chunk}")
        while start_index < len(chunk):
            try:
                processed_chunk = chunk[start_index : start_index + chunk_size]
                print(f"Processing chunk segment: {processed_chunk}")
                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=processed_chunk,
                    max_tokens=100,
                )
                results.append(response.choices[0].text.strip())
                start_index += chunk_size
            except Exception as e:
                print(f"Error occurred during API call: {e}")
    final_results = " ".join(results)
    #print(f"Final results: {final_results}")
    return final_results

def perform_sentiment_analysis(text):
    analysis = TextBlob(text)
    sentiment_score = analysis.sentiment.polarity

    if sentiment_score > 0:
        return "Positive"
    elif sentiment_score == 0:
        return "Neutral"
    else:
        return "Negative"

     
def equity_analyst(request):
    print(f"Received request: {request}")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "user",
            "content": f"Given the user request, what is the company name and the company stock ticker ?: {request}?"
        }],
        functions=[{
            "name": "get_data",
            "description": "Get financial data on a specific company for investment purposes",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "The name of the company",
                    },
                    "company_ticker": {
                        "type": "string",
                        "description": "the ticker of the stock of the company"
                    },
                    "period": {
                        "type": "string",
                        "description": "The period of analysis"
                    },
                    "filename": {
                        "type": "string",
                        "description": "the filename to store data"
                    }
                },
                "required": ["company_name", "company_ticker"],
            },
        }],
        function_call={"name": "get_data"},
    )

    message = response["choices"][0]["message"]

    if message.get("function_call"):
        # Parse the arguments from a JSON string to a Python dictionary
        arguments = json.loads(message["function_call"]["arguments"])
        print(arguments)
        company_name = arguments["company_name"]
        company_ticker = arguments["company_ticker"]

        # Parse the return value from a JSON string to a Python dictionary
        hist, tweet = get_data(company_name, company_ticker)
        print(hist)

        with open("investment.txt", "r") as file:
            content = file.read()[:14000]

        content_chunk = chunk_text(content)
        processed_chunk = process_chunks(content_chunk)

        # Perform sentiment analysis on the tweet data
        if tweet:
            tweet_data = ""
            for t in tweet['tweets']:
                tweet_data += t['snippet'] + " "
            sentiment_analysis_result = perform_sentiment_analysis(tweet_data)
            print(sentiment_analysis_result)

        second_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": request
                },
                message,
                {
                    "role": "system",
                    "content": """write a detailed investment thesis based on fundamental analysis to answer
                      the user request as an HTML document. Provide numbers to justify
                      your assertions from annual reports and balance sheet, ideally a lot. Always provide
                     a recommendation to buy the stock of the company
                     or not given the information available. Perform any sentiment analysis if you find data from Twitter."""
                },
                {
                    "role": "assistant",
                    "content": processed_chunk,
                },
            ],
        )

        return (second_response["choices"][0]["message"]["content"], hist)