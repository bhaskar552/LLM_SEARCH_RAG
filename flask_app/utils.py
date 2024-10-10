import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import anthropic
import os
import threading
import json

load_dotenv()

SERPER_API_KEY = os.getenv('SERPER_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
def check_api_keys():
    if not SERPER_API_KEY:
        raise ValueError("Missing SERPER_API_KEY. Please check your environment variables.")
    if not ANTHROPIC_API_KEY:
        raise ValueError("Missing ANTHROPIC_API_KEY. Please check your environment variables.")


def search_articles(query):
    """
    Searches for articles related to the query using Serper API.

    """
    check_api_keys()

    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": query  # The search query
    })
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }

    # Making the POST request to the API
    response = requests.request("POST", url, headers=headers, data=payload)

    if response.status_code == 200:
        result_data = response.json()
        articles = [{"url": item.get("link"), "title": item.get("title")} for item in result_data['organic']]
        articles = articles[:5]
        return articles
    else:
        print(f"Error with Serper API: {response.status_code}")
        return ""


def fetch_article_content(url, content_list, lock):
    """
    Fetches the article content from the given URL, extracting headings and text.
    The result is stored in a shared list with thread locking.
    """
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract headings and paragraphs
        headings = [h.get_text() for h in soup.find_all(['h1', 'h2', 'h3'])]
        paragraphs = [p.get_text() for p in soup.find_all('p')]

        content = '\n'.join(headings + paragraphs).strip()

        # Use lock to safely append to shared list
        with lock:
            content_list.append(content)
    else:
        print(f"Failed to retrieve content from {url}")


def concatenate_content(articles):
    """
    Fetches and concatenates content from the provided article URLs using threading.
    """
    content_list = []
    threads = []
    lock = threading.Lock()

    # Create and start a thread for each URL
    for article in articles:
        thread = threading.Thread(target=fetch_article_content, args=(article['url'], content_list, lock))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Combine all content into one string
    return "\n\n".join(content_list)


def generate_answer(content, query, conversation_history):
    conversation_string = "\n".join([f"{msg.type}: {msg.content}" for msg in conversation_history])
    check_api_keys()

    prompt = f"""
        You are a knowledgeable AI assistant tasked with delivering a well-informed, context-aware response. Below are the details to consider:

        Provided Information: {content}

        User's Current Query: {query}

        Previous Conversation Context:
        {conversation_string}

        Craft a response that incorporates both the conversation history and the present query for a relevant and thoughtful reply.
        """

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1000,
            temperature=0.1,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        if message.content and len(message.content) > 0:
            return message.content[0].text
        else:
            return "I’m sorry, I wasn’t able to generate a response. Could you kindly restate or clarify your question?"

    except Exception as e:
        print(f"An error occurred in generate_answer: {e}")
        return "Apologies, something went wrong while handling your request. Please try again in a little while.(Maybe purchase credits)"