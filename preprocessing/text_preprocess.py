import pandas as pd
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords, wordnet
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from typing import Union, List
import re

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
tokenizer = word_tokenize

# Function to map POS tags to WordNet format
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def rid_punctuation(tokens: List[str]) -> List[str]:
    """
    Remove punctuation and special characters from a list of tokens.
    
    Args:
        tokens (List[str]): List of tokens to process.
        
    Returns:
        List[str]: List of tokens with punctuation removed.
    """
    result = []
    for token in tokens:
        if token.isalpha():
            result.append(token)
        elif '-' in token and re.match(r'\w+(?:-\w+)+', token):
            result.append(token)
            result.extend(token.split('-'))
        else:
            # If the token is not purely alphabetic and does not contain hyphenated words, skip it
            continue
    return result

def text_process(text: str) -> str:
    """
    Process the input text by converting it to lowercase and stripping leading/trailing whitespace.
    
    Args:
        text (str): The input text to process.
        
    Returns:
        str: The processed text.
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    # Convert to lowercase and strip whitespace
    text = text.lower().strip()

    # Remove special characters and keep only letters and spaces
    tokens = tokenizer(text)
    # Remove all puctuation and special characters, but keep tokens including hyphenated words
    tokens = rid_punctuation(tokens)

    pos_tags = pos_tag(tokens)
    tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(tag)) for token, tag in pos_tags]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Join tokens back into a single string
    processed_text = ' '.join(tokens)

    return processed_text

def transform_quarter(date: str) -> str:
    """ Transform a date string into a quarter format "YYYY-QX".
    Args:
        date (str): The input date string, e.g., "Oct 4, 2015 to Mar 27, 2016".
    Returns:
        str: The transformed date in the format "YYYY-QX".
    Raises:
        ValueError: If the input date format is invalid.
    """
    
    if not isinstance(date, str):
        raise ValueError("Input must be a string")
    
    # Define a mapping from month abbreviations to quarters
    MtoQ = {
        "Jan": "01",
        "Feb": "01",
        "Mar": "01",
        "Apr": "02",
        "May": "02",
        "Jun": "02",
        "Jul": "03",
        "Aug": "03",
        "Sep": "03",
        "Oct": "04",
        "Nov": "04",
        "Dec": "04"
    }
    # Extract the Month and Year from the date string
    # date = Oct 4, 2015 to Mar 27, 2016

    if "to" in date:
        start = date.split("to")[0].strip()
        end = date.split("to")[1].strip()
    else:
        start = date.strip()
        end = date.strip()
    
    # Regex pattern to match the date format "Month Day, Year" or "Month, Year"
    pattern = re.compile(r'(\w{3})(?:\s+(\d{1,2}))?,\s*(\d{4})')

    quarters = []
    dates = [start, end]
    
    # Process both start and end dates
    for date in dates:
        match = re.match(pattern, date)
        if not match:
            year_match = re.match(r'(\d{4})', date)
            if year_match:
                # If only year is provided, assume it starts in Q1
                year = year_match.group(1)
                quarters.append(f"{year}-Q01")
                continue
            else:
                # print(f"[quarter processing]Invalid date format: {date}")
                quarters.append("NULL")
                continue
        
        month, day, year = match.groups()
        quarter = MtoQ.get(month, "01")
        # Format the date as "YYYY-QX"
        formatted_date = f"{year}-Q{quarter}"
        quarters.append(formatted_date)
    
    # Return the start and end quarters
    return quarters[0], quarters[1]

if __name__ == "__main__":
    # Example usage
    # sample_text = "Following their participation at the Inter-High, the Karasuno High School volleyball team attempts to refocus their efforts, aiming to conquer the Spring tournament instead. When they receive an invitation from long-standing rival Nekoma High, Karasuno agrees to take part in a large training camp alongside many notable volleyball teams in Tokyo and even some national level players. By playing with some of the toughest teams in Japan, they hope not only to sharpen their skills, but also come up with new attacks that would strengthen them. Moreover, Hinata and Kageyama attempt to devise a more powerful weapon, one that could possibly break the sturdiest of blocks. Facing what may be their last chance at victory before the senior players graduate, the members of Karasuno's volleyball team must learn to settle their differences and train harder than ever if they hope to overcome formidable opponents old and new—including their archrival Aoba Jousai and its world-class setter Tooru Oikawa. [Written by MAL Rewrite]"
    # sample_text = "state-of-the-art"
    sample_date = "Oct 4, 2015 to May, 2016"
    # processed = text_process(sample_text)
    processed = transform_quarter(sample_date)
    print(f"Original: {sample_date}\nProcessed: {processed}")

