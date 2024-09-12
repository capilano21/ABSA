import openai
import wandb
import torch
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer, util
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import defaultdict
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('stopwords')

# Global constants
openai.api_key = "YOUR_KEY_HERE"
client = openai
STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

# Utility functions
def clean_text(text):
    """Removes unnecessary characters and lemmatizes the text."""
    text = re.sub(r'[^\w\s]', '', text.lower())
    text = ' '.join([LEMMATIZER.lemmatize(word) for word in word_tokenize(text) if word not in STOP_WORDS])
    return text

# Load and preprocess data
def load_and_process_data(xml_file):
    """Load XML data and parse it into a dataframe with relevant fields."""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    data = []
    for sentence in root.findall('sentence'):
        text = sentence.find('text').text
        aspect_terms = sentence.find('aspectTerms')
        aspect_categories = sentence.find('aspectCategories')

        aspects = [term.get('term') for term in aspect_terms.findall('aspectTerm')] if aspect_terms else []
        polarities = [term.get('polarity') for term in aspect_terms.findall('aspectTerm')] if aspect_terms else []

        categories = [cat.get('category') for cat in aspect_categories.findall('aspectCategory')] if aspect_categories else []
        category_polarities = [cat.get('polarity') for cat in aspect_categories.findall('aspectCategory')] if aspect_categories else []

        data.append({
            'text': clean_text(text),
            'aspects': aspects,
            'polarities': polarities,
            'categories': categories,
            'category_polarities': category_polarities
        })

    return pd.DataFrame(data)

# Load Sentence Transformer model
def load_model():
    """Load a pre-trained sentence transformer for embeddings."""
    return SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings and calculate cosine similarities
def generate_embeddings_and_similarities(texts, model):
    """Generate sentence embeddings and calculate cosine similarities."""
    embeddings = model.encode(texts, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings, embeddings)
    return cosine_scores

# Create prompts for GPT-3.5 ABSA
def create_prompts_for_gpt3(indices, df, test_idx, top_n=75):
    """Creates the prompt for GPT-3.5 using sentence similarity scores."""
    messages = []
    for idx in indices[test_idx][1:top_n]:  # Top similar sentences, excluding the test sentence
        entry = int(idx)
        prompt = f"{df.iloc[entry]['text']} ASPECT={df.iloc[entry]['aspects']} POLARITY={df.iloc[entry]['polarities']}"
        messages.append({"role": "user", "content": prompt})
    
    # Final prompt to predict
    test_prompt = {"role": "user", "content": f"Return correct ASPECT and POLARITY for the following: {df.iloc[test_idx]['text']}"}
    messages.append(test_prompt)
    return messages

# Generate GPT-3.5 responses
def get_gpt3_responses(test_indices, cosine_similarities, df, model="gpt-3.5-turbo"):
    """Generate responses from GPT-3.5 for ABSA."""
    generations = []
    for i in test_indices:
        messages = create_prompts_for_gpt3(cosine_similarities, df, i)
        response = client.chat.completions.create(model=model, messages=messages)
        generations.append(response["choices"][0]["message"]["content"])
    return generations

# Extract aspects and polarities from GPT-3.5 responses
def extract_aspect_polarity(generations):
    """Extract aspect and polarity from GPT-3 responses."""
    aspect_list, polarity_list = [], []
    
    for response in generations:
        aspect_match = re.search(r"ASPECT=\[(.*?)\]", response)
        polarity_match = re.search(r"POLARITY=\[(.*?)\]", response)
        
        aspect_list.append(eval(aspect_match.group(1)) if aspect_match else [])
        polarity_list.append(eval(polarity_match.group(1)) if polarity_match else [])
    
    return pd.DataFrame({"ASPECT": aspect_list, "POLARITY": polarity_list})

# Custom metric calculation
class MetricTracker:
    def __init__(self):
        self.aspect_count_gt = defaultdict(int)
        self.aspect_count_pred = defaultdict(int)
        self.sentiment_count_gt = defaultdict(int)
        self.sentiment_count_pred = defaultdict(int)

    def update(self, ground_truth, prediction, sentiment_gt, sentiment_pred):
        """Update metrics for aspect and sentiment."""
        if ground_truth:
            for aspect in ground_truth:
                self.aspect_count_gt[aspect] += 1
                if aspect in prediction:
                    self.aspect_count_pred[aspect] += 1
                    if sentiment_gt[ground_truth.index(aspect)] == sentiment_pred[prediction.index(aspect)]:
                        self.sentiment_count_gt[f"{aspect}_{sentiment_gt}"] += 1
                    else:
                        self.sentiment_count_pred[f"{aspect}_{sentiment_gt}"] += 1

# Evaluation and result visualization
def evaluate_results(df, results):
    """Evaluate extracted aspects and visualize results."""
    metrics = MetricTracker()
    for i, row in df.iterrows():
        metrics.update(row['aspects'], results.iloc[i]['ASPECT'], row['polarities'], results.iloc[i]['POLARITY'])
    
    plot_metrics(metrics)

def plot_metrics(metrics):
    """Plot metrics for aspect extraction and sentiment analysis."""
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.bar(metrics.aspect_count_gt.keys(), metrics.aspect_count_gt.values(), alpha=0.5, label='GT')
    plt.bar(metrics.aspect_count_pred.keys(), metrics.aspect_count_pred.values(), alpha=0.5, label='Pred')
    plt.title("Aspect Extraction", fontsize=20)
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.bar(metrics.sentiment_count_gt.keys(), metrics.sentiment_count_gt.values(), alpha=0.5, label='Correct')
    plt.bar(metrics.sentiment_count_pred.keys(), metrics.sentiment_count_pred.values(), alpha=0.5, label='Wrong')
    plt.title("Sentiment Classification", fontsize=20)
    plt.legend()
    plt.grid(True)

    plt.show()

# Main execution flow
if __name__ == "__main__":
    # Load the data
    df = load_and_process_data("/path/to/Restaurants_Train_v2.xml")
    
    # Load sentence transformer model
    sentence_model = load_model()

    # Generate embeddings and cosine similarities
    cosine_similarities = generate_embeddings_and_similarities(df['text'].tolist(), sentence_model)
    
    # Split data into train/test
    test_indices = np.random.choice(len(df), 100, replace=False)

    # Get GPT-3.5 responses
    generations = get_gpt3_responses(test_indices, cosine_similarities, df)

    # Process GPT-3.5 outputs
    results = extract_aspect_polarity(generations)

    # Evaluate and visualize results
    evaluate_results(df.iloc[test_indices], results)
