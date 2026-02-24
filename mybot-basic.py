#!/usr/bin/env python3
import time

if not hasattr(time, 'clock'):
    time.clock = time.perf_counter
import xml.etree.ElementTree as ET
import aiml
import wikipedia
import fivbvis
from fivbvis import Article

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Create a Kernel object.
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="mybot-basic.xml")


class VolleyballQA:
    def __init__(self, csv_file_name="q&a-kb.csv"):
        # Load knowledge base
        self.knowledge_base = pd.read_csv(csv_file_name, encoding='utf-8')

        # Initialize text preprocessing tools
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 3),  # Increased to trigrams for better matching
            min_df=1,
            max_df=0.9
        )

        # Preprocess questions and create TF-IDF matrix
        self.preprocessed_questions = [self.preprocess_text(q) for q in self.knowledge_base.iloc[:, 0]]
        self.tfidf_matrix = self.vectorizer.fit_transform(self.preprocessed_questions)

    def preprocess_text(self, text):
        if pd.isna(text) or not isinstance(text, str):
            return ""
        text = re.sub(r'\(volleyball\)', '', text, flags=re.IGNORECASE)
        # Clean text
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = text.split()

        # Remove stop words and lemmatize
        processed_words = []
        for word in words:
            if word not in self.stop_words and len(word) > 2:
                processed_words.append(self.lemmatizer.lemmatize(word))

        return ' '.join(processed_words)

    def get_answer(self, user_query, threshold=0.25):  # Increased threshold
        # Preprocess query
        processed_query = self.preprocess_text(user_query)

        if not processed_query:
            return None  # Return None instead of error message

        # Vectorize and find similarities
        query_vector = self.vectorizer.transform([processed_query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        # Get best match
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]

        # Return answer if above threshold
        if best_similarity >= threshold:
            return self.knowledge_base.iloc[best_idx, 1]  # Return only the answer
        else:
            return None


# Welcome user
print(
    "Welcome to this chat bot. Please feel free to ask questions from me!\nI am specifically designed for volleyball questions.\nIf you want to know more about a specific match type in 'Volleyball Match' followed by an ID (e.g. 11500).")
volleyball_qa = VolleyballQA("q&a-kb.csv")

# Main loop
while True:
    # Get user input
    try:
        userInput = input("> ")
    except (KeyboardInterrupt, EOFError):
        print("Bye!")
        break

    # Get AIML response
    aiml_response = kern.respond(userInput)

    # Check if AIML response is a command
    if aiml_response and aiml_response[0] == '#':
        params = aiml_response[1:].split('$')
        cmd = int(params[0])

        # If it's command 99 (I don't know), try CSV
        if cmd == 99:
            csv_answer = volleyball_qa.get_answer(userInput)
            if csv_answer:
                print(csv_answer)
            else:
                print("I did not get that, please try again.")

        # If it's command 1 (Wikipedia), process it
        elif cmd == 1:
            try:
                wSummary = wikipedia.summary(params[1], sentences=3, auto_suggest=True)
                print(wSummary)
            except:
                print("Sorry, I do not know that. Be more specific!")

        # If it's command 2 (Volleyball match), process it
        elif cmd == 2:
            match_id = params[1]
            try:
                v = fivbvis.Volleyball()
                # Get the data, requesting specific fields
                match_data = v.getVolleyMatch(no=match_id,
                                              fields="City TeamNameA TeamNameB CountryName DateLocal MatchPointsA MatchPointsB")

                root = ET.fromstring(match_data)

                # Extract data from XML attributes
                city = root.get('City', 'Unknown')
                team_a = root.get('TeamNameA', 'Unknown')
                team_b = root.get('TeamNameB', 'Unknown')
                country = root.get('CountryName', 'Unknown')
                date = root.get('DateLocal', 'Unknown')
                points_a = root.get('MatchPointsA', 'Unknown')
                points_b = root.get('MatchPointsB', 'Unknown')

                # Check if this is a valid match (has at least a match number)
                match_no = root.get('No', '')

                # If the match number doesn't match what we requested or is missing, it's invalid
                if not match_no or match_no != match_id:
                    print(f"No match found with ID {match_id}")

                # Check if all essential fields are 'Unknown' (likely invalid match)
                elif city == 'Unknown' and team_a == 'Unknown' and team_b == 'Unknown':
                    print(f"No match found with ID {match_id}")

                # We have at least some real data
                else:
                    # Check if points exist (not Unknown and not empty)
                    if points_a not in ['Unknown', ''] and points_b not in ['Unknown', '']:
                        print(
                            f"Match {match_id} was played in {city}, {country} on {date} between {team_a} and {team_b} and finished {points_a} - {points_b}.")

                    # We have match info but no points
                    elif team_a not in ['Unknown', ''] and team_b not in ['Unknown', '']:
                        print(
                            f"Match {match_id} was played in {city}, {country} on {date} between {team_a} and {team_b}. No points are available for this match.")

                    # Very limited info
                    else:
                        print(
                            f"Match {match_id} was played in {city}, {country} on {date}. Limited information is available for this match.")

            except ET.ParseError as e:
                print(f"Sorry, I couldn't parse the match data. Error: {e}")
            except Exception as e:
                print(f"Sorry, I couldn't find that volleyball match. Error: {e}")

        # If it's command 0 (exit), handle it
        elif cmd == 0:
            print(params[1])
            break

        # Any other command
        else:
            print(aiml_response)

    # If AIML response is not a command (regular text)
    else:
        # Try CSV first for factual questions
        csv_answer = volleyball_qa.get_answer(userInput)
        if csv_answer:
            print(csv_answer)
        else:
            # If no CSV answer, use AIML response
            if aiml_response:
                print(aiml_response)