#!/usr/bin/env python3
import time

if not hasattr(time, 'clock'):
    time.clock = time.perf_counter

import xml.etree.ElementTree as ET
import aiml
import fivbvis
from fivbvis import Article
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import nltk
import unicodedata
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sem import Expression
from nltk.inference import ResolutionProver

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Create a Kernel object.
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="mybot-basic.xml")

read_expr = Expression.fromstring
prover = ResolutionProver()
KB_FILE = "logical-kb-bulgaria-men.csv"


def safe_predicate(text: str) -> str:
    text = (text or "").strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", "_", text)
    if not text:
        return "UnknownPredicate"
    if text[0].isdigit():
        text = "P_" + text
    return text[0].upper() + text[1:]


def safe_constant(text: str) -> str:
    text = (text or "").strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", "_", text)
    if not text:
        return "UnknownEntity"
    if text[0].isdigit():
        text = "C_" + text
    return text[0].upper() + text[1:]

POSITION_ALIASES = {
    "setter": "Setter",
    "middle blocker": "MiddleBlocker",
    "middle_blocker": "MiddleBlocker",
    "middleblocker": "MiddleBlocker",
    "outside hitter": "OutsideHitter",
    "outside_hitter": "OutsideHitter",
    "outsidehitter": "OutsideHitter",
    "opposite": "Opposite",
    "libero": "Libero",
}

def canonical_predicate(text: str) -> str:
    raw = (text or "").strip().lower()
    raw = unicodedata.normalize("NFKD", raw)
    raw = "".join(ch for ch in raw if not unicodedata.combining(ch))
    raw = re.sub(r"[^\w\s]", "", raw)
    raw = re.sub(r"\s+", " ", raw).strip()

    if raw in POSITION_ALIASES:
        return POSITION_ALIASES[raw]

    return safe_predicate(text)

def make_fact(obj: str, subj: str):
    predicate = canonical_predicate(subj)
    constant = safe_constant(obj)
    return read_expr(f"{predicate}({constant})")

def negate(expr):
    return read_expr(f"-({expr})")


def kb_entails(expr, kb):
    return prover.prove(expr, kb, verbose=False)


def expr_in_kb(expr, kb):
    return any(str(item) == str(expr) for item in kb)


def normalize_logic_row(text: str) -> str:
    if not isinstance(text, str):
        return text

    text = text.strip()

    def repl(match):
        name = match.group(1)
        args = match.group(2)
        safe_name = re.sub(r"\s+", "_", name.strip())
        return f"{safe_name}({args})"

    text = re.sub(r"([A-Za-z][A-Za-z\s]*?)\s*\(\s*([^)]+?)\s*\)", repl, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_kb(csv_path: str):
    knowledge_base = []
    data = pd.read_csv(csv_path, header=None, encoding="utf-8")

    for raw in data[0].dropna().astype(str):
        row = normalize_logic_row(raw)
        if row:
            knowledge_base.append(read_expr(row))

    return knowledge_base


def find_explicit_contradictions(kb):
    positive = set()
    negative = set()

    for expr in kb:
        text = str(expr).strip().replace(" ", "")
        if "->" in text or text.startswith("all") or "&" in text or "|" in text:
            continue
        if text.startswith("-"):
            negative.add(text[1:])
        else:
            positive.add(text)

    return sorted(positive.intersection(negative))


kb = load_kb(KB_FILE)
contradictions = find_explicit_contradictions(kb)

if contradictions:
    print("WARNING: KB contains explicit contradictions, e.g.:")
    for contradiction in contradictions[:5]:
        print(" ", contradiction, "and -(" + contradiction + ")")


class VolleyballQA:
    def __init__(self, csv_file_name="q&a-kb.csv"):
        self.knowledge_base = pd.read_csv(csv_file_name, encoding='utf-8')
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.9
        )
        self.preprocessed_questions = [self.preprocess_text(q) for q in self.knowledge_base.iloc[:, 0]]
        self.tfidf_matrix = self.vectorizer.fit_transform(self.preprocessed_questions)

    def preprocess_text(self, text):
        if pd.isna(text) or not isinstance(text, str):
            return ""
        text = re.sub(r'\(volleyball\)', '', text, flags=re.IGNORECASE)
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = text.split()

        processed_words = []
        for word in words:
            if word not in self.stop_words and len(word) > 2:
                processed_words.append(self.lemmatizer.lemmatize(word))

        return ' '.join(processed_words)

    def get_answer(self, user_query, threshold=0.25):
        processed_query = self.preprocess_text(user_query)

        if not processed_query:
            return None

        query_vector = self.vectorizer.transform([processed_query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]

        if best_similarity >= threshold:
            return self.knowledge_base.iloc[best_idx, 1]
        return None


print(
    "Welcome to this chat bot. Please feel free to ask questions from me!\n"
    "I am specifically designed for volleyball questions.\n"
    "If you want to know more about a specific match type in 'Volleyball Match' followed by an ID (e.g. 11500)."
)

volleyball_qa = VolleyballQA("q&a-kb.csv")

while True:
    try:
        userInput = input("> ")
    except (KeyboardInterrupt, EOFError):
        print("Bye!")
        break

    aiml_response = kern.respond(userInput)

    if aiml_response and aiml_response[0] == '#':
        params = aiml_response[1:].split('$')
        cmd = int(params[0])

        if cmd == 99:
            csv_answer = volleyball_qa.get_answer(userInput)
            if csv_answer:
                print(csv_answer)
            else:
                print("I did not get that, please try again.")

        elif cmd == 2:
            match_id = params[1]
            try:
                v = fivbvis.Volleyball()
                match_data = v.getVolleyMatch(
                    no=match_id,
                    fields="City TeamNameA TeamNameB CountryName DateLocal MatchPointsA MatchPointsB"
                )

                root = ET.fromstring(match_data)
                city = root.get('City', 'Unknown')
                team_a = root.get('TeamNameA', 'Unknown')
                team_b = root.get('TeamNameB', 'Unknown')
                country = root.get('CountryName', 'Unknown')
                date = root.get('DateLocal', 'Unknown')
                points_a = root.get('MatchPointsA', 'Unknown')
                points_b = root.get('MatchPointsB', 'Unknown')
                match_no = root.get('No', '')

                if not match_no or match_no != match_id:
                    print(f"No match found with ID {match_id}")
                elif city == 'Unknown' and team_a == 'Unknown' and team_b == 'Unknown':
                    print(f"No match found with ID {match_id}")
                else:
                    if points_a not in ['Unknown', ''] and points_b not in ['Unknown', '']:
                        print(
                            f"Match {match_id} was played in {city}, {country} on {date} "
                            f"between {team_a} and {team_b} and finished {points_a} - {points_b}."
                        )
                    elif team_a not in ['Unknown', ''] and team_b not in ['Unknown', '']:
                        print(
                            f"Match {match_id} was played in {city}, {country} on {date} "
                            f"between {team_a} and {team_b}. No points are available for this match."
                        )
                    else:
                        print(
                            f"Match {match_id} was played in {city}, {country} on {date}. "
                            f"Limited information is available for this match."
                        )

            except ET.ParseError as e:
                print(f"Sorry, I couldn't parse the match data. Error: {e}")
            except Exception as e:
                print(f"Sorry, I couldn't find that volleyball match. Error: {e}")

        elif cmd == 3:
            obj, subj = params[1].split(' is ')
            expr = make_fact(obj, subj)

            if expr_in_kb(expr, kb):
                print(f"OK, I already know that {obj} is {subj}.")
            elif kb_entails(negate(expr), kb):
                print(f"Sorry, that contradicts what I already know about {obj}.")
            else:
                kb.append(expr)
                print(f"OK, I will remember that {obj} is {subj}.")

        elif cmd == 4:
            obj, subj = params[1].split(' is ')
            expr = make_fact(obj, subj)

            positive = kb_entails(expr, kb)
            negative = kb_entails(negate(expr), kb)

            if positive and not negative:
                print("Correct")
            elif negative and not positive:
                print("Incorrect")
            elif positive and negative:
                print("I don't know (the knowledge base is inconsistent)")
            else:
                print("I don't know")

        elif cmd == 0:
            print(params[1])
            break

        else:
            print(aiml_response)

    else:
        csv_answer = volleyball_qa.get_answer(userInput)
        if csv_answer:
            print(csv_answer)
        else:
            if aiml_response:
                print(aiml_response)
