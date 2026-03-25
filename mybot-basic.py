#!/usr/bin/env python3

# This keeps compatibility with libraries that still expect time.clock().
import time

if not hasattr(time, 'clock'):
    time.clock = time.perf_counter

# These imports provide XML parsing, AIML, the volleyball API, CSV handling, similarity matching, logic, image classification, and file selection.
import xml.etree.ElementTree as ET
import aiml
import fivbvis
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
from tensorflow import keras
from PIL import Image
from tkinter import Tk, filedialog

# This sets the image size expected by the trained CNN model.
IMG_SIZE = 160

# These are the output classes used by the image classifier.
IMAGE_CLASSES = ['basketball', 'football', 'golf_ball', 'tennis_ball', 'volleyball']

# This loads the saved image classification model.
model = keras.models.load_model("best_sports_ball_classifier.keras")

# These downloads provide the NLTK resources needed for stopwords and lemmatization.
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# This creates and loads the AIML chatbot rules.
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="mybot-basic.xml")

# These objects are used for first-order logic parsing and logical inference.
read_expr = Expression.fromstring
prover = ResolutionProver()
KB_FILE = "logical-kb-bulgaria-men.csv"


# This converts free text into a safe logical predicate name.
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


# This converts a name into a safe logical constant.
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


# This maps alternative volleyball terms to the canonical predicates used in the knowledge base.
POSITION_ALIASES = {
    "setter": "Setter",
    "middle blocker": "MiddleBlocker",
    "middle_blocker": "MiddleBlocker",
    "middleblocker": "MiddleBlocker",
    "outside hitter": "OutsideHitter",
    "outside_hitter": "OutsideHitter",
    "outsidehitter": "OutsideHitter",
    "wing spiker": "OutsideHitter",
    "opposite": "Opposite",
    "opposite hitter": "Opposite",
    "opposite spiker": "Opposite",
    "libero": "Libero",
    "defensive specialist": "Libero",
    "defensive role": "DefensiveRole",
    "playmaker": "Playmaker",
    "attacker": "Attacker",
    "front row player": "FrontRowPlayer",
    "back row player": "BackRowPlayer",
}

# This list defines which terms should be treated as concepts instead of individual players.
CONCEPT_TERMS = {
    "setter",
    "middle blocker",
    "middle_blocker",
    "middleblocker",
    "outside hitter",
    "outside_hitter",
    "outsidehitter",
    "wing spiker",
    "opposite",
    "opposite hitter",
    "opposite spiker",
    "libero",
    "defensive specialist",
    "defensive role",
    "playmaker",
    "attacker",
    "front row player",
    "back row player",
}


# This normalizes free text so it can be matched consistently.
def normalize_free_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# This converts user wording into the canonical predicate form.
def canonical_predicate(text: str) -> str:
    raw = normalize_free_text(text)

    if raw in POSITION_ALIASES:
        return POSITION_ALIASES[raw]

    return safe_predicate(text)


# This checks whether a text term is a concept such as a role or category.
def is_concept_term(text: str) -> bool:
    return normalize_free_text(text) in CONCEPT_TERMS


# This splits a sentence of the form X is Y into two parts.
def split_is_statement(text: str):
    parts = re.split(r"\s+is\s+", text, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) != 2:
        return None, None
    return parts[0].strip(), parts[1].strip()


# This builds a logical fact for an individual player or entity.
def make_fact(obj: str, subj: str):
    predicate = canonical_predicate(subj)
    constant = safe_constant(obj)
    return read_expr(f"{predicate}({constant})")


# This builds a logical rule for class-level reasoning.
def make_rule(left: str, right: str):
    left_pred = canonical_predicate(left)
    right_pred = canonical_predicate(right)
    return read_expr(f"all x. ({left_pred}(x) -> {right_pred}(x))")


# This decides whether the input should become a fact or a rule.
def make_logic_expr(left: str, right: str):
    if is_concept_term(left):
        return make_rule(left, right)
    return make_fact(left, right)


# This creates the negated version of a logical expression.
def negate(expr):
    return read_expr(f"-({expr})")


# This checks whether a logical expression can be proved from the knowledge base.
def kb_entails(expr, kb):
    try:
        return prover.prove(expr, kb, verbose=False)
    except Exception:
        return False


# This checks whether the exact expression already exists in the knowledge base.
def expr_in_kb(expr, kb):
    return any(str(item) == str(expr) for item in kb)


# This normalizes KB rows loaded from CSV so they can be parsed correctly.
def normalize_logic_row(text: str):
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


# This loads the first-order logic knowledge base from the CSV file.
def load_kb(csv_path: str):
    knowledge_base = []
    data = pd.read_csv(csv_path, header=None, encoding="utf-8")

    for raw in data[0].dropna().astype(str):
        row = normalize_logic_row(raw)
        if row:
            knowledge_base.append(read_expr(row))

    return knowledge_base


# This checks the KB for direct positive and negative contradictions at startup.
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


# This loads the KB and warns if direct contradictions are found.
kb = load_kb(KB_FILE)
contradictions = find_explicit_contradictions(kb)

if contradictions:
    print("WARNING: KB contains explicit contradictions, e.g.:")
    for contradiction in contradictions[:5]:
        print(" ", contradiction, "and -(" + contradiction + ")")


# This class handles CSV-based fallback answers using TF-IDF and cosine similarity.
class VolleyballQA:
    # This loads the CSV data and fits the TF-IDF model on the stored questions.
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

    # This cleans and lemmatizes text before similarity matching.
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

    # This returns the answer whose question is most similar to the user input.
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


# This classifies a selected image using the trained CNN model.
def classify_image_file(file_path):
    try:
        img = Image.open(file_path).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))

        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array, verbose=0)[0]

        best_idx = int(np.argmax(preds))
        best_label = IMAGE_CLASSES[best_idx]
        best_confidence = float(preds[best_idx])

        return best_label, best_confidence

    except Exception as e:
        print("Error classifying image:", e)
        return None, None


# This opens a file picker so the user can choose an image.
def choose_image_file():
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("All files", "*.*")
        ]
    )

    root.destroy()
    return file_path


# This prints the greeting message when the chatbot starts.
print(
    "Welcome to this chat bot. Please feel free to ask questions from me!\n"
    "I am specifically designed for volleyball questions.\n"
    "If you want to know more about a specific match type in 'Volleyball Match' followed by an ID (e.g. 11500)."
)

# This creates the similarity-based volleyball question-answer helper.
volleyball_qa = VolleyballQA("q&a-kb.csv")


# This loop keeps the chatbot running until the user exits.
while True:
    try:
        userInput = input("> ")
    except (KeyboardInterrupt, EOFError):
        print("Bye!")
        break

    # This gets the AIML response for the user's input.
    aiml_response = kern.respond(userInput)

    # This handles AIML commands encoded as #<command>$<parameter>.
    if aiml_response and aiml_response[0] == '#':
        params = aiml_response[1:].split('$')
        cmd = int(params[0])

        # This uses TF-IDF similarity matching when AIML does not know the answer.
        if cmd == 99:
            csv_answer = volleyball_qa.get_answer(userInput)
            if csv_answer:
                print(csv_answer)
            else:
                print("I did not get that, please try again.")

        # This fetches volleyball match information from the FIVBVIS API using a match ID.
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

        # This adds new logical knowledge if it is not already known or contradictory.
        elif cmd == 3:
            left, right = split_is_statement(params[1])

            if left is None:
                print("Sorry, I could not understand that logical statement.")
                continue

            expr = make_logic_expr(left, right)

            if expr_in_kb(expr, kb):
                print(f"OK, I already know that {left} is {right}.")
            elif kb_entails(negate(expr), kb):
                print(f"Sorry, that contradicts what I already know.")
            else:
                kb.append(expr)
                print(f"OK, I will remember that {left} is {right}.")

        # This checks whether a logical statement is correct, incorrect, or unknown.
        elif cmd == 4:
            left, right = split_is_statement(params[1])

            if left is None:
                print("Sorry, I could not understand that logical query.")
                continue

            expr = make_logic_expr(left, right)

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

        # This lets the user select an image and classifies it with the CNN model.
        elif cmd == 5:
            file_path = choose_image_file()
            if not file_path:
                print("No image selected.")
            else:
                label, confidence = classify_image_file(file_path)
                if label is None:
                    print("Sorry, I couldn't classify that image.")
                else:
                    pretty_label = label.replace("_", " ")
                    print(f"I think this image contains a {pretty_label} ({confidence:.1%} confidence).")

        # This exits the chatbot when the AIML exit command is triggered.
        elif cmd == 0:
            print(params[1])
            break

        # This prints any other AIML command response directly.
        else:
            print(aiml_response)

    # This uses the CSV similarity fallback when the AIML response is not a command.
    else:
        csv_answer = volleyball_qa.get_answer(userInput)
        if csv_answer:
            print(csv_answer)
        else:
            if aiml_response:
                print(aiml_response)