"""
train.py - builds the StudyMate search model from the handbook passages and tests it.

How it works (simple version):
1. Read all passages from book_chunks.json (made from the CBSE Class XII AI handbook).
2. TF-IDF learns every word in the book and how important it is (rare words like "neuron"
   get a high score, common words like "the" get a low score).
3. Every passage becomes a list of numbers (a vector). We save this as index.pkl.
4. To test it, we ask 40 sample questions and check whether the bot finds a passage from the
   correct unit. We also ask some off-topic questions to check that the bot says "not sure".
"""

import json
import re

import joblib
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Words that carry no meaning for searching (English + common Hinglish words)
HINGLISH_STOP_WORDS = {
    "kya", "hai", "hota", "hote", "kaise", "kyun", "kyu", "batao", "bataiye", "ka", "ki", "ke",
    "me", "mein", "se", "ko", "aur", "hain", "kaun", "kaunsa", "samjhao", "matlab", "explain",
    "tell", "define", "please", "plz", "kare", "karo", "kar", "karta", "karte", "karna", "chahiye",
    "banate", "banata", "banaye", "banana", "hoti", "hoto", "bolo", "bata", "dikhao",
}
STOP_WORDS = list(ENGLISH_STOP_WORDS | HINGLISH_STOP_WORDS)

# (question, unit in which the answer is found)
TEST_QUESTIONS = [
    ("What is a DataFrame in Pandas?", 1),
    ("How to handle missing values in a dataset?", 1),
    ("What is linear regression?", 1),
    ("How do I import a CSV file in Python?", 1),
    ("What are the steps of the data science methodology?", 2),
    ("What is a confusion matrix?", {2, 4}),   # covered in Unit 2 and again in Unit 4 (Orange)
    ("How to calculate RMSE?", 2),
    ("How do we evaluate the performance of a model?", 2),
    ("What is a pixel?", 3),
    ("What is computer vision?", 3),
    ("How do machines see images?", 3),
    ("What are the applications of computer vision?", 3),
    ("What is OpenCV?", 3),
    ("What is the Orange data mining tool?", 4),
    ("What are widgets in Orange?", 4),
    ("How to use the Iris dataset in Orange?", 4),
    ("How to make a word cloud in Orange?", 4),
    ("What is big data?", 5),
    ("What are the types of big data?", 5),
    ("What are the 5 Vs of big data?", 5),
    ("What is data stream mining?", 5),
    ("What is a neural network?", 6),
    ("What is an activation function?", 6),
    ("What is backpropagation?", 6),
    ("What are hidden layers?", 6),
    ("neural network kya hota hai", 6),
    ("What is generative AI?", 7),
    ("What is the difference between generative and discriminative models?", 7),
    ("What is a large language model?", 7),
    ("What is a deepfake?", 7),
    ("How to write a good prompt for Gemini?", 7),
    ("What is data storytelling?", 8),
    ("What is Freytag's pyramid?", 8),
    ("What are the elements of a data story?", 8),
    ("How to choose a visualization for data?", {4, 8}),   # covered in Unit 4 and Unit 8
    ("Why is ethics important in data storytelling?", 8),
    ("generative ai kya hai", 7),
    ("big data ke characteristics batao", 5),
    ("computer vision kaise kaam karta hai", 3),
    ("pandas me missing values kaise handle kare", 1),
]

# (question, section number where the answer is) - a stricter test: right SECTION, not just right unit
SECTION_TESTS = [
    ("What is a neural network?", "6.1"),
    ("What are the types of neural networks?", "6.4"),
    ("What are the components of a neural network?", "6.2"),
    ("How does a neural network work?", "6.3"),
    ("What is big data?", "5.1"),
    ("What are the types of big data?", "5.2"),
    ("What are the advantages of big data?", "5.3"),
    ("What are the characteristics of big data?", "5.4"),
    ("What is velocity in big data?", "5.4.1"),
    ("What is generative AI?", "7.1"),
    ("What is a large language model?", "7.5"),
    ("What is the difference between generative and discriminative models?", "7.3"),
    ("What are the applications of generative AI?", "7.4"),
    ("What are the applications of computer vision?", "3.4"),
    ("What are the challenges of computer vision?", "3.5"),
    ("How to resize an image in OpenCV?", "3.7.3"),
    ("What is data mining?", "4.1"),
    ("What is the Orange data mining tool?", "4.2"),
    ("What is train test split?", "2.2.1"),
    ("What is K-fold cross validation?", "2.2.2"),
    ("How do I import a CSV file into a DataFrame?", "1.2"),
    ("How to handle missing values?", "1.3"),
]

# Questions that are NOT in the handbook - the bot should say "not sure"
OFF_TOPIC_QUESTIONS = [
    "What is the capital of France?",
    "Who won the IPL last year?",
    "Tell me a joke",
    "How to cook pasta?",
    "bst chapters",
    "Who is the prime minister of India?",
    "How to play cricket?",
    "What is photosynthesis?",
]

MIN_SCORE = 0.10      # the best passage must be at least this similar to the question
MIN_COVERAGE = 0.6    # and at least 60% of the question's words must exist in the book
# (the same two numbers are used in app.py)


def coverage(question, vectorizer):
    """Share of the question's meaningful words that appear somewhere in the book."""
    words = [w for w in re.findall(r"[a-z0-9]+", question.lower())
             if len(w) > 1 and w not in STOP_WORDS]
    if not words:
        return 0.0
    return sum(w in vectorizer.vocabulary_ for w in words) / len(words)


def is_answerable(question, best_score, vectorizer):
    return best_score >= MIN_SCORE and coverage(question, vectorizer) >= MIN_COVERAGE


TITLE_WEIGHT = 0.3   # how much the section title counts compared to the passage text


def section_title(chunk):
    """Section name without any sentence that got attached to it (cut at the first ':')."""
    return chunk["section"].split(":")[0]


def passage_text(chunk):
    """The text we search in: section name + passage."""
    return chunk["section"] + ". " + chunk["text"]


def combined_scores(question, vectorizer, matrix, title_vectorizer, title_matrix, is_overview):
    """Similarity of the question with every passage = passage text match + section title match."""
    q = question.lower()
    scores = linear_kernel(vectorizer.transform([q]), matrix)[0]
    scores = scores + TITLE_WEIGHT * linear_kernel(title_vectorizer.transform([q]), title_matrix)[0]
    scores[is_overview] = 0
    return scores


def main():
    with open("book_chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Passages in the book: {len(chunks)}")

    vectorizer = TfidfVectorizer(
        stop_words=STOP_WORDS, ngram_range=(1, 2), sublinear_tf=True, lowercase=True
    )
    matrix = vectorizer.fit_transform([passage_text(c) for c in chunks])
    print(f"Words/phrases learned from the book: {len(vectorizer.vocabulary_)}")

    # The "Overview" passages of each unit are only lists of learning objectives, so they are
    # not used as answers. We switch them off by giving them a score of 0 during search.
    is_overview = [c["section"] == "Overview" for c in chunks]

    title_vectorizer = TfidfVectorizer(stop_words=STOP_WORDS, ngram_range=(1, 2), sublinear_tf=True)
    title_matrix = title_vectorizer.fit_transform([section_title(c) for c in chunks])

    joblib.dump(
        {"vectorizer": vectorizer, "matrix": matrix,
         "title_vectorizer": title_vectorizer, "title_matrix": title_matrix},
        "index.pkl",
    )
    print("Saved the search model as index.pkl\n")

    # ---- Test 1: does the bot find the right unit? ----
    top1 = top3 = 0
    wrong = []
    for question, unit in TEST_QUESTIONS:
        expected = unit if isinstance(unit, set) else {unit}
        scores = combined_scores(question, vectorizer, matrix, title_vectorizer, title_matrix, is_overview)
        best = scores.argsort()[::-1][:3]
        units = [chunks[i]["unit"] for i in best]
        top1 += units[0] in expected
        top3 += any(u in expected for u in units)
        if units[0] not in expected:
            wrong.append((question, sorted(expected), units[0], scores[best[0]]))
        if not is_answerable(question, scores[best[0]], vectorizer):
            wrong.append((question + "  [bot would say NOT SURE]", sorted(expected), "-", scores[best[0]]))
    n = len(TEST_QUESTIONS)
    print(f"Test questions: {n}")
    print(f"Correct unit in the best answer : {top1}/{n} = {top1 / n * 100:.1f}%")
    print(f"Correct unit in the top 3 answers: {top3}/{n} = {top3 / n * 100:.1f}%")
    for question, expected, got, score in wrong:
        print(f"  missed: '{question}' (expected Unit {expected}, got Unit {got}, score {score:.2f})")

    # ---- Test 2: does the bot find the right SECTION? (stricter) ----
    s1 = s3 = 0
    for question, number in SECTION_TESTS:
        scores = combined_scores(question, vectorizer, matrix, title_vectorizer, title_matrix, is_overview)
        best = scores.argsort()[::-1][:3]
        found = [chunks[i]["section"].split()[0].rstrip(".") for i in best]
        hit = [f == number or f.startswith(number + ".") for f in found]
        s1 += hit[0]
        s3 += any(hit)
        if not hit[0]:
            print(f"  wrong section: '{question}' (expected {number}, got {found[0]})")
    m = len(SECTION_TESTS)
    print(f"\nSection tests: {m}")
    print(f"Correct section in the best answer : {s1}/{m} = {s1 / m * 100:.1f}%")
    print(f"Correct section in the top 3 answers: {s3}/{m} = {s3 / m * 100:.1f}%")

    # ---- Test 3: does the bot say "not sure" for off-topic questions? ----
    print("\nOff-topic questions (should be below the 'not sure' line):")
    for question in OFF_TOPIC_QUESTIONS:
        scores = combined_scores(question, vectorizer, matrix, title_vectorizer, title_matrix, is_overview)
        status = "ANSWERED (bad)" if is_answerable(question, scores.max(), vectorizer) else "not sure (good)"
        print(f"  '{question}' -> score {scores.max():.2f} -> {status}")


main()
