"""
prepare_book.py - turns the CBSE Class XII AI Student Handbook (text file) into
small passages ("chunks") that the chatbot can search.

Run:  python prepare_book.py
(You only need this if you want to rebuild book_chunks.json from the handbook text.)

What it does:
1. Reads the handbook text line by line.
2. Notices which UNIT and which numbered section (like "6.2 Types of Neural Networks") we are in.
3. Throws away page numbers, divider lines and the EXERCISES part at the end of each unit.
4. Cuts the remaining text into passages of about 100 words, each tagged with its unit and section.
5. Saves everything in book_chunks.json.
"""

import json
import re
import sys

INPUT_FILE = sys.argv[1] if len(sys.argv) > 1 else "AI_Student_HandbookXII.pdf.txt"
OUTPUT_FILE = "book_chunks.json"

UNIT_TITLES = {
    1: "Python Programming-II",
    2: "Data Science Methodology: An Analytic Approach to Capstone Project",
    3: "Making Machines See",
    4: "AI with Orange Data Mining Tool",
    5: "Introduction to Big Data and Data Analytics",
    6: "Understanding Neural Networks",
    7: "Generative AI",
    8: "Data Storytelling",
}

TARGET_WORDS = 100   # a passage is cut once it reaches about this many words
MIN_WORDS = 25       # tiny leftovers are merged into the previous passage

unit_line = re.compile(r"^\s*UNIT\s+(\d)\s*:")                      # "UNIT 6: Understanding..."
heading_line = re.compile(r"^(\d+\.\d+(?:\.\s?\d+)*\.?)\s+([A-Z].{2,90})$")  # "6.2. Types of ..."
activity_line = re.compile(r"^(Practical Activity|Activity)\b.*$")
sentence_split = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9•\-\(])")


def clean(line):
    return re.sub(r"\s+", " ", line).strip()


def split_into_chunks(text):
    """Cut one section's text into passages of roughly TARGET_WORDS words."""
    sentences = sentence_split.split(text)
    chunks, current = [], []
    for sentence in sentences:
        current.append(sentence)
        if sum(len(s.split()) for s in current) >= TARGET_WORDS:
            chunks.append(" ".join(current))
            current = []
    if current:
        leftover = " ".join(current)
        if chunks and len(leftover.split()) < MIN_WORDS:
            chunks[-1] += " " + leftover
        else:
            chunks.append(leftover)
    return chunks


def main():
    with open(INPUT_FILE, encoding="utf-8") as f:
        lines = f.read().splitlines()

    sections = []          # list of (unit, section_name, text)
    unit = None
    section = "Overview"
    buffer = []
    in_exercises = False

    def flush():
        text = clean(" ".join(buffer))
        if unit and text:
            sections.append((unit, section, text))
        buffer.clear()

    for raw in lines:
        line = clean(raw)

        m = unit_line.match(raw)
        if m:                                  # a new unit starts
            flush()
            unit = int(m.group(1))
            section = "Overview"
            in_exercises = False
            continue
        if unit is None:                       # skip the cover pages and index
            continue
        if line.upper() == "EXERCISES":        # exercises have questions, not answers
            flush()
            in_exercises = True
            continue
        if in_exercises:
            continue
        if not line or line.isdigit() or set(line) <= set("_- "):   # blank, page number, divider
            continue
        if line.lower().startswith("fig"):     # figure captions
            continue

        h = heading_line.match(line)
        if h:
            flush()
            section = f"{h.group(1).replace(' ', '').rstrip('.')} {h.group(2).strip()}"
            continue
        a = activity_line.match(line)
        if a and len(line) < 90:
            flush()
            section = line
            continue

        buffer.append(line)
    flush()

    chunks = []
    for unit_no, section_name, text in sections:
        for part in split_into_chunks(text):
            if len(part.split()) < 15:          # too tiny to be a useful answer
                continue
            chunks.append({
                "id": len(chunks),
                "unit": unit_no,
                "unit_title": UNIT_TITLES[unit_no],
                "section": section_name,
                "text": part,
            })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=1, ensure_ascii=False)

    print(f"Saved {len(chunks)} passages to {OUTPUT_FILE}")
    for u in sorted(UNIT_TITLES):
        n = sum(1 for c in chunks if c["unit"] == u)
        print(f"  Unit {u}: {n} passages  ({UNIT_TITLES[u]})")


main()
