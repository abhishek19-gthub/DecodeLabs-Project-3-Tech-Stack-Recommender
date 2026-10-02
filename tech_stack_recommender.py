"""
DecodeLabs - Project 3: AI Recommendation Logic
Tech Stack Recommender (Content-Based Filtering)

Pipeline:  Input (3 skills) -> Process (TF-IDF + Cosine Similarity) -> Output (Top 3 roles)

Run:  python tech_stack_recommender.py
Needs: pip install pandas scikit-learn
"""

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CSV_FILE = "raw_skills.csv"   # first column = job role, second column = skills
TOP_N = 3

# Used only if raw_skills.csv is not found next to this script
DEFAULT_DATA = {
    "role": [
        "Data Scientist",
        "Machine Learning Engineer",
        "Data Analyst",
        "DevOps Engineer",
        "Cloud Architect",
        "System Administrator",
        "Backend Developer",
        "Frontend Developer",
        "Full Stack Developer",
        "Cyber Security Analyst",
    ],
    "skills": [
        "python, sql, machine learning, data analysis, statistics",
        "python, machine learning, deep learning, tensorflow, docker",
        "sql, excel, data analysis, power bi, statistics",
        "aws, docker, kubernetes, ci/cd, git, automation, linux",
        "aws, cloud computing, azure, networking, automation, security",
        "linux, networking, bash, automation, security",
        "java, python, sql, apis, git, data structures",
        "html, css, javascript, react, git",
        "javascript, react, node.js, sql, apis, git",
        "networking, security, linux, python, cryptography",
    ],
}


# ---------- Step 0: load the dataset ----------
def load_data():
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE).iloc[:, :2]
        df.columns = ["role", "skills"]
        print(f"Loaded dataset from {CSV_FILE}")
    else:
        df = pd.DataFrame(DEFAULT_DATA)
        print(f"{CSV_FILE} not found - using built-in dataset")

    # clean: lowercase, and make ; and | behave like commas
    df["skills"] = (
        df["skills"].astype(str).str.lower()
        .str.replace(";", ",", regex=False).str.replace("|", ",", regex=False)
    )
    return df


# ---------- Step 1: Ingestion (take 3 skills from the user) ----------
def get_user_skills(n=3):
    print(f"\nEnter {n} skills (example: Python, Cloud Computing, Automation)")
    skills = []
    while len(skills) < n:
        skill = input(f"Skill {len(skills) + 1}: ").strip().lower()
        if skill == "":
            print("  Skill cannot be empty, try again.")
        elif skill in skills:
            print("  You already entered that skill, try another.")
        else:
            skills.append(skill)
    return skills


# split on commas so multi-word skills like "machine learning" stay as ONE feature
def skill_tokenizer(text):
    return [s.strip() for s in text.split(",") if s.strip()]


# ---------- Step 2: Scoring (TF-IDF + cosine similarity) ----------
def score_roles(df, user_skills):
    vectorizer = TfidfVectorizer(tokenizer=skill_tokenizer, lowercase=True, token_pattern=None)
    role_vectors = vectorizer.fit_transform(df["skills"])        # job role vectors

    user_text = ", ".join(user_skills)
    user_vector = vectorizer.transform([user_text])              # same vocabulary as roles

    # cold-start check: none of the user's skills exist in the dataset
    if user_vector.nnz == 0:
        return None, vectorizer

    scores = cosine_similarity(user_vector, role_vectors).flatten()
    df = df.copy()
    df["score"] = scores
    return df, vectorizer


# ---------- Steps 3 & 4: Sorting and Filtering ----------
def get_top_n(scored_df, n=TOP_N):
    ranked = scored_df.sort_values("score", ascending=False)     # Step 3: sort
    return ranked.head(n)                                        # Step 4: Top-N


def show_results(top_df):
    print("\n===== TOP RECOMMENDED CAREER PATHS =====")
    for rank, (_, row) in enumerate(top_df.iterrows(), start=1):
        print(f"{rank}. {row['role']}  -  {row['score'] * 100:.1f}% match")
        print(f"   Skills: {row['skills']}")
    print("=========================================")


def main():
    print("=== Tech Stack Recommender ===")
    df = load_data()
    user_skills = get_user_skills(3)

    scored_df, vectorizer = score_roles(df, user_skills)

    if scored_df is None:
        print("\nSorry, none of your skills were found in our dataset.")
        print("Try skills from this list:")
        print(", ".join(sorted(vectorizer.get_feature_names_out())))
        return

    show_results(get_top_n(scored_df))


if __name__ == "__main__":
    main()