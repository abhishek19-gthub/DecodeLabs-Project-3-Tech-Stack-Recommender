**DecodeLabs AI Internship - Project 3: AI Recommendation Logic**

A content-based recommendation system that suggests career paths (job roles) based on the skills a user enters. It uses **TF-IDF** to weight skills and **Cosine Similarity** to find the closest matching roles.

## How it works

The project follows the Input -> Process -> Output model:

1. **Input:** the user enters 3 skills (for example Python, Cloud Computing, Automation).
2. **Process:** job roles and the user's skills are converted into TF-IDF vectors in the same vocabulary space. Cosine similarity is then calculated between the user vector and every role vector.
3. **Output:** the roles are sorted by score and the **Top 3** are shown with a percentage match.

Rare skills (like Kubernetes) get more weight than common skills (like Python), so the matches are more meaningful than simple tag counting.

## Features

- Content-based filtering (no data about other users needed)
- TF-IDF weighting to reward specific skills
- Cosine similarity scoring
- Top-N ranking (default Top 3)
- Cold-start handling: if no entered skill exists in the dataset, the program shows the available skills instead of crashing
- Reads `raw_skills.csv` if present, otherwise uses a built-in dataset

## Tech used

- Python 3
- pandas
- scikit-learn

## How to run

```bash
pip install -r requirements.txt
python tech_stack_recommender.py
```

## Dataset format

`raw_skills.csv` should have the job role in the first column and the skills in the second column, separated by commas:

```
role,skills
Data Scientist,"python, sql, machine learning, data analysis, statistics"
DevOps Engineer,"aws, docker, kubernetes, ci/cd, git, automation, linux"
```

## Sample output

```
=== Tech Stack Recommender ===
Skill 1: Python
Skill 2: Cloud Computing
Skill 3: Automation

===== TOP RECOMMENDED CAREER PATHS =====
1. Cloud Architect  -  52.6% match
   Skills: aws, cloud computing, azure, networking, automation, security
2. System Administrator  -  21.9% match
   Skills: linux, networking, bash, automation, security
3. Data Scientist  -  17.8% match
   Skills: python, sql, machine learning, data analysis, statistics
=========================================
```

## Concepts covered

Feature extraction, vector mapping, TF-IDF, cosine similarity, Top-N ranking, cold-start problem.
