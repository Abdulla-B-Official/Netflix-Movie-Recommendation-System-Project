# Netflix Movie & Show Recommendation System

[![Live Demo](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://netflix-movie-recommendation-system.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-TF--IDF%20%26%20Cosine-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Abdulla-B-Official/Netflix-Movie-Recommendation-System-Project)

<p align="center">
  <b>An end-to-end content-based machine learning recommendation engine designed to discover similar movies and TV shows using custom metadata feature engineering, TF-IDF vectorization, and Cosine Similarity.</b>
</p>

---

## Live Application

 **[netflix-movie-recommendation-system.streamlit.app]([https://netflix-movie-recommendation-system.streamlit.app](https://netflix-movie-recommendation-system-project-emydybmb2quxvuo85v.streamlit.app/))**

<p align="center">
  <img src="https://img.shields.io/badge/Algorithm-TF--IDF_Vectorization-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Metric-Cosine_Similarity-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Dataset-10%2C000%2B_Titles-red?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Deployment-Streamlit_Cloud-brightgreen?style=for-the-badge" />
</p>

---

## Overview

**Netflix Recommendation System** is a content-based recommendation system designed to analyze textual metadata across thousands of titles and generate similarity-scored media suggestions in real time.

Instead of relying solely on popularity or simple genre tags, the system builds an enriched feature profile ("metadata soup") incorporating:

* 🎬 **Main Genre** (Primary category)
* 🏷️ **Sub Genres** (Granular sub-themes)
* 🔞 **Maturity Rating** (Audience classification)
* 🗣️ **Original Audio** (Language preference)

The system transforms raw text metadata into high-dimensional numerical vectors using **TF-IDF Vectorization** and computes item-to-item similarity scores via **Cosine Similarity**.

Integrated into an interactive **Streamlit web application**, the platform allows users to search by title or `N_id` (Netflix ID), dynamically adjust top-$N$ recommendation outputs, and inspect exact similarity score calculations transparently.

---

### Application Features

* **Interactive Search & Lookup**: Filter titles easily via title selection or unique Netflix Content ID (`N_id`).
* **Dynamic Top-N Recommendations**: Adjust recommendations interactively from 1 to 10 items.
* **Transparent Similarity Scoring**: View exact scalar similarity weights ($0.0$ to $1.0$) for every recommended item.
* **Metadata Granularity**: Combines main genre, sub-genres, maturity rating, and language into a unified metadata matrix.
* **Ground-Truth Validation**: Benchmarks algorithmic cosine similarity against manual validation targets (`Recommendations` column).
* **Live Cloud Deployment**: Fully hosted and operational on Streamlit Community Cloud.

---

## Project Objective

The primary objective is to build a scalable, interactive content recommendation system that can:

* Clean and structure high-dimensional textual media attributes from raw CSV datasets.
* Handle index integrity by using unique primary keys (`N_id`) alongside display titles.
* Transform categorical and textual metadata into sparse numerical feature matrices using `TfidfVectorizer`.
* Compute $N \times N$ pairwise Cosine Similarity matrices for fast, continuous lookup.
* Deliver real-time recommendations through a clean, responsive web interface.
* Deploy an end-to-end Machine Learning pipeline accessible online via GitHub and Streamlit Cloud.

---

## Problem Statement

Navigating vast streaming content libraries can lead to decision fatigue for users looking for media tailored to specific sub-genres, language preferences, and age ratings. 

Standard filter systems often fail to:

* Measure non-linear similarity across multiple combined metadata fields simultaneously.
* Provide scoring transparency explaining *why* a particular movie was suggested.
* Handle fast, real-time matrix lookups on cloud-hosted web interfaces.

### Proposed Solution

This project introduces a content-based ML recommendation pipeline that executes:

$$\text{Raw Metadata} \longrightarrow \text{Soup Construction} \longrightarrow \text{TF-IDF Matrix} \longrightarrow \text{Cosine Similarity} \longrightarrow \text{Streamlit UI Output}$$

For every item selected, the system produces:

```text
Target Content Title & ID
Top-N Recommended Media Titles
Matched Main & Sub-Genres
Maturity & Audio Attributes
Exact Pairwise Similarity Scores
