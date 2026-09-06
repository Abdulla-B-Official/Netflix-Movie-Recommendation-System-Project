import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(page_title="Netflix Content Recommender", layout="wide")

st.title("🎬 Netflix Movie & Show Recommendation Engine")
st.write("Find similar movies and TV shows using TF-IDF and Cosine Similarity.")

# 1. Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("Netflix Data new.csv")
    features = ['N_id', 'Title', 'Main Genre', 'Sub Genres', 'Maturity Rating', 'Original Audio', 'Recommendations']
    df = df[features].fillna('')
    df['N_id'] = df['N_id'].astype(str)
    
    # Feature Engineering
    def create_metadata_soup(row):
        return f"{row['Main Genre']} {row['Sub Genres']} {row['Maturity Rating']} {row['Original Audio']}".lower()

    df['metadata'] = df.apply(create_metadata_soup, axis=1)
    return df

df = load_data()

# 2. Vectorization & Similarity Matrix Computation
@st.cache_resource
def build_engine(data):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['metadata'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = build_engine(df)

# Mappings
title_list = df['Title'].tolist()
title_to_idx = pd.Series(df.index, index=df['Title'].str.lower()).drop_duplicates()

# 3. Sidebar Inputs
st.sidebar.header("Recommendation Settings")
selected_title = st.sidebar.selectbox("Select a Movie/Show:", title_list)
top_n = st.sidebar.slider("Number of Recommendations:", min_value=1, max_value=10, value=5)

# 4. Recommendation Logic
def get_recommendations(title, top_n_count):
    idx = title_to_idx[title.lower()]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n_count+1]
    
    rec_indices = [i[0] for i in sim_scores]
    sim_values = [round(i[1], 4) for i in sim_scores]
    
    results = df[['N_id', 'Title', 'Main Genre', 'Sub Genres', 'Maturity Rating', 'Original Audio']].iloc[rec_indices].copy()
    results['Similarity Score'] = sim_values
    return results, df.iloc[idx]

# 5. UI Layout & Render Output
if st.sidebar.button("Get Recommendations"):
    results_df, target_info = get_recommendations(selected_title, top_n)
    
    st.subheader(f"Selected Item: {target_info['Title']}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Main Genre", target_info['Main Genre'])
    col2.metric("Maturity Rating", target_info['Maturity Rating'])
    col3.metric("Original Audio", target_info['Original Audio'])
    
    st.markdown("---")
    st.subheader(f"Top {top_n} Similar Recommendations")
    
    # Display styled interactive table
    st.dataframe(
        results_df[['Title', 'Main Genre', 'Sub Genres', 'Maturity Rating', 'Original Audio', 'Similarity Score']],
        use_container_width=True
    )