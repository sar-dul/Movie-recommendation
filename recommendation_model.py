import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
import streamlit as st

# Load and preprocess data
@st.cache_data
def load_and_preprocess_data():
    df_movies = pd.read_csv('big_movies.csv')
    df_ratings = pd.read_csv('big_ratings.csv')
    
    df_movies.rename(columns={'movieId': 'movie_id'}, inplace=True)
    df_ratings.rename(columns={'movieId': 'movie_id', 'userId': 'User_ID'}, inplace=True)

    df_movies = df_movies.drop_duplicates()
    df_ratings = df_ratings.drop_duplicates()

    dropped_movie_ids = df_movies[df_movies['genres'].isna()]['movie_id'].tolist()
    df_ratings = df_ratings[~df_ratings['movie_id'].isin(dropped_movie_ids)]

    df_movies = df_movies.loc[~df_movies['keywords'].isnull()]
    df_movies = df_movies.loc[~df_movies['cast'].isnull()]
    df_movies = df_movies.loc[~df_movies['genres'].isnull()]

    df_movies['production_countries'].fillna(df_movies['production_countries'].mode()[0], inplace=True)
    df_movies = df_movies.reset_index(drop=True)

    unique_ids = df_ratings['movie_id'].unique()
    df_movies = df_movies[df_movies['movie_id'].isin(unique_ids)]
    df_ratings = df_ratings[df_ratings['movie_id'].isin(df_movies['movie_id'])]

    df_ratings = pd.merge(df_ratings, df_movies[['title', 'movie_id']], on='movie_id')

    scaler = StandardScaler()
    df_ratings['normalized_ratings'] = scaler.fit_transform(df_ratings[['rating']])
    interaction_matrix = df_ratings.pivot_table(index='User_ID', columns='movie_id', values='normalized_ratings').fillna(0)

    df_movies['tags'] = df_movies['genres'] + ' ' + df_movies['keywords'] + ' ' + df_movies['overview'] + ' ' + df_movies['production_countries'] + ' ' + df_movies['cast']
    df_movies['tags'] = df_movies['tags'].apply(lambda x: x.replace(".","").replace(",","").replace(" &","")).str.lower()
    df_movies = df_movies[['title', 'tags']]

    return df_movies, interaction_matrix


def collaborative_recommend(movie_name, matrix, df, n_recs):
    # Fit the SVD model on the transposed interaction matrix
    svd_model = TruncatedSVD(n_components=50)
    svd_matrix = svd_model.fit_transform(matrix.T)

    # Extract input movie ID
    movie_id = df[df['title'] == movie_name].index[0]

    # Check if the movie_id is valid
    if movie_id >= len(svd_matrix):
        raise ValueError(f"Movie ID {movie_id} is out of range for the SVD matrix.")

    # Calculate similarity scores
    movie_vector = svd_matrix[movie_id]
    similarities = svd_matrix.dot(movie_vector)

    # Get the indices of the most similar movies
    similar_movie_ids = similarities.argsort()[::-1][1:n_recs+1]

    # List to store recommendations
    recs = []
    for i in similar_movie_ids:
        recs.append({'Title': df['title'][i], 'Similarity': similarities[i]})

    return recs

def content_based_recommend(title, df, n_recs):
    # Extract input movie ID
    movie_id = df[df['title'] == title].index[0]

    tfidf = TfidfVectorizer(max_features=8000, stop_words='english')
    vectors = tfidf.fit_transform(df['tags']).toarray()
    similarity = cosine_similarity(vectors)

    # Calculating cosine similarity with other movies
    distances = similarity[movie_id]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:n_recs+1]

    # Construct the list of recommended movie titles
    recs = [{'Title': df.iloc[i[0]].title, 'Similarity': i[1]} for i in movies_list]

    return recs

def get_similar_title(title, df):
    # Extracting the index of similar title to the input title
    movie_id = process.extractOne(title, df['title'])[2]

    movie_title = df.loc[movie_id]['title']

    return movie_id, movie_title


# Functions to scale the similarity scores after recommendation
def min_max_scaling(recommendations):
    min_score = min([rec['Similarity'] for rec in recommendations])
    max_score = max([rec['Similarity'] for rec in recommendations])
    scaled_recommendations = []

    for rec in recommendations:
        scaled_score = (rec['Similarity'] - min_score) / (max_score - min_score)
        scaled_recommendations.append({'Title': rec['Title'], 'Similarity': scaled_score})

    return scaled_recommendations

def scale_to_range(recommendations, new_min=0, new_max=1):
    min_score = min([rec['Similarity'] for rec in recommendations])
    max_score = max([rec['Similarity'] for rec in recommendations])
    scaled_recommendations = []

    for rec in recommendations:
        scaled_score = ((rec['Similarity'] - min_score) / (max_score - min_score)) * (new_max - new_min) + new_min
        scaled_recommendations.append({'Title': rec['Title'], 'Similarity': scaled_score})

    return scaled_recommendations



def hybrid_recommendation(title,interaction_matrix, df_movies, n_recs, weight_collab=0.4, weight_content=0.6):

    movie_id , movie_title = get_similar_title(title, df_movies)
    content_based_recommendation = content_based_recommend(movie_title,df_movies, n_recs)
    collaborative_recommendation = collaborative_recommend(movie_title, interaction_matrix, df_movies, n_recs)
    
    # Scaling the recommendations similarity scores
    scaled_collab_recommendations = scale_to_range(collaborative_recommendation, new_min=0, new_max=1)
    scaled_content_recommendations = min_max_scaling(content_based_recommendation)
    
    # Combine the recommendations
    hybrid_scores = {}
    
    for collab_rec in scaled_collab_recommendations:
        title = collab_rec['Title']
        score = collab_rec['Similarity']
        hybrid_scores[title] = weight_collab * score
    
    for content_rec in scaled_content_recommendations:
        title = content_rec['Title']
        score = content_rec['Similarity']
        if title in hybrid_scores:
            hybrid_scores[title] += weight_content * score
        else:
            hybrid_scores[title] = weight_content * score
    
    # Sort the hybrid scores to get the top recommendations
    sorted_hybrid_scores = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:n_recs]
    
    # Convert to list of dictionaries for the final recommendation
    final_recommendations = [{'Title': title, 'Similarity': score} for title, score in sorted_hybrid_scores]
    
    return final_recommendations


def display_recommendations_with_chart(recommendations):
    # Convert recommendations to DataFrame for easier plotting
    df_recommendations = pd.DataFrame(recommendations)
    
    # Bar Chart for Similarity Scores
    st.write("### Movie Recommendations")
    fig, ax = plt.subplots()
    sns.barplot(x='Similarity', y='Title', data=df_recommendations, ax=ax, palette="viridis")
    ax.set_xticklabels([])
    ax.set_xlabel("Similarity")
    ax.set_ylabel("Movie Title")
    st.pyplot(fig)

    # Display recommendations in text format as well
    for recommendation in recommendations:
        st.write(f"**Title**: {recommendation['Title']}")
        st.write(f"**Similarity**: {recommendation['Similarity']:.2f}")
        st.write("---")