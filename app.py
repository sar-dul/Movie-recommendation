import streamlit as st
from recommendation_model import load_and_preprocess_data, collaborative_recommend, content_based_recommend, hybrid_recommendation, get_similar_title, display_recommendations_with_chart

# Load and preprocess data
df_movies, interaction_matrix = load_and_preprocess_data()

st.title('Movie Recommendation System')

# Sidebar for user inputs
st.sidebar.header('User Input')
movie_name = st.sidebar.text_input('Enter a Movie Title:', 'e.g. Spiderman')

# Select the type of recommendation
recommendation_type = st.sidebar.selectbox(
    'Select Recommendation Type:',
    ['Collaborative Filtering', 'Content-Based Filtering', 'Hybrid Filtering']
)

# Add a "Go" button
if st.sidebar.button('Get Recommendations'):
    with st.spinner("Generating..."):
        _, movie_title = get_similar_title(movie_name, df_movies)

    # Show recommendations based on user input
    if recommendation_type == 'Collaborative Filtering':
        st.write('Collaborative Filtering Recommendations:')
        recommendations = collaborative_recommend(movie_title, interaction_matrix, df_movies, 5)
    elif recommendation_type == 'Content-Based Filtering':
        st.write('Content-Based Filtering Recommendations:')
        recommendations = content_based_recommend(movie_title, df_movies, 5)
    else:
        st.write('Hybrid Filtering Recommendations:')
        recommendations = hybrid_recommendation(movie_title,interaction_matrix, df_movies, 5)

    display_recommendations_with_chart(recommendations)
