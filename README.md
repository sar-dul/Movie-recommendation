# Movie Recommendation System

This project is a **Movie Recommendation System** that suggests movies based on collaborative filtering and content-based filtering. The system is implemented using Python and Streamlit, and it provides visualizations to enhance user experience.

## Features

- **Collaborative Filtering**: Recommends movies based on user interaction data.
- **Content-Based Filtering**: Recommends movies based on cosine similarity of movie tags.
- **Hybrid Recommendation**: Combines collaborative and content-based filtering to provide better recommendations.
- **Visualization**: Interactive charts and graphs to visualize recommendations using Streamlit.

## Dataset

The dataset used in this project can be found [here](https://drive.google.com/drive/folders/1WsUhSHgrMBzjfrLeQDKrf_Br0CXLQXam?usp=sharing).

## Project Structure

- **app.py**: The main application script for the Streamlit app.
- **recommendation_model.py**: Contains the logic for generating recommendations.
- **requirements.txt**: A list of Python dependencies required to run the project.

## Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/yourusername/yourrepository.git
    cd yourrepository
    ```

2. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Download the dataset**:

   Download the dataset from the [link](https://drive.google.com/drive/folders/1WsUhSHgrMBzjfrLeQDKrf_Br0CXLQXam?usp=sharing)

4. **Run the Streamlit app**:

    ```bash
    streamlit run app.py
    ```

## Filtering Methods

### Content-Based Filtering

This method recommends movies based on the similarity between movie tags and the movie you liked. The cosine similarity of movie tags is used to find similar movies.

![Content-Based](https://github.com/user-attachments/assets/7951252f-4f4c-4803-b44e-df21821df6cb)

### Collaborative Filtering

This method recommends movies based on the interaction data of other users who have rated or interacted with similar movies.

![Collaborative](https://github.com/user-attachments/assets/7dcc23cb-adf0-4566-bd24-bc9660971f6e)

### Hybrid Recommendation

This method combines both collaborative and content-based filtering to provide better recommendations by considering both movie features and user interactions.

![Hybrid](https://github.com/user-attachments/assets/0ac1353e-ef68-4da1-a92c-255b5274cb65)

## Usage

Once the app is running, you can:
- **Search for a movie** and get recommendations based on collaborative filtering or content-based filtering.
- **View visualizations** of the similarity scores and recommended movies.
- **Explore the hybrid recommendation system** that combines both filtering methods.
