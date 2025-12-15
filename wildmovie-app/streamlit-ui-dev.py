# =============================================================
# streamlit run "WCS/github/wildmovies-app-dev/streamlit-ui-dev.py"
# =============================================================

# =============================================================
# libraries
# =============================================================

import streamlit as st
import pandas as pd
import requests
import base64
import random
from pathlib import Path
import joblib

# =============================================================
# chemin
# =============================================================

current_script_folder = Path(__file__).parent
images_folder = current_script_folder / "img"
model_files_folder = current_script_folder / "files"

# =============================================================
# charger le modèle & data
# =============================================================

# @st.cache_resource garde tout dans le cache
@st.cache_resource
def load_model_artifacts():
    movies_dataframe = pd.read_parquet(model_files_folder / "df_concat.parquet")
    features_csv = pd.read_csv(model_files_folder / "feature_columns.csv")
    feature_column_names = features_csv["feature"].tolist()
    data_scaler = joblib.load(model_files_folder / "scaler.joblib")
    nearest_neighbors_model = joblib.load(model_files_folder / "nn_model.joblib")
    
    return movies_dataframe, feature_column_names, data_scaler, nearest_neighbors_model

df_concat, feature_columns, scaler, nn_model = load_model_artifacts()

# =============================================================
# img --> base64 (img de fond & logo)
# =============================================================

def convert_image_to_base64(image_file_path):
    file_handle = open(image_file_path, "rb")
    image_bytes = file_handle.read()
    file_handle.close()
    
    base64_bytes = base64.b64encode(image_bytes)
    base64_string = base64_bytes.decode()
    
    return base64_string

def set_background_image(image_file_path, opacity=0.3):
    image_base64 = convert_image_to_base64(image_file_path)
    css_code = f"""
        <style>
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 59%;
            background-image: url("data:image/png;base64,{image_base64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            opacity: {opacity};
        }}
        </style>
    """
    st.markdown(css_code, unsafe_allow_html=True)

# =============================================================
# requêtes: recherche
# =============================================================

# fonction de recherche dans le modèle
def find_movie_id_by_title(search_title, movies_dataframe):
    matches = movies_dataframe[movies_dataframe["Titre"].str.lower() == search_title.strip().lower()]
    
    if len(matches) == 0:
        st.error("Film non trouvé dans la base de données.")
        return None
    
    return matches.iloc[0]["ID_film"]

# API: fonction de récupération de posters & infos d'imdbapi.dev
# --> à changer pour TMDb pour avoir les versions localisées
def fetch_movie_details_from_api(imdb_id):
     # API URL
    api_url = f"https://api.imdbapi.dev/titles/{imdb_id}"
    
    try:
        api_response = requests.get(api_url)

        if api_response.status_code != 200:
            st.error(f"""
                Erreur API (status {api_response.status_code}):
                URL: {api_url}
                Response: {api_response.text}
            """)
            return None

        response_data = api_response.json()
        
        # ce que nous voulons
        movie_title = response_data.get("primaryTitle", "Unknown Title")
        movie_plot = response_data.get("plot", "No plot available.")
        primary_image_data = response_data.get("primaryImage", {})
        poster_url = primary_image_data.get("url")
        movie_genres = response_data.get("genres", [])
        rating_data = response_data.get("rating", {})
        movie_rating = rating_data.get("aggregateRating")
        
        # dictionnaire avec le tout
        movie_details = {
            "id": imdb_id,
            "title": movie_title,
            "plot": movie_plot,
            "poster": poster_url,
            "genres": movie_genres,
            "rating": movie_rating
        }
        
        return movie_details
        
    except Exception as error:
        st.error(f"""
            Erreur lors de la récupération des données:
            URL: {api_url}
            Type d'erreur: {type(error).__name__}
            Message: {error}
        """)
        return None


# =============================================================
# requêtes: recommendations
# =============================================================

# fonction pour recommendations
def get_movie_recommendations(imdb_id, number_of_recommendations):
    # trouver le film
    movie_row_index = df_concat.index[df_concat["ID_film"] == imdb_id][0]
    
    # extraire et scaler les features
    movie_features = df_concat.loc[[movie_row_index], feature_columns]
    scaled_features = scaler.transform(movie_features)
    
    # trouver les voisins
    distances, indices = nn_model.kneighbors(scaled_features, n_neighbors=number_of_recommendations + 1)
    
    # convertir indices en IDs imdb
    neighbor_ids = df_concat.iloc[indices[0]]["ID_film"].tolist()
    
    # enlever le film original et retourner
    neighbor_ids.remove(imdb_id)
    return neighbor_ids

# fonction pour films aléatoires à l'ouverture de page
# critère: l'année
def pick_random_recent_movies(number_of_movies=3, minimum_year=2018):
    year_values = pd.to_numeric(df_concat["Année"], errors="coerce")
    recent_movies = df_concat[year_values >= minimum_year]
    unique_ids = recent_movies["ID_film"].dropna().drop_duplicates().tolist()
    
    return random.sample(unique_ids, k=number_of_movies)


# =============================================================
# résultats
# =============================================================

# film recherché
def display_searched_movie(movie_data):
    if movie_data is None:
        return
    poster_column, info_column = st.columns([1, 3], gap="medium")
    with poster_column:
        st.image(movie_data["poster"], width=200)
    with info_column:
        st.markdown(
            '<h3 class="title-search">Votre recherche:</h3>', 
            unsafe_allow_html=True
        )
        st.markdown(f"#### {movie_data['title']}")
        st.write(movie_data["plot"])
        genres_text = ", ".join(movie_data["genres"])
        st.write(f"**Genres:** {genres_text}")

# recommendations & aléatoires
def display_movie_card(movie_data, show_full_card=False):
    if movie_data is None:
        return
    st.image(movie_data["poster"], width=200)
    st.markdown(f"#### {movie_data['title']}")
    # afficher version entendue pour recommendations
    if show_full_card:
        st.write(movie_data["plot"])
        genres_text = ", ".join(movie_data["genres"])
        st.write(f"**Genres:** {genres_text}")
        rating_html = f'<span class="rating-badge">⭐ {movie_data["rating"]}</span>'
        st.markdown(rating_html, unsafe_allow_html=True)


# =============================================================
# css & images
# =============================================================

background_image_path = images_folder / "posters-bg.jpg"
set_background_image(background_image_path, opacity=0.05)

logo_path = images_folder / "logo-wildmovies.svg"
logo_base64 = convert_image_to_base64(logo_path)

custom_css = """
    <style>
    /* Define reusable color variables */
    :root {
        --font-accent-color: #D5BAFF;
        --button-accent-color: #5F3A9A;
        --button-accent-color-2: #7248B2;
        --white: #EAEAEA;
    }
    
    /* Style for horizontal lines */
    hr {
        margin-top: 20px !important;
        margin-bottom: 20px !important;      
    }
    
    /* Style for paragraphs */
    p {
        color: var(--white) !important;  
    }
    
    /* Height for certain input elements */
    .st-ba {
        height: 50px;
    }
    
    /* Style for the search button */
    .stFormSubmitButton>button {
        background-color: var(--button-accent-color) !important;
        color: white;
        border-radius: 10px;
        padding: 10px 24px;
        height: 46px;
        border: none;
        font-weight: 500;
    }
    
    /* Search button hover effect */
    .stFormSubmitButton>button:hover {
        background-color: var(--button-accent-color-2) !important;
    }
    
    /* Style for section titles */
    .title-search,
    .title-reco {
        color: var(--font-accent-color) !important;
        font-weight: 900 !important;
    }
    
    .title-search {
        padding-top: 0px !important;
        padding-bottom: 20px !important;
    }
    
    .title-reco {
        padding-bottom: 25px !important;
    }
    
    /* Style for rating badges */
    .rating-badge {
        background-color: rgba(213, 186, 255, 0.1) !important;
        color: white !important;
        padding: 6px 16px !important;
        border-radius: 6px !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        display: inline-block !important;
        margin-top: 8px !important;
    }
    
    /* Sidebar text wrapping */
    [data-testid="stSidebarContent"] pre {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
    }
    
    [data-testid="stSidebarContent"] code {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
    }
    
    /* Sidebar logo spacing */
    .sidebar-logo {
        margin-bottom: 70px !important;
    }
    </style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# =============================================================
# logo
# =============================================================

logo_html = f"""
<style>
.fixed-logo {{
    position: fixed;
    top: 100px;
    left: 60px;
    z-index: 9999;
    width: 120px;
}}
.fixed-logo img {{ 
    width: 100%; 
    height: auto; 
}}
</style>
<div class="fixed-logo">
    <img src="data:image/svg+xml;base64,{logo_base64}" alt="Logo">
</div>
"""
st.markdown(logo_html, unsafe_allow_html=True)


# =============================================================
# UI: recherche
# =============================================================

with st.form("search_form", border=False):
    # 80/20%
    search_input_column, button_column = st.columns(
        [4, 1], 
        vertical_alignment="center"
    )
    # barre de recherche
    with search_input_column:
        user_search_input = st.text_input(
            label="Recherchez le titre d'un film",
            placeholder="🔍 Recherchez le titre d'un film que vous avez aimé pour commencer...",
            # label caché
            label_visibility="collapsed"
        )
    # bouton
    with button_column:
        search_button_clicked = st.form_submit_button("Rechercher")

st.markdown("---")


# =============================================================
# UI: films
# =============================================================

user_performed_search = search_button_clicked and user_search_input

if user_performed_search:
    
    found_imdb_id = find_movie_id_by_title(user_search_input, df_concat)
    if found_imdb_id is None:
        st.stop()
    
    searched_movie_data = fetch_movie_details_from_api(found_imdb_id)
    if searched_movie_data is None:
        st.stop()
    
    # film recherché
    display_searched_movie(searched_movie_data)
    st.markdown("---")
    
    st.markdown(
        f'<h3 class="title-reco">Similaires à {searched_movie_data["title"]}:</h3>',
        unsafe_allow_html=True
    )
    
    recommended_movie_ids = get_movie_recommendations(
        imdb_id=found_imdb_id,
        number_of_recommendations=3
    )
    
    recommendation_columns = st.columns(3, gap="medium")
    
    for i in range(len(recommended_movie_ids)):
        with recommendation_columns[i]:
            recommendation_data = fetch_movie_details_from_api(recommended_movie_ids[i])
            display_movie_card(recommendation_data, show_full_card=True)

else:
    # pas de recherche, affiche "Actuellement en salles"
    promo_movie_ids = pick_random_recent_movies()
    st.markdown(
        '<h3 class="title-reco">Actuellement à l\'affiche:</h3>', 
        unsafe_allow_html=True
    )
    promo_columns = st.columns(3, gap="medium")
    
    for i in range(len(promo_movie_ids)):
        with promo_columns[i]:
            promo_movie_data = fetch_movie_details_from_api(promo_movie_ids[i])
            display_movie_card(promo_movie_data)


# =============================================================
# UI: panel de gauche
# =============================================================

# logo
sidebar_logo_html = f"""
<div class="sidebar-logo" style="display:flex; justify-content:center; margin: 0.5rem 0 1rem 0;">
    <img src="data:image/svg+xml;base64,{logo_base64}" style="width:120px; height:auto;" alt="Logo">
</div>
"""
st.sidebar.markdown(sidebar_logo_html, unsafe_allow_html=True)

# info projet
st.sidebar.code("# WCS PROJET 2:\n\nMoteur de recommendation pour films.")

# l'équipe
team_info = """
# L'équipe:
                
Mourad
Romain
Priscilla
Sebastian"""
st.sidebar.code(team_info)

# tech utilisée
tech_stack = """
# Tech utilisée:
                
Python
Pandas
Machine Learning
Streamlit"""
st.sidebar.code(tech_stack)