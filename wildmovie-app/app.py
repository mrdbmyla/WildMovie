# =============================================================
# streamlit run "WCS/github/wildmovie-app-dev/app.py"
# =============================================================

# =============================================================
# libraries
# =============================================================

import streamlit as st
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
# réglages
# =============================================================

st.set_page_config(
    page_title="Wild Movies",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =============================================================
# les poids pour favoriser les 'features'
# =============================================================

ACTOR_WEIGHT = 2.5 # 1.0 = neutre / 2.5 = biais acteurs
GENRE_WEIGHT = 1.0 # pas besoin de modifier, 1.0 = neutre

# =============================================================
# charger le modèle & data
# =============================================================

# @st.cache_resource garde tout dans le cache
@st.cache_resource
def load_light_model():
    artifacts = joblib.load(model_files_folder / "nn_model_light.joblib")
    nn_model = artifacts["model"]
    id_by_index = artifacts["id_by_index"]
    title_by_index = artifacts["title_by_index"]
    title_to_id = {title.lower(): mid for title, mid in zip(title_by_index, id_by_index)}
    actor_cols = artifacts["actor_cols"]
    genre_cols = artifacts["genre_cols"]
    
    return nn_model, id_by_index, title_by_index, title_to_id, actor_cols, genre_cols

nn_model, id_by_index, title_by_index, title_to_id, actor_cols, genre_cols = load_light_model()

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
def find_movie_id_by_title(search_title):
    imdb_id = title_to_id.get(search_title.strip().lower())
    
    if imdb_id is None:
        st.error("Film non trouvé dans la base de données.")
    
    return imdb_id

# API: fonction de récupération de posters & infos d'imdbapi.dev
# --> à changer pour TMDb pour avoir les versions localisées
def fetch_movie_details_from_api(imdb_id):
    api_url = f"https://api.imdbapi.dev/titles/{imdb_id}"
    
    try:
        api_response = requests.get(api_url)
        response_data = api_response.json()
        
        movie_details = {
            "id": imdb_id,
            "title": response_data.get("primaryTitle", "Unknown Title"),
            "plot": response_data.get("plot", "No plot available."),
            "poster": response_data.get("primaryImage", {}).get("url"),
            "genres": response_data.get("genres", []),
            "rating": response_data.get("rating", {}).get("aggregateRating")
        }
        
        return movie_details
        
    except:
        st.error(f"Erreur lors de la récupération: {api_url}")
        return None


# =============================================================
# requêtes: recommendations
# =============================================================
from scipy.sparse import csr_matrix

# fonction pour recommendations
def get_movie_recommendations(imdb_id, number_of_recommendations=3, actor_cols=None, genre_cols=None):
    film_index = id_by_index.index(imdb_id)

    # copier le modèle pour version poids
    film_features = nn_model._fit_X[film_index].copy()

    # utiliser des poids
    film_features[:, actor_cols] *= ACTOR_WEIGHT
    film_features[:, genre_cols] *= GENRE_WEIGHT

    # 3) trouver les voisins les plus proches (+1 car le film lui-même est inclus)
    distances, indices = nn_model.kneighbors(
        film_features,
        n_neighbors=number_of_recommendations + 1
    )

    neighbor_ids = [id_by_index[i] for i in indices[0]]
    neighbor_ids.remove(imdb_id)

    return neighbor_ids

# fonction pour 3 films aléatoires à l'ouverture de page
def pick_random_recent_movies(number_of_movies=3):
    return random.sample(id_by_index, k=number_of_movies)


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

    /* logo */
    .fixed-logo {
    position: fixed;
    top: 100px;
    left: 60px;
    z-index: 9999;
    width: 120px;
    }

    .fixed-logo img { 
    width: 100%; 
    height: auto; 
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

    [data-testid="stSidebarNav"] {
        display: none !important;
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
    imdb_id = find_movie_id_by_title(user_search_input)
    if imdb_id:
        movie_data = fetch_movie_details_from_api(imdb_id)
        if movie_data:
            display_searched_movie(movie_data)
            st.markdown("---")
            st.markdown(f'<h3 class="title-reco">Similaires à {movie_data["title"]}:</h3>', unsafe_allow_html=True)
            reco_ids = get_movie_recommendations(imdb_id, number_of_recommendations=3, actor_cols=actor_cols, genre_cols=genre_cols)
            cols = st.columns(3, gap="medium")
            for i, rid in enumerate(reco_ids):
                with cols[i]:
                    reco_data = fetch_movie_details_from_api(rid)
                    display_movie_card(reco_data, show_full_card=True)
else:
    st.markdown('<h3 class="title-reco">Actuellement à l\'affiche:</h3>', unsafe_allow_html=True)
    promo_ids = pick_random_recent_movies()
    cols = st.columns(3, gap="medium")
    for i, pid in enumerate(promo_ids):
        with cols[i]:
            promo_data = fetch_movie_details_from_api(pid)
            display_movie_card(promo_data)


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
st.sidebar.code("# WCS PROJET 2:\n\nMoteur de recommandation pour films.")

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
# Tech utilisées:
                
Python
Pandas
Machine Learning
Streamlit
DataViz"""
st.sidebar.code(tech_stack)