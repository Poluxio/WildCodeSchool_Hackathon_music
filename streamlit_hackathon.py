# ---------------------------------------------------------------------
# les libraires
# ---------------------------------------------------------------------
import streamlit as st
import time

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import random

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

# ---------------------------------------------------------------------
# les fonctions
# ---------------------------------------------------------------------
# fonction chargement dataframe
@st.cache_data
def load_data():
    data = pd.read_csv("https://raw.githubusercontent.com/Poluxio/WildCodeSchool_Hackathon_music/main/data_music_processed.csv")

    return data

# fonction jeu
def main():
    st.subheader("Quelles musiques allons nous écouter ?")

    # Demande le nombre de joueurs
    nombre_joueurs = st.number_input("Combien êtes-vous à vouloir danser ce soir ?", min_value=1, step=1)

    # Liste pour stocker les noms des joueurs
    noms_joueurs = []

    # Saisie des noms des joueurs
    st.write("Quel est ton petit nom ?")
    for i in range(nombre_joueurs):
        nom_joueur = st.text_input(f"Joueur {i+1} :", key=f"joueur_{i}")
        noms_joueurs.append(nom_joueur)

    # Bouton pour afficher le classement aléatoire des joueurs
    if st.button("Qui choisira sa musique en premier ?"):
        if len(noms_joueurs) <= 0:
            st.warning("Veuillez entrer au moins un nom de joueur.")
            
        else:
            classement_aleatoire = random.sample(noms_joueurs, len(noms_joueurs))
            
            for i, joueur in enumerate(classement_aleatoire, start=1):
                st.write(f"{i}. {joueur}")
            countdown_text = st.empty()  # Créer un espace vide pour afficher le compte à rebours

            for i in range(10, -1, -1):
                countdown_text.text(f"Temps restant avant le choix des musiques: {i} secondes")
                time.sleep(1)  # Attendre 1 seconde avant la prochaine itération
            st.write("⚠️ Nous allons tester votre mémoire, as tu bien mémorisé ton ordre de passage ? ⚠️ ")
            st.write("⬇️ c'est parti, choisissez vos musiques dans l'ordre ⬇️")

      

# ---------------------------------------------------------------------
# programme
# ---------------------------------------------------------------------
data_load_state = st.text('Chargement des données...')

dummies_music = load_data()

data_load_state.text('Chargement des données... terminé!')
time.sleep(2)

data_load_state.empty()

# Liste  genre, artistes et musiques

liste_genre = ['pop', 'hip hop', 'rap', 'reggae', 'afro', 'dance / electro', 'soul / r&b', 'rock', 'country', 'indie', 'band', 'metal', 'jazz']

liste_artist = dummies_music['artist'].unique().tolist()
liste_music = dummies_music['title_artist'].unique().tolist()
musiques_preferees_utilisateurs = []

col1, col2, col3 = st.columns(3)
with col1 :
    st.write("")
with col2 :
    st.write("")
with col3 :
   st.write("")   

st.title("🍻 🎵 FiestaTunes 🎵 🍻")

if __name__ == "__main__":
    main()


# Initialiser la liste des musiques préférées des utilisateurs
if 'musiques_preferees_utilisateurs' not in st.session_state:
    st.session_state.musiques_preferees_utilisateurs = []

# Sélection musique
choix_music = st.selectbox(
'Quel est ta musique préférée ?',
(liste_music))

if st.button("Ajoute à notre liste d'inspirations"):
    # Ajouter le choix de musique de l'utilisateur à la liste
    st.session_state.musiques_preferees_utilisateurs.append(choix_music)
# Affichage de la liste des musiques préférées des utilisateurs
df_choix = pd.DataFrame(st.session_state.musiques_preferees_utilisateurs, columns= ['musiques choisies'])
df_choix.index += 1
st.write("Votre liste d'inspirations : ")
df_choix

if st.button('🎵 Envoie la musique 🎵 '):

    # Sélection des colonnes numériques comme caractéristiques X et la colonne 'title' comme cible y
    X = dummies_music.select_dtypes(include = 'number')
    y = dummies_music['title']

    # Standardisation des colonnes X
    X_scaled =  RobustScaler().fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled)
    X_scaled.columns = X.columns
    
    # Entraînement du modèle K-Nearest Neighbors avec 10 voisins
    model = KNeighborsClassifier(n_neighbors = 30).fit(X_scaled, y)

    # Préparation des recommandations en fonction des musiques préférées des utilisateurs
    df_titre_scaled = pd.DataFrame()
    n = 3
    for music in st.session_state.musiques_preferees_utilisateurs:
        # Récupérer le nom du film 
        # nom_titre = dummies_music.loc[(dummies_music['title_artist'] == music )][1][0]
        # st.write(nom_titre)
        df_titre = dummies_music.loc[(dummies_music['title_artist'] == music )]
        df_titre_scaled_temp = X_scaled.iloc[df_titre.index, :].apply(lambda x : x*30/n)
        df_titre_scaled = pd.concat([df_titre_scaled, df_titre_scaled_temp])
        n+=1
    
    # Calcul de la moyenne des caractéristiques des musiques préférées pour obtenir une seule entrée
    df_titre_scaled = pd.DataFrame(df_titre_scaled.mean()).T

    # Recherche des voisins les plus proches (recommandations) pour la musique préférée de l'utilisateur
    recommandation_titre = model.kneighbors(df_titre_scaled)[1][0]

    # Récupération des musiques recommandés en fonction des genres et des musiques préférées des utilisateurs
    
    reco_titre = dummies_music.iloc[recommandation_titre, [0, 1, 2, 14]]
    reco_titre.rename(columns={"top genre": "sous genre", "genre affiné": "genre"}, inplace =True)
    reco_titre
    # Concaténation des recommandations finales pour obtenir un total de 30 musiques recommandés
