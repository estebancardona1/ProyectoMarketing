###         LIBRERIAS
### --------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import sqlite3 as sql
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import interact ## para análisis interactivo
from sklearn import neighbors ### basado en contenido un solo producto consumido
import joblib
import a_funciones as fn
from sklearn.neighbors import NearestNeighbors

###         CARGAR DATOS
### --------------------------------------------------------------------------------

conn=sql.connect('data\\db_movies') 
cur=conn.cursor() 

#### ver tablas disponibles en base de datos

cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
cur.fetchall()

df_final = pd.read_sql('select * from  final_ratings',conn)

df_final2 = pd.read_sql('''SELECT user_id, movie_id, title,
                        movie_rating, genres,
                        datetime(timestamp, 'unixepoch') AS rating_date
                        FROM final_ratings;''', conn)


####################################################################################
######################## 1. SISTEMAS BASADOS EN POPULARIDAD ########################
####################################################################################


###         TOP 10 (PELÍCULAS MEJORES CALIFICADAS)
### --------------------------------------------------------------------------------

top_score = pd.read_sql("""SELECT title, 
                ROUND(AVG(movie_rating), 2) AS score,
                COUNT(*) as views
                FROM final_ratings
                GROUP by title
                HAVING views >= 20 
                ORDER by score desc
                limit 10
                """, conn)

# Extraer el título de la película #1
top_movie = top_score.iloc[0]['title']

# Eliminar los últimos 6 caracteres
top_movie = top_movie[:-6]

poster = fn.fetch_movie_poster(top_movie)


###         TOP 10 (PELÍCULAS MÁS VISTAS)
### --------------------------------------------------------------------------------

top_score = pd.read_sql("""SELECT title, 
                ROUND(AVG(movie_rating), 2) AS score,
                COUNT(*) as views
                FROM final_ratings
                GROUP by title
                HAVING score >= 4 
                ORDER BY views desc
                LIMIT 10
                """, conn)

# Extraer el título de la película #1
top_movie = top_score.iloc[0]['title']

# Eliminar los últimos 6 caracteres
top_movie = top_movie[:-6]

fn.fetch_movie_poster(top_movie)
top_score

###         TOP 10 (PELÍCULAS MÁS VISTAS EN EL ÚLTIMO AÑO)
### --------------------------------------------------------------------------------



top_score = pd.read_sql('''
            SELECT title, 
            ROUND(AVG(movie_rating), 2) AS score,
            COUNT(movie_id) AS views
            FROM final_ratings
            WHERE strftime('%Y', datetime(timestamp, 'unixepoch')) = '2018'
            GROUP BY title
            HAVING score >= 4
            ORDER BY views DESC
            LIMIT 10;
''', conn)

# Extraer el título de la película #1
top_movie = top_score.iloc[0]['title']

# Eliminar los últimos 6 caracteres
top_movie = top_movie[:-6]

fn.fetch_movie_poster(top_movie)
top_score


###         TOP 10 (PELÍCULAS MÁS VISTAS EN EL ÚLTIMO MES)
### --------------------------------------------------------------------------------

top_score = pd.read_sql('''
            SELECT title, 
            ROUND(AVG(movie_rating), 2) AS score,
            COUNT(movie_id) AS views
            FROM final_ratings
            WHERE strftime('%Y-%M', datetime(timestamp, 'unixepoch')) = '2018-08'
            GROUP BY title
            HAVING score >= 4
            ORDER BY views DESC
            LIMIT 10;
''', conn)

# Extraer el título de la película #1
top_movie = top_score.iloc[0]['title']

# Eliminar los últimos 6 caracteres
top_movie = top_movie[:-6]

fn.fetch_movie_poster(top_movie)
top_score


############################################################################################
####### 2.1 Sistema de recomendación basado en contenido un solo producto - Manual #########
############################################################################################

## ----- SEPARAR EL AÑO EN UNA NUEVA VARIABLE

df_final2['year'] = df_final2['title'].str.extract(r'\((\d{4})\)')  # Extrae el año
df_final2['title'] = df_final2['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True)  # Elimina el año del título

df_final2.drop('rating_date', axis=1, inplace=True)

## ----- SEPARAR LOS GÉNEROS EN COLUMNAS

#--------- Separar los géneros en columnas teniendo en cuenta el criterio de separación '|'
genres_dummies = df_final2['genres'].str.get_dummies(sep='|')

# Concatenar las columnas de géneros con el DataFrame original
df_final3 = pd.concat([df_final2, genres_dummies], axis=1)

# Eliminar las columnas de 'genres' y 'timestamp'
df_final3.drop(['genres'], axis=1, inplace=True)

## ----- SISTEMA DE RECOMENDACIÓN BASADO EN CONTENIDO KNN

# Verificar la posición de las columnas dummy (géneros)
gen_dummies = df_final3.columns[5:]  
movies_dum = df_final3[gen_dummies]

# Entrenar el modelo KNN
model = NearestNeighbors(n_neighbors=11, metric='euclidean')
model.fit(movies_dum)

def MovieRecommender(movie_name):
    movie_list_name = []
    
    # Extraer el índice de la película seleccionada
    movie_id = df_final2[df_final3['title'] == movie_name].index
    
    # Verificar si se encontró la película
    if len(movie_id) == 0:
        return f"No se encontró la película: {movie_name}"
    
    movie_id = movie_id[0]  

    # Obtener las distancias y los índices de las películas más cercanas
    distances, idlist = model.kneighbors(movies_dum.iloc[movie_id].values.reshape(1, -1))

    # Para cada recomendación, agregar la película si no es la misma seleccionada
    for newid in idlist[0]:
        recommended_movie = df_final3.loc[newid, 'title']
        if recommended_movie != movie_name:  # Para evitar agregar la misma película
            movie_list_name.append(recommended_movie)

    # Eliminar duplicados de las recomendaciones
    movie_list_name = list(set(movie_list_name))

    return movie_list_name

movie_titles = sorted(df_final3['title'].unique())

# Mostrar el sistema de recomendación interactivo
print(interact(MovieRecommender, movie_name=movie_titles))





