###         LIBRERIAS
### --------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import sqlite3 as sql
from sklearn.preprocessing import MinMaxScaler
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

fn.fetch_movie_poster(top_movie)
top_score


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




















## MAS ADELANTE ------------------------------ KNN

df_final.drop('movieId', axis=1, inplace=True)
df_final.drop('movieId:1', axis=1, inplace=True)
df_final.drop('cnt_rat', axis=1, inplace=True)

df_final['year'] = df_final['title'].str.extract(r'\((\d{4})\)')  # Extrae el año
df_final['title'] = df_final['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True)  # Elimina el año del título

## ----- SEPARAR LOS GÉNEROS EN COLUMNAS

# Separar los géneros en columnas teniendo en cuenta el criterio de separación '|'
genres_dummies = df_final['genres'].str.get_dummies(sep='|')

# Concatenar las columnas de géneros con el DataFrame original
df_final2 = pd.concat([df_final, genres_dummies], axis=1)

# Eliminar la columna de genres
df_final2.drop('genres', axis=1, inplace=True)
df_final2.drop('timestamp', axis=1, inplace=True)



# Cargar el dataframe preprocesado
df_final2 = pd.read_csv('data/df_movies_processed.csv')


#### Sistema de recomendación basado en contenido KNN un solo producto visto ########

gen_dummies = df_final2.columns[5:]  
movies_dum = df_final2[gen_dummies]

# Entrenar el modelo KNN
model = NearestNeighbors(n_neighbors=11, metric='euclidean')
model.fit(movies_dum)

# Para la recomendaciòn
def MovieRecommender(movie_name):
    movie_list_name = []
    
    # Extraer el índice de la película seleccionada
    movie_id = df_final2[df_final2['title'] == movie_name].index
    
    if len(movie_id) == 0:
        return f"No se encontró la película: {movie_name}"
    
    movie_id = movie_id[0]
    
    # Obtener las distancias y los índices de las películas más cercanas
    distances, idlist = model.kneighbors(movies_dum.iloc[movie_id].values.reshape(1, -1))

    # Para cada recomendación, agregar la película
    for newid in idlist[0]:  
        recommended_movie = df_final2.loc[newid, 'title']
        if recommended_movie != movie_name:  # Evitar agregar la misma película
            movie_list_name.append(recommended_movie)

    # Eliminar duplicados
    movie_list_name = list(set(movie_list_name))

    return movie_list_name

movie_titles = df_final2['title'].tolist()

print(interact(MovieRecommender, movie_name=movie_titles))



