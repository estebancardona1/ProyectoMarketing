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

# Guardar el dataframe final
joblib.dump(df_final3,"salidas\\df_final3.joblib")


####################################################################################
######################## 1. SISTEMAS BASADOS EN POPULARIDAD ########################
####################################################################################

###         TOP 10 (PELÍCULAS MEJORES CALIFICADAS - GLOBAL)
### --------------------------------------------------------------------------------

top_a = pd.read_sql("""SELECT title, 
                ROUND(AVG(movie_rating), 2) AS score,
                COUNT(*) as views
                FROM final_ratings
                GROUP by title
                HAVING views >= 20 
                ORDER by score desc
                limit 10
                """, conn)


###         TOP 10 (PELÍCULAS MÁS VISTAS - GLOBAL)
### --------------------------------------------------------------------------------

top_b = pd.read_sql("""SELECT title, 
                ROUND(AVG(movie_rating), 2) AS score,
                COUNT(*) as views
                FROM final_ratings
                GROUP by title
                HAVING score >= 4 
                ORDER BY views desc
                LIMIT 10
                """, conn)

###         TOP 10 (PELÍCULAS MÁS VISTAS EN EL ÚLTIMO AÑO)
### --------------------------------------------------------------------------------

top_c = pd.read_sql('''  
            SELECT title, 
            COUNT(fr.movie_id) AS views,
            (SELECT ROUND(AVG(movie_rating), 2) 
             FROM final_ratings 
             WHERE movie_id = fr.movie_id) AS score
            FROM final_ratings fr
            WHERE strftime('%Y', datetime(fr.timestamp, 'unixepoch')) = "2018"
            GROUP BY title
            HAVING score >= 4
            ORDER BY views DESC
            LIMIT 10''', conn)


###         TOP 10 (PELÍCULAS MEJORES CALIFICADAS EN EL ÚLTIMO AÑO)
### --------------------------------------------------------------------------------

top_d = pd.read_sql('''  
            SELECT title, 
            COUNT(fr.movie_id) AS views,
            (SELECT ROUND(AVG(movie_rating), 2) 
             FROM final_ratings 
             WHERE movie_id = fr.movie_id) AS score
            FROM final_ratings fr
            WHERE strftime('%Y', datetime(fr.timestamp, 'unixepoch')) = "2018"
            GROUP BY title
            HAVING views >= 10
            ORDER BY score DESC
            LIMIT 10''', conn)


###         TOP 10 (PELÍCULAS MÁS VISTAS POR GÉNERO)
### --------------------------------------------------------------------------------
import ipywidgets as widgets
from IPython.display import display
from prettytable import PrettyTable


# Obtener los géneros únicos
genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Drama', 
          'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery', 
          'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

# Crear un widget de lista desplegable
genre_dropdown = widgets.Dropdown(
    options=genres,
    description='Select Genre:',
    value='Action'  # Género por defecto
)

def show_top_movies(selected_genre):
    # Filtrar las películas según el género seleccionado
    genre_movies = df_final3[df_final3[selected_genre] == 1]

    # Agrupar por título y calcular el promedio de las calificaciones y el conteo de vistas
    top_movies = genre_movies.groupby('title').agg(
        score=('movie_rating', 'mean'),  # Promedio de ratings
        views=('movie_rating', 'count')   # Contar el número de calificaciones
    ).reset_index()

    # Renombrar la columna del promedio
    top_movies['score'] = top_movies['score'].round(2)  # Redondear a 2 decimales

    # Filtrar por score mayor o igual a 4
    top_movies = top_movies[top_movies['score'] >= 4]

    # Ordenar por la puntuación en orden descendente y obtener las 10 mejores
    top_movies = top_movies.sort_values(by='views', ascending=False).head(10)

    # Mostrar el resultado
    # Extraer el título de la película #1
    top_movie = top_movies.iloc[0]['title']

    fn.fetch_movie_poster(top_movie)

    print("La película número 1 es: ", top_movie)
    print("")
    print("Aquí un listado con las películas que te podrían interesar: ")
    print("")
    
    # Usar PrettyTable para alinear
    table = PrettyTable()
    table.field_names = ["Title", "Score", "Views"]
    for index, row in top_movies.iterrows():
        table.add_row([row['title'], row['score'], row['views']])

    # Alinear a la derecha
    table.align = "l"
    print(table)

    print("")
    

# Conectar la función al evento de cambio de selección
widgets.interactive(show_top_movies, selected_genre=genre_dropdown)


############################################################################################
####### 2.1 Sistema de recomendación basado en contenido un solo producto - Manual #########
############################################################################################

## ----- SISTEMA DE RECOMENDACIÓN BASADO EN CONTENIDO KNN

# Verificar la posición de las columnas dummy (géneros)
gen_dummies = df_final3.columns[5:]  
movies_dum = df_final3[gen_dummies]

# Datos para sistema de recomendación 3
gen_dummies2 = df_final3.columns[4:]  
movies_dum2 = df_final3[gen_dummies2]
sc=MinMaxScaler()
movies_dum2[["year"]]=sc.fit_transform(movies_dum2[['year']])

joblib.dump(movies_dum2,"salidas\\movies_dum2.joblib")

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





# Extraer el título de la película #1
top_movie = top_a.iloc[0]['title']

# Eliminar los últimos 6 caracteres
top_movie = top_movie[:-6]

poster = fn.fetch_movie_poster(top_movie)