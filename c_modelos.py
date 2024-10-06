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
import ipywidgets as widgets
from IPython.display import display
import re

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

## ----- CORREGIR LOS TÍTULOS QUE ESTÁN MAL ESCRITOS
def fix_titles(title):
    # Usamos una expresión regular para identificar títulos como "Anaconda, The" o "Matrix, The"
    pattern = r'(.+),\s(The|A|An)$'
    match = re.match(pattern, title)
    
    # Si encontramos una coincidencia, reordenamos el título
    if match:
        new_title = f"{match.group(2)} {match.group(1)}"
        return new_title
    else:
        return title

df_final3['title'] = df_final3['title'].apply(fix_titles)

df_complete = df_final3

display(df_complete)

# Guardar el dataframe final
joblib.dump(df_final3,"salidas\\df_complete.joblib")


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


###         SISTEMAS BASADOS EN POPULARIDAD
### --------------------------------------------------------------------------------

# Opciones para la lista desplegable principal
options = ['Top 10 Mejores Calificadas Global', 
           'Top 10 Más Vistas Global', 
           'Top 10 Más Vistas Último Año', 
           'Top 10 Mejores Calificadas Último Año']

# Crear una lista desplegable para las opciones
option_dropdown = widgets.Dropdown(
    options=options,
    description='Elige:',
    value='Top 10 Mejores Calificadas Global'  # Valor por defecto
)

# Función para mostrar el top según la selección del usuario
def show_top(selected_option):

    if selected_option == 'Top 10 Mejores Calificadas Global': 
        top_movie = top_a.iloc[0]['title']# Extraer el título de la película #1
        top_movie = top_movie[:-6]# Eliminar los últimos 6 caracteres
        print("EN EL TOP 1: ", top_movie)
        fn.post_img(top_movie)
        print("TOP 10 Mejores Calificadas Global")
        display(top_a)
    
    elif selected_option == 'Top 10 Más Vistas Global':
        top_movie = top_b.iloc[0]['title']# Extraer el título de la película #1
        top_movie = top_movie[:-6]# Eliminar los últimos 6 caracteres
        print("EN EL TOP 1: ", top_movie)
        fn.post_img(top_movie)
        print("TOP 10 Más Vistas Global")
        display(top_b)
    
    elif selected_option == 'Top 10 Más Vistas Último Año':
        top_movie = top_c.iloc[0]['title']# Extraer el título de la película #1
        top_movie = top_movie[:-6]# Eliminar los últimos 6 caracteres
        print("EN EL TOP 1: ", top_movie)
        fn.post_img(top_movie)
        print("TOP 10 Más Vistas Último Año")
        display(top_c)
    
    else:
        selected_option == 'Top 10 Mejores Calificadas Último Año'
        top_movie = top_d.iloc[0]['title']# Extraer el título de la película #1
        top_movie = top_movie[:-6]# Eliminar los últimos 6 caracteres
        print("EN EL TOP 1: ", top_movie)
        fn.post_img(top_movie)
        print("TOP 10 Mejores Calificadas Último Año")
        display(top_d)
    
# Conectar la función al evento de cambio de selección
widgets.interactive(show_top, selected_option=option_dropdown)

###         TOP 10 (MEJORES PELÍCULAS POR GÉNERO)
### --------------------------------------------------------------------------------

# Obtener los géneros únicos
genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Drama', 
          'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery', 
          'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

# Crear un widget de lista desplegable
genre_dropdown = widgets.Dropdown(
    options=genres,
    description='Elige un género:',
    value='Western'  # Género por defecto
)

def show_top_movies(selected_genre):
    # Filtrar las películas según el género seleccionado
    genre_movies = df_complete[df_complete[selected_genre] == 1]

    # Agrupar por título, año y calcular el promedio de las calificaciones y el conteo de vistas
    top_movies = genre_movies.groupby(['title', 'year']).agg(
        score=('movie_rating', 'mean'),  # Promedio de ratings
        views=('movie_rating', 'count')  # Contar el número de calificaciones
    ).reset_index()

    # Redondear el promedio de calificaciones a 2 decimales
    top_movies['score'] = top_movies['score'].round(2)

    # Filtrar por score mayor o igual a 4
    top_movies = top_movies[top_movies['score'] >= 4]

    # Ordenar por la cantidad de vistas en orden descendente y obtener las 10 mejores
    top_movies = top_movies.sort_values(by='views', ascending=False).head(10)

    # Mostrar el resultado
    # Extraer el título de la película #1
    top_movie = top_movies.iloc[0]['title']

    print("La película número 1 es: ", top_movie)
    fn.post_img(top_movie)  # Mostrar el póster de la película #1
    print("Aquí un listado con las películas que te podrían interesar: ")
    print("")
    
    # Mostrar el DataFrame con el año
    display(top_movies[['title', 'year', 'score', 'views']])
    print("")

# Conectar la función al evento de cambio de selección
widgets.interactive(show_top_movies, selected_genre=genre_dropdown)


####################################################################################
######## 2. SISTEMA DE RECOMENDACIÓN BASADO EN CONTENIDO UN SOLO PRODUCTO ##########
####################################################################################

movies = pd.read_sql('select * from movies_final', conn)
movies = movies.drop(['movieId:1', 'cnt_rat'], axis=1)

## ----- SEPARAR EL AÑO EN UNA NUEVA VARIABLE

movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')  # Extrae el año
movies['title'] = movies['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True)  # Elimina el año del título

## ----- SEPARAR LOS GÉNEROS EN COLUMNAS

# Separar los géneros en columnas teniendo en cuenta el criterio de separación '|'
genres_dummies = movies['genres'].str.get_dummies(sep='|')

# Concatenar las columnas de géneros con el DataFrame original
movies_sep = pd.concat([movies, genres_dummies], axis=1)

movies_sep['title'] = movies_sep['title'].apply(fix_titles)

## ----- EXPORTAR LOS DATOS
joblib.dump(movies_sep,"salidas\\movies_final.joblib")

## ----- ESCALAR EL AÑO PARA ESTAR EN EL MISMO RANGO
sc = MinMaxScaler()
movies_sep[["year_sc"]] = sc.fit_transform(movies_sep[["year"]])

## ----- ELIMINAR LAS VARIABLES QUE NO SE VAN A USAR
movies_dum = movies_sep.drop(columns=["movieId", "title", "genres", "year"]) 

## ----- EXPORTAR LOS DATOS
joblib.dump(movies_dum,"salidas\\movies_dum2.joblib")


###         SISTEMA DE RECOMENDACIÓN BASADO EN CONTENIDO DE UN SOLO PRODUCTO
### --------------------------------------------------------------------------------

def recomendacion(movie=sorted(list(movies_sep['title']))):
    
    # Obtener el índice de la película seleccionada
    ind_movie = movies_sep[movies_sep['title'] == movie].index.values.astype(int)[0]
    
    # Calcular la correlación entre la película seleccionada y todas las demás
    similar_movies = movies_dum.corrwith(movies_dum.iloc[ind_movie, :], axis=1)
    
    # Ordenar las correlaciones de mayor a menor
    similar_movies = similar_movies.sort_values(ascending=False)
    
    # Convertir en un DataFrame y redondear la correlación a 2 decimales
    top_similar_movies = similar_movies.to_frame(name="correlación").round(3)
    
    # Eliminar la película seleccionada de las recomendaciones
    top_similar_movies = top_similar_movies.drop(index=ind_movie)
    
    # Seleccionar las 10 mejores recomendaciones (sin la película original)
    top_similar_movies = top_similar_movies.iloc[0:10, :]
    
    # Agregar los títulos y el año de las películas correspondientes a los índices
    top_similar_movies['title'] = movies_sep.loc[top_similar_movies.index, 'title']
    top_similar_movies['year'] = movies_sep.loc[top_similar_movies.index, 'year']
    
    # Obtener el título de la primera película recomendada
    top_movie = top_similar_movies.iloc[0]['title']
    
    print("Si viste ", movie, " te recomendamos ", top_movie)
    fn.post_img(top_movie)
    print("Además algunos títulos adicionales que te podrían gustar: ")
    
    return top_similar_movies[['title', 'year', 'correlación']]

# Interactuar con la función de recomendación
print(interact(recomendacion))


###         SISTEMA DE RECOMENDACIÓN BASADO EN CONTENIDO KNN
### --------------------------------------------------------------------------------

## el coseno de un angulo entre dos vectores es 1 cuando son perpendiculares y 0 cuando son paralelos(indicando que son muy similar324e-06	3.336112e-01	3.336665e-01	3.336665e-es)
model = neighbors.NearestNeighbors(n_neighbors=11, metric='cosine')
model.fit(movies_dum)
dist, idlist = model.kneighbors(movies_dum)

distancias=pd.DataFrame(dist) ## devuelve un ranking de la distancias más cercanas para cada fila(libro)
id_list=pd.DataFrame(idlist) ## para saber esas distancias a que item corresponde

def MovieRecommender(movie=sorted(list(movies_sep['title'].value_counts().index))):
    movie_list = []  # Lista para almacenar nombres de películas recomendadas, años y sus distancias
    
    # Obtener el índice de la película seleccionada
    movie_id = movies_sep[movies_sep['title'] == movie].index
    movie_id = movie_id[0]
    
    # Recopilar los nombres de las películas recomendadas, años y las distancias
    for i, newid in enumerate(idlist[movie_id]):
        movie_name = movies_sep.loc[newid].title
        movie_year = movies_sep.loc[newid].year  # Obtener el año de la película
        distance = dist[movie_id][i]
        if movie_name != movie:  # Evitar recomendar la misma película seleccionada
            movie_list.append((movie_name, movie_year, distance))
    
    # Convertir la lista de recomendaciones a un DataFrame
    movie_recommendations_df = pd.DataFrame(movie_list, columns=['Movie', 'Year', 'Distance'])
    
    # Ordenar las recomendaciones por las distancias más cercanas
    movie_recommendations_df = movie_recommendations_df.sort_values(by='Distance', ascending=True).reset_index(drop=True)
    
    # Obtener la primera recomendación
    top_movie = movie_recommendations_df.iloc[0]['Movie']
    
    print(f"Si viste '{movie}', te recomendamos '{top_movie}'.")
    fn.post_img(top_movie)  # Mostrar la imagen del póster de la película recomendada
    print("Además, algunos títulos adicionales que te podrían gustar:")
    
    # Mostrar el DataFrame de las películas recomendadas con el año
    display(movie_recommendations_df)

# Interactuar con la función
print(interact(MovieRecommender))