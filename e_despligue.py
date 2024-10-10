import numpy as np
import pandas as pd
import sqlite3 as sql
import a_funciones as fn  # para procesamiento
import openpyxl
import joblib
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

def preprocesar():
    # Conectar base de datos
    conn = sql.connect('C:\\Users\\juane\\OneDrive\\Escritorio\\A\\Universidad UdeA\\ANALITICA III\\ProyectoMarketing\\data\\db_movies')
    cur = conn.cursor()

    # Ejecutar SQL para filtrar usuarios
    fn.ejecutar_sql('C:\\Users\\juane\\OneDrive\\Escritorio\\A\\Universidad UdeA\\ANALITICA III\\ProyectoMarketing\\b_preprocesamiento.sql', cur)

    # Cargar datos
    movies = pd.read_sql('select * from movies_final', conn)
    ratings = pd.read_sql('select * from ratings', conn)

    # Preprocesamiento de películas
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')
    movies['title'] = movies['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True)
    movies = movies.dropna(subset=['year'])
    movies = movies.sort_values(by='year')

    # Crear variables dummy para géneros
    genres_dummies = movies['genres'].str.get_dummies(sep='|')
    
    # Combinar movies con genres_dummies
    movies_sep = pd.concat([movies, genres_dummies], axis=1)


    # Escalar el año
    sc = MinMaxScaler()
    movies_sep["year_sc"] = sc.fit_transform(movies_sep[["year"]])


    # Crear movies_dum para el modelo
    movies_dum = movies_sep.drop(columns=["movieId", "title", "genres", "year"])

    # Guardar datos preprocesados
    joblib.dump(movies_sep, "salidas/movies_only.joblib") 
    joblib.dump(movies_dum, "salidas/movies_dum.joblib")

    return ratings, movies_sep, conn, cur

def recomendar_peliculas(user_id):
    # Cargar datos preprocesados
    movies_dum = joblib.load("salidas/movies_dum.joblib")
    movies_sep = joblib.load("salidas/movies_only.joblib")
    
    # Conectar a la base de datos
    conn = sql.connect('C:\\Users\\juane\\OneDrive\\Escritorio\\A\\Universidad UdeA\\ANALITICA III\\ProyectoMarketing\\data\\db_movies')
    
    # Obtener las calificaciones del usuario específico
    ratings = pd.read_sql('select * from ratings_final where userId=:user', conn, params={'user': user_id})
    l_movies_r = ratings['movieId'].to_numpy()
    
    # Separar películas calificadas y no calificadas por el usuario
    movies_r = movies_dum[movies_dum.index.isin(l_movies_r)] #recomendadas
    movies_nr = movies_dum[~movies_dum.index.isin(l_movies_r)]#no recomendadas
    
    if movies_r.empty:
        conn.close()
        return pd.DataFrame()  # Retornar DataFrame vacío si el usuario no ha calificado películas
    
    # Calcular el centroide de las películas calificadas
    centroide = movies_r.mean().to_frame().T
    
    # Usar NearestNeighbors para encontrar películas similares
    model = NearestNeighbors(n_neighbors=min(11, len(movies_nr)), metric='cosine')
    model.fit(movies_nr)
    dist, idlist = model.kneighbors(centroide)
    
    # Obtener las películas recomendadas
    ids = idlist[0]
    recomend_m = movies_sep.loc[movies_nr.index[ids]][['title', 'year']]
    
    conn.close()
    
    return recomend_m

def main(list_user):
    # Asegurarse de que los datos preprocesados estén disponibles
    preprocesar()
    
    recomendaciones_todos = pd.DataFrame()
    for user_id in list_user:
        recomendaciones = recomendar_peliculas(user_id)
        if not recomendaciones.empty:
            recomendaciones["user_id"] = user_id
            recomendaciones.reset_index(inplace=True, drop=True)
            recomendaciones_todos = pd.concat([recomendaciones_todos, recomendaciones])

    if not recomendaciones_todos.empty:
        recomendaciones_todos.to_excel('C:\\Users\\juane\\OneDrive\\Escritorio\\A\\Universidad UdeA\\ANALITICA III\\ProyectoMarketing\\recomendaciones.xlsx', index=False)
        print("Archivo de recomendaciones generado exitosamente.")
    else:
        print("No se generaron recomendaciones para los usuarios proporcionados.")

if __name__ == "__main__":
    list_user = [52853, 31226, 167471, 8066]
    main(list_user)

import sys
print(sys.executable)

