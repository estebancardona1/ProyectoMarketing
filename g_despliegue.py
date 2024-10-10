import numpy as np
import pandas as pd
import sqlite3 as sql
import a_funciones as fn ## para procesamiento
import openpyxl
import joblib
import re

####Paquete para sistema basado en contenido ####
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors

def preprocesar():

    #### conectar_base_de_Datos#################
    conn=sql.connect('C:\\Users\\juane\\OneDrive\\Escritorio\\A\\Universidad UdeA\\ANALITICA III\\ProyectoMarketing\\data\\db_movies')
    cur=conn.cursor()

    # ----- FUNCIÓN PARA CORREGIR ALGUNOS TÍTULOS

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
    
    ######## convertir datos crudos a bases filtradas por usuarios que tengan cierto número de calificaciones
    fn.ejecutar_sql('C:\\Users\\juane\\OneDrive\\Escritorio\\A\\Universidad UdeA\\ANALITICA III\\ProyectoMarketing\\b_preprocesamiento.sql', cur)

    ##### llevar datos que cambian constantemente a python ######
    movies=pd.read_sql('select * from movies_final', conn )
    ratings=pd.read_sql('select * from full_ratings', conn)
    usuarios=pd.read_sql('select distinct userId from full_ratings', conn)

    
    #### transformación de datos crudos - Preprocesamiento ################
     # Preprocesamiento de películas
    #movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')
    #movies['title'] = movies['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True)
    #movies = movies.dropna(subset=['year'])
    #movies = movies.sort_values(by='year')

# Escalar el año
    movies_dum= joblib.load('salidas\\movies_dum.joblib')
    movies_final=joblib.load('salidas\\movies_only.joblib')


    return ratings, movies_final, movies,movies_dum, conn, cur





def recomendar(userId):
    
    ratings, movies_final, movies,movies_dum, conn, cur= preprocesar()

    ###seleccionar solo los ratings del usuario seleccionado
    
    ratings=pd.read_sql('select * from full_ratings where userId=:user',conn, params={'user':userId,})
    
    ###convertir ratings del usuario a array
    l_movies_w=ratings['movieId'].to_numpy()
    
    ###agregar la columna de movieId y titulo de la película a dummie para filtrar y mostrar nombre
    movies_dum[['movieId','title']]=movies_final[['movieId','title']]
    
    ### filtrar películas calificados por el usuario
    movies_w=movies_dum[movies_dum['movieId'].isin(l_movies_w)]
    
    ## eliminar columna title e movieId
    movies_w=movies_w.drop(columns=['movieId','title'])
    movies_w["indice"]=1 ### para usar group by y que quede en formato pandas tabla de centroide
    
    ##centroide o perfil del usuario
    centroide=movies_w.groupby("indice").mean()
    
    ### filtrar pelícuas no vistas
    movies_nw=movies_dum[~movies_dum['movieId'].isin(l_movies_w)]
    
    ## eliminar nombre e movieId
    movies_nw=movies_nw.drop(columns=['movieId','title'])
    
    ### entrenar modelo 
    model=neighbors.NearestNeighbors(n_neighbors=11, metric='cosine')
    model.fit(movies_nw)
    dist, idlist = model.kneighbors(centroide)
    
    ids=idlist[0] ### queda en un array anidado, para sacarlo
    recomend=movies_final.loc[ids][['movieId','title','year']]
    
    ## ENSAYOOOOO
    rating_data = pd.read_sql('''SELECT movieId,ROUND(AVG(rating), 2) as score, 
                                COUNT(rating) as views FROM full_ratings
                                GROUP BY movieId''', conn)
    
    
    ### Unir los datos de rating y views a las recomendaciones
    recomend = recomend.merge(rating_data, on='movieId', how='left')
    
    recomend = recomend.drop(columns=['movieId'])
    
    recomend = recomend[recomend['score'] >= 3]
    
    recomend = recomend[recomend['views'] >= 20]
    
    ## ENSAYOOOOO
    
    # Obtener el título de la primera película recomendada
    top_movie = recomend.iloc[0]['title']
    print("Basado en tus gustos creemos que te podría gustar: ", top_movie)
    fn.post_img(top_movie)  # Mostrar imagen del póster de la película recomendada
    print("Además algunos títulos adicionales que te podrían gustar: ")
    
    
    vistos=movies_final[movies_final['movieId'].isin(l_movies_w)][['title','movieId']]
    
    return recomend


def main(list_user):
        
    recomendaciones_todos=pd.DataFrame()
    for userId in list_user:
            
        recomendaciones=recomendar(userId)
        recomendaciones["userId"]=userId
        recomendaciones.reset_index(inplace=True,drop=True)
        
        recomendaciones_todos=pd.concat([recomendaciones_todos, recomendaciones])

    recomendaciones_todos.to_excel('C:\\Users\\juane\\OneDrive\\Escritorio\\A\\Universidad UdeA\\ANALITICA III\\ProyectoMarketing\\salidas\\reco\\recomendaciones.xlsx')
    recomendaciones_todos.to_csv('C:\\Users\\juane\\OneDrive\\Escritorio\\A\\Universidad UdeA\\ANALITICA III\\ProyectoMarketing\\salidas\\reco\\recomendaciones.csv')


if __name__=="__main__":
    list_user=[52853,31226,167471,8066 ]
    main(list_user)
    

import sys
sys.executable