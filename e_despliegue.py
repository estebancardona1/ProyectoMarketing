###         LIBRERIAS
### --------------------------------------------------------------------------------

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

###         TAREAS PARA EL DESPLIEGUE
### --------------------------------------------------------------------------------


def preprocesar():

    #### conectar_base_de_Datos#################
    conn=sql.connect('D:\\Desktop\\Universidad\\2024-2\\Analítica\\Repositorios\\ProyectoMarketing\\data\\db_movies')
    cur=conn.cursor()
    
    fn.ejecutar_sql('D:\\Desktop\\Universidad\\2024-2\\Analítica\\Repositorios\\ProyectoMarketing\\b_preprocesamiento.sql', cur)
    
    movies=pd.read_sql('select * from movies_final', conn )
    #ratings=pd.read_sql('select * from ratings_final', conn)
    usuarios=pd.read_sql('select distinct userId from full_ratings', conn) 
    
    
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

    ## ----- SEPARAR LOS GÉNEROS EN COLUMNAS

    # Separar los géneros en columnas teniendo en cuenta el criterio de separación '|'
    genres_dummies = movies['genres'].str.get_dummies(sep='|')

    # Concatenar las columnas de géneros con el DataFrame original
    movies_sep = pd.concat([movies, genres_dummies], axis=1)

    movies_sep['title'] = movies_sep['title'].apply(fix_titles)
    
        ## ----- ESCALAR EL AÑO PARA ESTAR EN EL MISMO RANGO
    sc = MinMaxScaler()
    movies_sep[["year_sc"]] = sc.fit_transform(movies_sep[["year"]])

    ## ----- ELIMINAR LAS VARIABLES QUE NO SE VAN A USAR
    movies_dum = movies_sep.drop(columns=["movieId", "title", "genres", "year"]) 
    
    return movies, movies_dum, movies_sep, conn, cur

## ----- SELECCIONAR USUARIOS PARA LA RECOMENDACIÓN

def recomendar(userId):
    
    movies, movies_dum, movies_sep, conn, cur = preprocesar()
    
    ###seleccionar solo los ratings del usuario seleccionado
    ratings=pd.read_sql('select * from full_ratings where userId=:user',conn, params={'user':userId,})
    
    ###convertir ratings del usuario a array
    l_movies_w=ratings['movieId'].to_numpy()
    
    ###agregar la columna de movieId y titulo de la película a dummie para filtrar y mostrar nombre
    movies_dum[['movieId','title']]=movies_sep[['movieId','title']]
    
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
    recomend=movies_sep.loc[ids][['movieId','title','year']]
    
    ## ENSAYOOOOO
    rating_data = pd.read_sql('''SELECT movieId,ROUND(AVG(rating), 2) as score, 
                                COUNT(rating) as views FROM full_ratings
                                GROUP BY movieId''', conn)
    
    
    ### Unir los datos de rating y views a las recomendaciones
    recomend = recomend.merge(rating_data, on='movieId', how='left')
    
    recomend = recomend.drop(columns=['movieId'])
    
    recomend = recomend[recomend['score'] >= 3]
    
    recomend = recomend[recomend['views'] >= 20]
    
    vistos=movies_sep[movies_sep['movieId'].isin(l_movies_w)][['title','movieId']]
    
    return recomend

###         ITEREAR SOBRE LOS USUARIOS Y EXPORTAR LOS ARCHIVOS 
### --------------------------------------------------------------------------------


def main(list_user):
        
    recomendaciones_todos=pd.DataFrame()
    for userId in list_user:
            
        recomendaciones=recomendar(userId)
        recomendaciones["userId"]=userId
        recomendaciones.reset_index(inplace=True,drop=True)
        
        recomendaciones_todos=pd.concat([recomendaciones_todos, recomendaciones])

    recomendaciones_todos.to_excel('D:\\Desktop\\Universidad\\2024-2\\Analítica\\Repositorios\\ProyectoMarketing\\salidas\\reco\\recomendaciones.xlsx')
    recomendaciones_todos.to_csv('D:\\Desktop\\Universidad\\2024-2\\Analítica\\Repositorios\\ProyectoMarketing\\salidas\\reco\\recomendaciones.csv')


if __name__=="__main__":
    list_user=[1,2,3,4,5,609]
    main(list_user)
    

import sys
sys.executable