###### LIBRERIAS
import numpy as np
import pandas as pd
import sqlite3 as sql
import plotly.graph_objs as go ### para gráficos
import plotly.express as px
#from mlxtend.preprocessing import TransactionEncoder
import a_funciones as fn

# Cargar datos
conn=sql.connect('data\\db_movies') ### crear cuando no existe el nombre de cd  y para conectarse cuando sí existe.
cur=conn.cursor() ###para funciones que ejecutan sql en base de datos

cur.execute('select name from sqlite_master where type = "table"')
cur.fetchall()

# Nombrar las tablas

movies = pd.read_sql("SELECT * from movies", conn)
movies # Se tiene una primera tabla *movies*, la cual contiene información 
# sobre las películas (título, año de lanzamineto y los géneros asociados a la misma)

ratings = pd.read_sql("SELECT * from ratings", conn)
ratings # Se tiene también, una segunda tabla, llamada *ratings*, la cual contiene información asociada 
# a las calificaciones que ha obtenido una película y la cantidad de usuarios que han calificado la película.

# Separar el año en una nueva columna
query = """
SELECT 
    movieId, 
    TRIM(SUBSTR(title, 1, INSTR(title, '(') - 1)) AS title,  -- Elimina el año del título
    SUBSTR(title, INSTR(title, '(') + 1, 4) AS year,         -- Extrae el año
    genres
FROM 
    movies;
"""

movies = pd.read_sql(query, conn)
movies

# Se procede a dummizar la columna de género, separando los carácteres contenidos en la misma.
# Esta dummización se hace con el fin de poder analizar más fácilmente la base de datos

# Separar los géneros en columnas teniendo en cuenta el criterio de separación '|'
genres_dummies = movies['genres'].str.get_dummies(sep='|')

# Concatenar las columnas de géneros con el DataFrame original
movies_sep = pd.concat([movies, genres_dummies], axis=1)
movies_sep

# Se elimina la columna 'genres' 
movies_sep = movies_sep.drop(columns=['genres'])
movies_sep.head()


# Cuántas películas hay por género 
genre_counts = movies_sep.iloc[:, 3:].sum()
print("Películas por género:")
genre_counts


# Calificaciones generales
cr = pd.read_sql("""SELECT
                 "rating" as rating,
                 count(*) as conteo
                 FROM ratings
                 group by "rating"
                 order by rating""", conn)

cr


# Consultar cuántas películas ha calificado cada usuario
rating_users = pd.read_sql('''
    SELECT userId AS user_id,
           COUNT(*) AS cnt_rat
    FROM ratings
    GROUP BY userId
    ORDER BY cnt_rat ASC
''', conn)

fig = px.histogram(rating_users, x='cnt_rat', 
                   title='Histograma de Frecuencia de Número de Calificaciones por Usuario',
                   labels={'cnt_rat': 'Número de Calificaciones'},
                   color_discrete_sequence=['skyblue'])

fig.update_layout(xaxis_title='Número de Calificaciones', yaxis_title='Frecuencia')
fig.show()

rating_users.describe()

# Consultar cuántas calificaciones tiene cada película
rating_movies = pd.read_sql(''' 
    SELECT movieId AS movie_id, 
           COUNT(*) AS cnt_rat 
    FROM ratings 
    GROUP BY movieId 
    ORDER BY cnt_rat DESC
''', conn)

# Mostrar las primeras filas del resultado
rating_movies.head()
rating_movies.describe()

fig  = plt.hist(rating_movies['cnt_rat'])













