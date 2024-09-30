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

# VISUALIZACIÓN DE LOS DATOS

cr = pd.read_sql("""
    SELECT 
        "rating" AS rating,
        COUNT(*) AS conteo,
        (COUNT(*) * 100.0 / SUM(COUNT(*)) OVER ()) AS porcentaje
    FROM ratings
    GROUP BY "rating"
    ORDER BY "rating"
""", conn)

cr


# Visualización del conteo de calificaciones

pd.read_sql("select count(*) from ratings", conn)

# Definir los colores según las calificaciones
colors = []
for rating in cr['rating']:
    if rating == 0.5:
        colors.append('#264653')  
    elif 1 <= rating <= 2:
        colors.append('#fe4a49')  
    elif 2.5 <= rating <= 3.5:
        colors.append('#fed766')  
    elif 4 <= rating <= 5:
        colors.append('#009fb7')  

data  = go.Bar( x=cr.rating,y=cr.conteo, text=cr.conteo, textposition="outside", marker_color=colors)

layout = go.Layout(
    title={
        'text': "Conteo de Calificaciones",
        'y': 0.94,
        'x': 0.5, 
        'xanchor': 'center',  # Anclar el título al centro
        'yanchor': 'top'
    },
    xaxis={
        'title': 'Calificación',
        'tickvals': cr['rating']  # Asegurar que todos los valores del eje X se muestren
    }, 
    yaxis={'title': 'Cantidad'},
    width=800,   # Ancho del gráfico
    height=600   # Alto del gráfico
)

# Crear la figura y mostrar
fig = go.Figure(data=data, layout=layout)
fig.show()


# Calificaciones por usuario
rating_users=pd.read_sql(''' SELECT "userId" as user_id,
                         count(*) as cnt_rat
                         FROM ratings
                         group by "userId"
                         order by cnt_rat asc
                         ''',conn)

rating_users

# Histograma número de calificaciones por usuario
fn.plot_histogram(rating_users, 'cnt_rat', bins=20, color='#264653')

## PREPROCESAMIENTO


