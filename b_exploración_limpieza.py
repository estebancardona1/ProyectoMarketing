###         LIBRERIAS
### --------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import sqlite3 as sql
import plotly.graph_objs as go 
import plotly.express as px
import a_funciones as fn
from IPython.display import display
import matplotlib.pyplot as plt

###         CARGAR DATOS
### --------------------------------------------------------------------------------

# Crear cuando no existe el nombre de cd y para conectarse cuando sí existe.

conn = sql.connect('data\\db_movies')
cur = conn.cursor()
#conn.close() ### cerrar conexion base de datos


# Para funciones que ejecutan sql en base de datos.
cur.execute('select name from sqlite_master where type = "table"')
cur.fetchall()

movies = pd.read_sql("SELECT * from movies", conn)
display(movies) # Base de datos películas (MovieID, título, géneros)

ratings = pd.read_sql("SELECT * from ratings", conn)
display(ratings) # Base de datos calificaciones (UserID, Movie ID, título, género)

###         TRATAMIENTO DE DATOS
### --------------------------------------------------------------------------------

## ----- TÍTULOS DUPLICADOS

pd.read_sql("""SELECT title, COUNT(*) as count
            FROM movies
            GROUP BY title
            HAVING COUNT(*) > 1;""", conn)

#COMENTARIO: Se identifican algunos títulos duplicados (5)


## ----- SEPARAR EL AÑO EN UNA NUEVA VARIABLE

movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')  # Extrae el año
movies['title'] = movies['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True)  # Elimina el año del título
movies = movies.dropna(subset=['year'])
movies = movies.sort_values(by='year')

display(movies) # Base de datos películas (MovieID, título, Año, Géneros)

ratings = pd.read_sql("""SELECT userId, movieId, rating, 
                    strftime('%Y', datetime(timestamp, 'unixepoch')) AS year_view 
                    FROM ratings""", conn)


display(ratings)

## ----- SEPARAR LOS GÉNEROS EN COLUMNAS

# Separar los géneros en columnas teniendo en cuenta el criterio de separación '|'
genres_dummies = movies['genres'].str.get_dummies(sep='|')

# Sumar cada género y ordenar de mayor a menor
genre_counts = genres_dummies.sum().sort_values(ascending=False)

## ----- GRAFICAR LOS GÉNEROS

# Calcular el total y el porcentaje
total_movies = genre_counts.sum()
percentages = (genre_counts / total_movies) * 100

# Crear un gráfico de barras
plt.figure(figsize=(12, 6))
bars = genre_counts.plot(kind='bar', color=('#532D87'))

# Añadir etiquetas de cantidad y porcentaje
for bar, count, percentage in zip(bars.patches, genre_counts, percentages):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{percentage:.1f}%',
             ha='center', va='bottom')

plt.title('Distribución de Géneros')
plt.xlabel('Géneros')
plt.ylabel('Cantidad de Películas')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

## ----- CALIFICACIONES GENERALES

cr = pd.read_sql("""
    SELECT 
        "rating" AS rating,
        COUNT(*) AS conteo,
        (COUNT(*) * 100.0 / SUM(COUNT(*)) OVER ()) AS porcentaje
        FROM ratings
        GROUP BY "rating"
        ORDER BY "rating"
""", conn)

pd.read_sql("select count(*) from ratings", conn)

# Crear un gráfico de barras
plt.figure(figsize=(12, 6))

# Definir colores según el rango de rating
colors = []
for rating in cr['rating']:
    if rating == 0.5:
        colors.append('#264653')
    elif 1 <= rating <= 2:
        colors.append('#0e7774')
    elif 2.5 <= rating <= 3.5:
        colors.append('#069a7e')
    elif 4 <= rating <= 5:
        colors.append('#3ebdac')

# Crear el gráfico de barras
bars = plt.bar(cr['rating'], cr['conteo'], color=colors, width=0.4, zorder=3)

# Añadir etiquetas de cantidad y porcentaje
for bar, count, percentage in zip(bars, cr['conteo'], cr['porcentaje']):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{percentage:.1f}%',
             ha='center', va='bottom', zorder=4)  # Asegurar que las etiquetas estén encima

# Estilizar el gráfico
plt.title('Distribución de Ratings')
plt.xlabel('Rating')
plt.ylabel('Conteo de Ratings')

# Añadir cuadrículas detrás de las barras
plt.grid(axis='y', zorder=1)  # Zorder menor para las cuadrículas

# Configurar las marcas en el eje x
plt.xticks(cr['rating'])
plt.tight_layout()
plt.show()

# COMENTARIO: La base de datos inicial tiene un total de 100.836 películas calificadas por los
#             usuarios. Aproximadamente el 1.4% de los datos está en 0.5, que, para casos prácticos
#             se tomará como calificación implícita y será extraido del análisis para los resultados. 

## ----- CALIFICACIONES DE LOS USUARIOS

rating_users = pd.read_sql(''' SELECT "userId" as userId,
                         count(*) as cnt_rat
                         FROM ratings
                         group by "userId"
                         order by cnt_rat asc
                         ''',conn)

# Histograma número de calificaciones por usuario 

#plot_histogram(dataframe, count_column, title='Histograma', xlabel='Número de calificaciones', bins=20, color=("#532D87"))

fn.plot_histogram(rating_users, 'cnt_rat',title = "Calificaciones por usuario (Original)", xlabel='Número de calificaciones',bins=20, color='#532D87')

# Filtro de cantidad de ratings

rating_users2=pd.read_sql(''' select "userId" as userId,
                         count(*) as cnt_rat
                         FROM ratings
                         group by "userId"
                         having cnt_rat <=1000
                         order by cnt_rat asc
                         ''',conn )

# COMENTARIO: si consideramos que una persona ve, en promedio, una película a la semana, 
#             eso suma alrededor de 52 películas al año. Algunas personas ven más, especialmente si 
#             son cinéfilas, mientras que otras podrían ver menos. 
#             Tienendo en cuenta esto, la base de datos cuenta con registros de 22 años, por lo que
#             52 películas al año x 22 años = 1144 películas vistas (o calificadas)


# Histograma número de calificaciones por usuario 

fn.plot_histogram(rating_users2, 'cnt_rat',title = "Calificaciones por usuario (Tratado)", xlabel='Número de calificaciones',bins=20, color='#532D87')

# COMENTARIO: Los histogramas al compararlos muestran la diferencia en la distribución
#             en los datos al filtrarlos teniendo en cuenta los criterios de limitación
#             preferiblemente menores de 1200 calificaciones por usuario, basados en la
#             información mencionada anteriormente.

rating_movies=pd.read_sql(''' select movieId,
                         count(*) as cnt_rat
                         from ratings
                         group by "movieId"
                         order by cnt_rat desc
                         ''',conn)

fn.plot_histogram(rating_movies, 'cnt_rat',title = "Cantidad de películas calificadas (Original)", xlabel='Número de calificaciones',bins=20, color='#532D87')

rating_movies2=pd.read_sql(''' select movieId ,
                         count(*) as cnt_rat
                         from ratings
                         group by "movieId"
                         having cnt_rat >= 20
                         order by cnt_rat desc
                         ''',conn )

fn.plot_histogram(rating_movies2, 'cnt_rat',title = "Cantidad de películas calificadas (Tratado)", xlabel='Número de calificaciones',bins=20, color='#532D87')

# COMENTARIO: Los histogramas al compararlos muestran la diferencia en la distribución
#             en los datos al filtrarlos teniendo en cuenta los criterios de limitación
#             preferiblemente películas con más de 20 calificaciones.

## ----- CALIFICACIONES DE LOS USUARIOS POR CADA CALIFICACIÓN

# Separar las calificaciones por puntaje
ratings_list = np.arange(0.5, 5.5, 0.5)

# Contar cuántas veces calificó cada usuario
rating_counts = ratings.groupby('userId')['rating'].value_counts().unstack(fill_value=0)

# Reindexar para asegurarse de que todas las calificaciones estén presentes
rating_counts = rating_counts.reindex(columns=ratings_list, fill_value=0)

# Añadir la columna de total de calificaciones
rating_counts['Total'] = rating_counts.sum(axis=1)

rating_counts = rating_counts.sort_values(by='Total', ascending=False)

print("Distribución de las calificaciones por usuario: ")
print("")
display(rating_counts)

# Crear una nueva matriz binaria
binary_matrix = (rating_counts.iloc[:, :-1] >= 1).astype(int)

# Calcular la columna Total en la matriz binaria
binary_matrix['Total'] = binary_matrix.sum(axis=1)

# Mostrar la nueva matriz con Total
print("Matriz binaria de la distribucion de las calificaciones: ")
print("")
display(binary_matrix)

# COMENTARIO: Se crea una matriz binaria con el fin de identificar el comportamiento
#             en los rangos de calificaciones de los usuarios, es decir, si un usuario
#             ha calificado al menos 1 vez una película en alguna de las puntuaciones
#             este se convertirá en 1, mientras que si no ha hecho ninguna calificación
#             en alguno de los valores este se convertirá en 0.


#Convertir matriz a dataframe
id_binary = binary_matrix.index.tolist()

binary_total = binary_matrix['Total'].tolist()

# Unión de ID con Total
df_binary = pd.DataFrame()
df_binary['userId'] = id_binary
df_binary['Range'] = binary_total
display(df_binary)

# Pegar total de 'df_binary' en 'rating_counts' según 'userId'
df_merged_ratings = pd.merge(rating_counts,df_binary, on=['userId'], how='outer')
df_merged_ratings.sort_values(by='Total', ascending=False)

#Visualización cantidad de calificaciones por puntaje

# Contar cuántos usuarios tienen cada total
total_counts = binary_matrix['Total'].value_counts().sort_index()

## ----- DISTRIBUCIÓN DE LOS RANGOS DE CALIFICACIÓN DE LOS USUARIOS

# Crear el gráfico de barras
plt.figure(figsize=(10, 6))

plt.bar(total_counts.index, total_counts.values, color='#532D87')

# Personalizar el diseño del gráfico
plt.title("Distribución en los rangos de calificaciones")
plt.xlabel('Distribución de los rangos')
plt.ylabel('Número de Usuarios')

# Ajustar el tamaño y la presentación
plt.tight_layout()

# Mostrar el gráfico
plt.show()

# Visualización 2

# Definir los bins
bins = [0, 2, 4, 6, 11]  # Bins para los rangos (1-2), (3-4), (5-6), (+6)
labels = ['1-2', '3-4', '5-6', '+6']  # Etiquetas para los bins

# Agrupar los totales en los bins definidos
total_binned = pd.cut(total_counts.index, bins=bins, labels=labels, right=True)

# Contar cuántos usuarios caen en cada bin
binned_counts = total_counts.groupby(total_binned).sum().reindex(labels, fill_value=0)

# Crear el gráfico de barras
plt.figure(figsize=(10, 6))

plt.bar(binned_counts.index, binned_counts.values, color='#532D87')

# Personalizar el diseño del gráfico
plt.title("Distribución en los rangos de calificaciones (Agrupado)")
plt.xlabel('Total de Calificaciones (bins)')
plt.ylabel('Número de Usuarios')

# Ajustar el tamaño y la presentación
plt.tight_layout()

# Mostrar el gráfico
plt.show()

# COMENTARIO: Para este caso, la cantidad de usuarios que hacen una calificación
#             en menos de dos rangos es despreciable con respecto a las demás rangos
#             de calificaciones, por eso, no se considera la eliminación de los mismos
#             sin embargo es un críterio que se debería considerar en casos posteriores.


####################################################################################
###         P R E P R O C E S A M I E N T O
### --------------------------------------------------------------------------------

fn.ejecutar_sql('b_preprocesamiento.sql', cur)

cur.execute("select name from sqlite_master where type='table' ")
cur.fetchall()

### verficar tamaño de tablas con filtros ####

## ratings
pd.read_sql('select count(*) from ratings', conn)
pd.read_sql('select count(*) from ratings_final', conn)

## movies
pd.read_sql('select count(*) from movies', conn)
pd.read_sql('select count(*) from movies_final', conn)

## tabla final
pd.read_sql('select count(*) from full_ratings', conn)

final=pd.read_sql('select * from full_ratings',conn)
final.duplicated().sum() ## al cruzar tablas a veces se duplican registros
final.info()
final