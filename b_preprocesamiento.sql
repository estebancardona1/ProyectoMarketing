-- --- USER_SEL: Usuarios con más de 20 calificaciones y menos de 1000
DROP TABLE IF EXISTS user_sel;

CREATE TABLE user_sel AS 
SELECT userId, COUNT(*) AS cnt_rat
FROM ratings
GROUP BY userId
HAVING cnt_rat >= 20 AND cnt_rat <= 1000
ORDER BY cnt_rat asc;

-- --- MOVIES_SEL: Películas con más de 20 calificaciones
DROP TABLE IF EXISTS movies_sel;

CREATE TABLE movies_sel AS 
SELECT movieId, COUNT(*) AS cnt_rat
FROM ratings
GROUP BY movieId
HAVING cnt_rat >= 20
ORDER BY cnt_rat DESC;

-- --- RATINGS_FINAL
DROP TABLE IF EXISTS ratings_final;

CREATE TABLE ratings_final AS 
SELECT a.userId, a.movieId, a.rating, a.timestamp
FROM ratings a
INNER JOIN movies_sel b ON a.movieId = b.movieId
INNER JOIN user_sel c ON a.userId = c.userId;

-- --- MOVIE_FINAL
DROP TABLE IF EXISTS movies_final;

CREATE TABLE movies_final AS 
SELECT 
    a.movieId, 
    TRIM(SUBSTR(a.title, 1, LENGTH(a.title) - 6)) AS title,  -- Eliminar el año del título
    a.genres,
    SUBSTR(a.title, LENGTH(a.title) - 4, 4) AS year  -- Extraer el año de los últimos 6 caracteres
FROM movies a
INNER JOIN movies_sel c ON a.movieId = c.movieId
WHERE INSTR(a.title, '(') > 0;  -- Asegurar que hay al menos un paréntesis


-- --- TABLA COMPLETA
DROP TABLE IF EXISTS full_ratings;

CREATE TABLE full_ratings AS 
SELECT 
    a.userId, 
    a.movieId, 
    a.rating, 
    b.title,
    b.year, 
    b.genres, 
    a.timestamp
FROM ratings_final a 
INNER JOIN movies_final b ON a.movieId = b.movieId
WHERE a.rating > 0.5;