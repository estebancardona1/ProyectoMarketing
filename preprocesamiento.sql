---tabla con las películas que han sido calificadas por más de 20 usuarios

drop table if exists user_sel;

create table user_sel as select "userId" as user_id,
                         count(*) as cnt_rat
                         FROM ratings
                         group by userId
                         having cnt_rat
                         order by cnt_rat asc ;

---

drop table if exists movies_sel;

create table movies_sel as select movieId,
                         count(*) as cnt_rat
                         from ratings
                         group by movieId
                         having cnt_rat >= 20
                         order by cnt_rat desc ;                  

-------crear tablas filtradas de películas, usuarios y calificaciones ----

drop table if exists merge_ratings;

create table merge_ratings as
select a.userId as user_id,
a.movieId as movie_id,
a.rating as movie_rating
a.timestamp as timestamp
from ratings a 
inner join movies_sel b
on a.movieId = b.movieId;

---crear tabla completa ----

drop table if exists ratings_final;

CREATE TABLE ratings_final AS
SELECT a.user_id,
       a.movie_id,
       a.movie_rating,
       a.timestamp,
       b.*
FROM merge_ratings a
INNER JOIN movies_sel b
ON a.movie_id = b.movieId;