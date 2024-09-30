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

drop table if exists ratings_filtered;

create table ratings_filtered as
select a.userId as user_id,
a.movieId as movie_id,
a.rating as movie_rating,
a.timestamp as timestamp
from ratings a 
inner join movies_sel b
on a.movieId = b.movieId;
 
drop table if exists movies_final;

create table movies_final as select 
a.*,
b.*
from movies a inner join
movies_sel b on a.movieId = b.movieId;

---crear tabla completa ----

drop table if exists ratings_final;

create table ratings_final as
select a.user_id,
       a.movie_id,
       a.movie_rating,
       a.timestamp,
       b.*
from ratings_filtered a
inner join movies_final b
on a.movie_id = b.movieId
WHERE a.movie_rating > 0.5;