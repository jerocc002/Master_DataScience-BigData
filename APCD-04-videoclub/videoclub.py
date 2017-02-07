"""
  Author:  Jerónimo Carranza Carranza
  Created: 19-dic-2016
"""
    
import sqlalchemy

url = 'mysql+mysqlconnector://root:admin_mysql@localhost:3306/videoclub'
mysql = sqlalchemy.create_engine(url)


def create_user(name, phone, address):
    sqlq = "INSERT INTO users (name, phone, address) VALUES (%s,%s,%s)"
    params = (name, phone, address)
    mysql.execute(sqlq, params)


def delete_user(user_id):
    sqlq = "DELETE * FROM users WHERE user_id = %s"
    params = (user_id)
    mysql.execute(sqlq, params)


def get_user(user_id):
    sqlq = "SELECT * FROM users WHERE id = %s"
    params = (user_id)
    results = mysql.execute(sqlq, params)
    for r in results:
        print(r)
    return results


def search_user(user_name):
    sqlq = "SELECT * FROM users WHERE name LIKE %s"
    params = (user_name+'%')
    results = mysql.execute(sqlq, params)
    for r in results:
        print(r)
    return results


def create_movie(title, year):
    sqlq = "INSERT INTO movies (title, year) VALUES (%s,%s)"
    params = (title, year)
    mysql.execute(sqlq, params)    


def delete_movie(movie_id):
    sqlq = "DELETE * FROM movies WHERE movie_id = %s"
    params = (movie_id)
    mysql.execute(sqlq, params)


def get_movie(movie_id):
    sqlq = "SELECT * FROM movies WHERE id = %s"
    params = (movie_id)
    results = mysql.execute(sqlq, params)
    for r in results:
        print(r)
    return results


def search_movie(movie_title):
    sqlq = "SELECT * FROM movies WHERE title LIKE %s"
    params = (movie_title+'%')
    results = mysql.execute(sqlq, params)
    for r in results:
        print(r)
    return results


def is_movie_available(movie_id):
    available = True
    sqlq = "SELECT count(*) N FROM rentals "
    sqlq += "WHERE returnday is NULL AND movie_id = %s"
    params = (movie_id)
    results = mysql.execute(sqlq, params)
    for r in results:
        # print(r)
        if r[0] != 0:
            available = False
    return available


def rent_movie(user_id, movie_id):
    if is_movie_available(movie_id):
        sqlq = "INSERT INTO rentals (user_id, movie_id) VALUES (%s,%s)"
        params = (user_id, movie_id)
        mysql.execute(sqlq, params)
    else:
        print('Película no disponible')


def deliver_movie(user_id, movie_id): 
    sqlSelect  = "SELECT * FROM rentals "
    sqlUpdate  = "UPDATE rentals SET returnday = CURRENT_TIMESTAMP() "
    sqlWhere  = "WHERE user_id = %s AND movie_id = %s AND returnday is NULL"
    params = (user_id, movie_id)
    results = mysql.execute(sqlSelect+sqlWhere, params)
    r = results.fetchone()
    # print(r)
    if r:
        mysql.execute(sqlUpdate+sqlWhere, params)
        print ('Devolución realizada')
    else:
        print('No es posible la devolución')


def get_rental_history():
    sqlq = "SELECT r.id, r.user_id, r.movie_id, u.name, m.title, count(r.id) NRentals,  " \
           "MIN(exitday) AS fromDate, MAX(exitday) AS toDate " \
           "FROM rentals r " \
           "JOIN users u ON r.user_id=u.id " \
           "JOIN movies m ON r.movie_id=m.id " \
           "GROUP BY r.user_id, r.movie_id " \
           "ORDER BY u.name, m.title"
    results = mysql.execute(sqlq)
    for r in results:
        print(r)


