/**
 * Author:  Jerónimo Carranza Carranza
 * Created: 19-dic-2016
 */

CREATE SCHEMA videoclub;

USE videoclub;

CREATE TABLE users (
    id INT NOT NULL AUTO_INCREMENT,
    name VARCHAR(256),
    phone VARCHAR(256) NOT NULL,
    address  VARCHAR(256) NOT NULL,
    PRIMARY KEY (id)
);

CREATE TABLE movies (
    id INT NOT NULL AUTO_INCREMENT,
    title VARCHAR(512) NOT NULL,
    year INT NOT NULL,
    PRIMARY KEY (id)
);

CREATE TABLE rentals (
    id INT NOT NULL AUTO_INCREMENT,
    user_id INT NOT NULL,
    movie_id INT NOT NULL,
    exitday TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    returnday DATETIME,
    PRIMARY KEY(id),
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (movie_id) REFERENCES movies(id)
); 

INSERT INTO users (name, phone, address) VALUES
('Elena', '655 240 340', 'Luna, 12.'),
('Juan', '955 411 120', 'Flor, 34. 1-A.'),
('Alicia', '640 450 230', 'Plaza Mayor, 26.'),
('Javier', '987 675 432', 'Las Torres, 5. 13ºC.');

INSERT INTO movies (title, year) VALUES
('Alguien voló sobre el nido del cuco', 1975),
('La Isla Mínima', 2014),
('La Furia del Dragón', 1974),
('El padrino', 1972),
('El furor del dragón', 1972),
('Marte', 2015);

INSERT INTO rentals (user_id, movie_id, exitday, returnday) VALUES
(1, 6, '2016-12-01 17:17:00', NULL),
(1, 5, '2016-11-07 17:17:00', '2016-11-10 19:20:00'),
(2, 5, '2016-11-11 14:11:00', NULL),
(3, 1, '2016-10-05 20:01:00', '2016-10-08 20:11:00'),
(4, 4, '2016-12-08 17:17:00', NULL),
(4, 2, '2016-10-21 15:17:00', '2016-10-25 15:00:00'),
(3, 3, '2016-12-11 19:27:00', NULL);
