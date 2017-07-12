library(DT)
library(plotly)
library(leaflet)
library(dplyr)
library(ggmap)

creditos = read.csv("./data/credit.csv")
names(creditos) = c("X","Ingresos", "Limite", "Tasa","NumTarjeta","Edad", "Eduacion", "Sexo", "Estudiante", "Casado" , "Raza", "Balance")

var_categ = c(8:11)
var_num = c(2,3,5,6,7)
