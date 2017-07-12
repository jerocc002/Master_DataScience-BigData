
# This is the user-interface definition of a Shiny web application.
# You can find out more about building applications with Shiny here:
#
# http://shiny.rstudio.com
#

var_texto = "Esta aplicacion Shiny tiene objetivo la difusión de datos del sistema de bicis compartidas Sevici y los resultados de la aplicación de un modelo predictivo sobre los mismos."

var_texto_mitfm = "El título de mi TFM es: Análisis y predicción a corto plazo de disponibilidad y uso en el sistema de bicicletas compartidas de Sevilla (Sevici)."

library(shiny)

shinyUI(fluidPage(
  navbarPage("Datos Bancarios",
             ####### tabPanel: Inf
             tabPanel("Información",
                      tags$img(src="logoR.png",alig="left", width="150px"),
                      br(),br(),br(),br(),
                      hr(),
                      h3("Objetivos"),
                      var_texto,
                      h3("Autor"),
                      strong("Jerónimo Carranza Carranza"),
                      h3("Resumen de mi trabajo Fin de Master"),
                      var_texto_mitfm
             ),
             # ####### tabPanel: Datos
             tabPanel("Datos",
                      DT:: dataTableOutput("OUTdatos")
             ),
             # ####### tabPanel: Estudio unidimensional
             # tabPanel("Estudio Unidimensional",
             #          sidebarLayout(
             #            sidebarPanel(
             #              selectInput("Selvariable01Uni",
             #                          "Selecciona variable",
             #                          choices = names(creditos)[c(var_num,var_categ)],
             #                          selected = names(creditos)[var_num[1]])
             #            ),
             #            mainPanel(
             #              tabsetPanel(
             #                tabPanel(),
             #                tabPanel(),
             #                tabPanel()
             #              )
             #            )
             #          )
             # ),
             # ####### tabPanel: Estudio bidimensional
             tabPanel("Estudio bidimensional",
                      sidebarLayout(
                        sidebarPanel(
                          selectInput("Selvariable01",
                                      "Selecciona variable X:",
                                      choices = names(creditos)[c(var_num)],
                                      selected = names(creditos)[var_num[1]]),
                          selectInput("Selvariable02",
                                      "Selecciona variable Y:",
                                      choices = names(creditos)[c(var_num)],
                                      selected = names(creditos)[var_num[2]]),
                          selectInput("Selvariable03",
                                      "Selecciona variable categorica",
                                      choices = names(creditos)[c(var_categ)],
                                      selected = names(creditos)[var_categ[1]])
                        ),
                        mainPanel(
                          tabsetPanel(
                            ##############   Tab: 
                            tabPanel("Dispersion",
                                     plotly:: plotlyOutput("OUTdispersion")     
                            ),
                            ##############   Tab: 
                            tabPanel("Regresion Lineal",
                                     shiny:: verbatimTextOutput("OUTRegLineal")          
                            ),
                            ##############   Tab: 
                            tabPanel("Dispersion Categorica",
                                     plotly:: plotlyOutput("OUTdispersioncat")  
                            )
                          )
                        )
                      )
             ),
             ####### tabPanel: Mapa
             tabPanel("Mapa",
                      br(),
                      leaflet:: leafletOutput("map", height = "600px")
             )
             # ########### fin: navbarPage             
  )
))