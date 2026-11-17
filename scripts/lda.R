# Análise discriminante linear: exemplo completo ----
# Pacotes: MASS (lda), klaR (fronteiras), caret (validação), ggplot2
# install.packages(c("MASS", "klaR", "caret"))

data(iris)
str(iris)
table(iris$Species)

# Médias por grupo e covariância agrupada
aggregate(. ~ Species, data = iris, FUN = mean)
by(iris[, 1:4], iris$Species, function(x) round(cov(x), 4))

# O ajuste
library(MASS)
m <- lda(Species ~ ., data = iris)
m                      # priores, médias por grupo e coeficientes LD1 e LD2
m$svd^2 / sum(m$svd^2) # proporção do poder discriminante de cada eixo
m$svd^2 / (nrow(iris) - nlevels(iris$Species))   # os autovalores de W^-1 B

# Escores e classificação
pr <- predict(m)
head(pr$x)             # escores nos eixos canônicos
head(round(pr$posterior, 4))   # probabilidades a posteriori
table(Real = iris$Species, Predita = pr$class)          # RECLASSIFICAÇÃO
mean(pr$class == iris$Species)

# Validação cruzada deixa-um-de-fora: é esta a taxa a reportar
mcv <- lda(Species ~ ., data = iris, CV = TRUE)
table(Real = iris$Species, Predita = mcv$class)
mean(mcv$class == iris$Species)

# As flores mal classificadas
erradas <- which(mcv$class != iris$Species)
cbind(iris[erradas, ], predita = mcv$class[erradas],
      round(mcv$posterior[erradas, ], 4))

# Probabilidades a priori: o padrão são as proporções da amostra
lda(Species ~ ., data = iris, prior = c(0.10, 0.10, 0.80))
pr2 <- predict(lda(Species ~ ., data = iris, prior = c(0.10, 0.10, 0.80)))
table(Real = iris$Species, Predita = pr2$class)

# Pressupostos
library(biotools)
boxM(iris[, 1:4], iris$Species)      # igualdade das covariâncias
library(MVN)
# mvn(iris[iris$Species == "setosa", 1:4])   # normalidade multivariada por grupo
# Relação com a MANOVA: o mesmo W^-1 B
summary(manova(cbind(Sepal.Length, Sepal.Width, Petal.Length, Petal.Width)
               ~ Species, data = iris), test = "Hotelling-Lawley")

# Validação mais honesta ainda: separar treino e teste
library(caret)
set.seed(1)
idx <- createDataPartition(iris$Species, p = 0.7, list = FALSE)
mt  <- lda(Species ~ ., data = iris[idx, ])
confusionMatrix(predict(mt, iris[-idx, ])$class, iris[-idx, ]$Species)

# Gráficos
plot(m, col = as.integer(iris$Species))
library(klaR)
partimat(Species ~ ., data = iris, method = "lda")   # fronteiras aos pares
