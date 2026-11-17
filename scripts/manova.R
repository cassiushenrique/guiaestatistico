# MANOVA: exemplo completo ----
# Pacotes: stats (base), car (testes multivariados), biotools (M de Box), MASS (discriminante)
# install.packages(c("car", "biotools", "MASS", "MVN"))

# PARTE 1 - por que uma MANOVA e não várias ANOVAs
# Dados fictícios: três lotes de um polímero, com duas medidas por corpo de prova.
resistencia <- c(51.62,49.33,48.71,51.01,49.47,50.83,50.19,51.66,49.92,50.79,52.05,50.06,47.94,49.26,49.31,
                 52.30,49.77,50.40,48.27,50.21,50.20,52.54,48.56,50.72,50.08,49.83,52.58,49.98,51.20,51.70,
                 50.45,51.49,48.84,50.42,50.59,49.51,50.13,49.74,47.87,49.70,50.59,50.74,49.25,49.57,48.08)
alongamento <- c(31.10,28.81,29.43,31.04,29.77,30.33,29.88,32.14,30.19,30.52,32.30,30.22,28.15,29.17,29.59,
                 30.93,28.42,29.58,27.82,29.63,29.47,31.75,27.62,30.36,29.42,28.83,31.39,29.25,30.31,30.91,
                 30.77,31.55,30.04,31.17,31.05,29.75,30.40,29.89,28.50,30.53,32.10,31.07,29.70,30.63,28.82)
lote <- factor(rep(c("A", "B", "C"), each = 15))
polimero <- data.frame(lote, resistencia, alongamento)
aggregate(cbind(resistencia, alongamento) ~ lote, data = polimero, FUN = mean)

# As duas ANOVAs separadas
summary(aov(resistencia ~ lote, data = polimero))
summary(aov(alongamento ~ lote, data = polimero))

# A MANOVA
mod <- manova(cbind(resistencia, alongamento) ~ lote, data = polimero)
summary(mod)                              # padrão: traço de Pillai
summary(mod, test = "Wilks")
summary(mod, test = "Hotelling-Lawley")
summary(mod, test = "Roy")
summary.aov(mod)                          # as ANOVAs univariadas, de dentro do objeto

# O gráfico que explica o resultado
plot(resistencia, alongamento, col = lote, pch = 19,
     xlab = "Resistência (MPa)", ylab = "Alongamento (%)")
legend("topleft", levels(lote), col = 1:3, pch = 19)

# PARTE 2 - o exemplo canônico: iris
data(iris)
mod2 <- manova(cbind(Sepal.Length, Sepal.Width, Petal.Length, Petal.Width) ~ Species, data = iris)
summary(mod2)
summary(mod2, test = "Wilks")
summary.aov(mod2)

# Pressupostos
library(biotools)
boxM(iris[, 1:4], iris$Species)                      # igualdade das covariâncias
library(MVN)
mvn(iris[iris$Species == "setosa", 1:4])             # normalidade multivariada por grupo
# (em versões recentes do pacote a função chama-se mvn_test)

# Comparações depois da MANOVA: ANOVAs com ajuste, ou funções discriminantes
p.adjust(sapply(1:4, function(j) summary(aov(iris[, j] ~ iris$Species))[[1]][1, 5]),
         method = "bonferroni")
library(MASS)
ld <- lda(Species ~ ., data = iris)
ld                                                    # coeficientes de LD1 e LD2
plot(ld)

# Alternativa com car, que dá mais controle sobre o tipo de soma de quadrados
library(car)
m3 <- lm(cbind(Sepal.Length, Sepal.Width, Petal.Length, Petal.Width) ~ Species, data = iris)
Anova(m3, test.statistic = "Pillai")
