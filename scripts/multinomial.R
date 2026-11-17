# Regressão logística multinomial: exemplo completo ----
# Pacote: nnet. Instale uma única vez: install.packages("nnet")
library(nnet)

# 1. Dados: iris (150 flores de 3 espécies); preditor = comprimento da sépala
dados <- iris[, c("Sepal.Length", "Species")]
aggregate(Sepal.Length ~ Species, data = dados, FUN = function(v) c(n = length(v), media = mean(v), dp = sd(v)))

# 2. Ajuste (a primeira categoria do fator, setosa, é a referência)
modelo <- multinom(Species ~ Sepal.Length, data = dados)
summary(modelo)

# 3. Testes de Wald (o multinom não imprime os valores-p)
z <- summary(modelo)$coefficients / summary(modelo)$standard.errors
round(2 * pnorm(abs(z), lower.tail = FALSE), 5)

# 4. Razões de chances e intervalos de confiança de 95%
exp(coef(modelo))
exp(confint(modelo))

# 5. Teste da razão de verossimilhanças contra o modelo nulo
nulo <- multinom(Species ~ 1, data = dados)
anova(nulo, modelo)

# 6. Probabilidades estimadas
novo <- data.frame(Sepal.Length = c(4.5, 5.0, 5.5, 6.0, 6.5, 7.5))
cbind(novo, round(predict(modelo, novo, type = "probs"), 3))

# 7. Classificação nos próprios dados (medida otimista)
tabela <- table(observado = dados$Species, previsto = predict(modelo))
tabela; sum(diag(tabela)) / sum(tabela)

# 8. Separação perfeita: com o comprimento da pétala as estimativas divergem
# multinom(Species ~ Petal.Length, data = iris)   # observe a magnitude dos coeficientes

# 9. Gráfico: probabilidades estimadas
grade <- data.frame(Sepal.Length = seq(4, 8.2, length.out = 200))
P <- predict(modelo, grade, type = "probs")
matplot(grade$Sepal.Length, P, type = "l", lty = 1, lwd = 2,
        xlab = "Comprimento da sépala (cm)", ylab = "Probabilidade estimada")
legend("left", legend = levels(iris$Species), col = 1:3, lty = 1, lwd = 2, bty = "n")
