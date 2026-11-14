# M de Box: exemplo completo ----
# Pacote: biotools. Instale uma única vez: install.packages("biotools")
library(biotools)

# 1. Dados: iris (150 flores, 3 espécies, 4 medidas em cm)
X <- iris[, 1:4]
grupo <- iris$Species
table(grupo)

# 2. Logaritmo do determinante da matriz de covariâncias de cada espécie
sapply(split(X, grupo), function(d) log(det(cov(d))))

# 3. Teste M de Box
boxM(X, grupo)

# 4. Cálculo manual da estatística
g <- nlevels(grupo); p <- ncol(X)
ni <- as.numeric(table(grupo)); N <- sum(ni)
Si <- lapply(split(X, grupo), cov)
Sp <- Reduce(`+`, Map(function(S, n) (n - 1) * S, Si, ni)) / (N - g)   # covariância combinada
M  <- (N - g) * log(det(Sp)) - sum((ni - 1) * sapply(Si, function(S) log(det(S))))
c1 <- (2 * p^2 + 3 * p - 1) / (6 * (p + 1) * (g - 1)) * (sum(1 / (ni - 1)) - 1 / (N - g))
X2 <- M * (1 - c1); gl <- p * (p + 1) * (g - 1) / 2
c(M = M, X2 = X2, gl = gl, valor_p = pchisq(X2, gl, lower.tail = FALSE))

# 5. Gráfico: dispersão das medidas da pétala por espécie
plot(iris$Petal.Length, iris$Petal.Width, col = grupo, pch = 19,
     xlab = "Comprimento da pétala (cm)", ylab = "Largura da pétala (cm)")
legend("topleft", legend = levels(grupo), col = 1:3, pch = 19, bty = "n")
