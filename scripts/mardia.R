# Mardia (normalidade multivariada): exemplo completo ----
# O cálculo abaixo usa apenas o R base. O pacote MVN oferece o teste pronto (item 5).

# 1. Dados: 50 flores da espécie setosa, 4 medidas (cm)
X <- as.matrix(iris[iris$Species == "setosa", 1:4])
n <- nrow(X); p <- ncol(X)
colMeans(X)

# 2. Assimetria e curtose multivariadas de Mardia
Xc <- scale(X, center = TRUE, scale = FALSE)
S  <- crossprod(Xc) / n                      # covariância com divisor n
D  <- Xc %*% solve(S) %*% t(Xc)              # produtos de Mahalanobis
b1 <- sum(D^3) / n^2                         # assimetria
b2 <- sum(diag(D)^2) / n                     # curtose

# 3. Estatísticas de teste e valores-p
est_assimetria <- n * b1 / 6
gl <- p * (p + 1) * (p + 2) / 6
p_assimetria <- pchisq(est_assimetria, gl, lower.tail = FALSE)
z_curtose <- (b2 - p * (p + 2)) / sqrt(8 * p * (p + 2) / n)
p_curtose <- 2 * pnorm(abs(z_curtose), lower.tail = FALSE)
c(b1 = b1, estatistica = est_assimetria, gl = gl, valor_p = p_assimetria)
c(b2 = b2, z = z_curtose, valor_p = p_curtose)

# 4. Normalidade de cada variável
apply(X, 2, function(v) shapiro.test(v)$p.value)

# 5. Teste pronto no pacote MVN (install.packages("MVN"))
# library(MVN)
# mvn(X, mvn_test = "mardia")      # versões 6 ou mais recentes
# mvn(X, mvnTest = "mardia")       # versões anteriores

# 6. Gráfico Q-Q das distâncias de Mahalanobis contra o qui-quadrado
d2 <- sort(diag(D))
q  <- qchisq(ppoints(n), df = p)
plot(q, d2, pch = 19, xlab = "Quantis do qui-quadrado", ylab = "Distância de Mahalanobis ao quadrado")
abline(0, 1, col = "red", lwd = 2)
