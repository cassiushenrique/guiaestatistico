# Bartlett: exemplo completo ----
# Pacote: apenas o stats (já vem com o R).

# 1. Dados
dados <- mtcars
dados$cil <- factor(dados$cyl)
tapply(dados$mpg, dados$cil, var)

# 2. Normalidade dentro de cada grupo (condição do teste)
tapply(dados$mpg, dados$cil, function(v) shapiro.test(v)$p.value)

# 3. Teste de Bartlett
bartlett.test(mpg ~ cil, data = dados)

# 4. Cálculo manual da estatística
ni <- table(dados$cil); vi <- tapply(dados$mpg, dados$cil, var)
N <- sum(ni); k <- length(ni)
sp2 <- sum((ni - 1) * vi) / (N - k)                       # variância combinada
num <- (N - k) * log(sp2) - sum((ni - 1) * log(vi))
den <- 1 + (1 / (3 * (k - 1))) * (sum(1 / (ni - 1)) - 1 / (N - k))
K2 <- num / den
c(sp2 = sp2, K2 = K2, valor_p = pchisq(K2, df = k - 1, lower.tail = FALSE))

# 5. Gráfico das variâncias
barplot(vi, col = "gray85", ylab = "Variância", xlab = "Cilindros")
abline(h = sp2, col = "red", lwd = 2, lty = 2)
