# Correlação de Spearman: exemplo completo ----
# Pacote: apenas o stats (já vem com o R). Dados: mtcars.

# 1. Dados: potência (hp) e consumo (mpg) de 32 automóveis
x <- mtcars$hp
y <- mtcars$mpg
n <- length(x)

# 2. Dispersão dos dados e dos postos
par(mfrow = c(1, 2))
plot(x, y, pch = 19, xlab = "Potência (hp)", ylab = "Consumo (mpg)")
plot(rank(x), rank(y), pch = 19, col = "darkgreen",
     xlab = "Posto da potência", ylab = "Posto do consumo")
par(mfrow = c(1, 1))

# 3. Correlação de Spearman (exact = FALSE porque há empates)
cor.test(x, y, method = "spearman", exact = FALSE)

# 4. Spearman é o Pearson calculado sobre os postos
cor(rank(x), rank(y))
sum((rank(x) - rank(y))^2)            # soma de d²

# 5. Comparação com Pearson
cor(x, y)

# 6. Intervalo de confiança de 95% por bootstrap (percentil)
set.seed(2026)
boot <- replicate(5000, { i <- sample(n, replace = TRUE)
                          cor(x[i], y[i], method = "spearman") })
quantile(boot, c(0.025, 0.975))
