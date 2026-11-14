# Correlação de Pearson: exemplo completo ----
# Pacote: apenas o stats (já vem com o R). Dados: mtcars.

# 1. Dados: peso (wt, em 1000 libras) e consumo (mpg) de 32 automóveis
x <- mtcars$wt
y <- mtcars$mpg
n <- length(x)

# 2. Gráfico de dispersão (sempre antes do coeficiente)
plot(x, y, pch = 19, xlab = "Peso (1000 lb)", ylab = "Consumo (mpg)")
abline(lm(y ~ x), col = "red", lwd = 2)

# 3. Correlação de Pearson, teste de hipóteses e intervalo de confiança de 95%
resultado <- cor.test(x, y, method = "pearson", conf.level = 0.95)
print(resultado)

# 4. Cálculo pela definição
Sxy <- sum((x - mean(x)) * (y - mean(y)))
Sxx <- sum((x - mean(x))^2); Syy <- sum((y - mean(y))^2)
r <- Sxy / sqrt(Sxx * Syy)
t <- r * sqrt((n - 2) / (1 - r^2))
c(r = r, t = t, r2 = r^2)

# 5. Intervalo de confiança pela transformação z de Fisher
z <- atanh(r); ep <- 1 / sqrt(n - 3)
tanh(z + c(-1, 1) * qnorm(0.975) * ep)

# 6. Normalidade das duas variáveis (para o teste e o IC)
shapiro.test(x)
shapiro.test(y)
