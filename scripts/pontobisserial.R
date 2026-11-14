# Correlação ponto-bisserial: exemplo completo ----
# Pacote: apenas o stats (já vem com o R). Dados: mtcars.

# 1. Dados: câmbio (am: 0 = automático, 1 = manual) e consumo (mpg)
grupo <- mtcars$am
y <- mtcars$mpg
n <- length(y)

# 2. Descritivas por grupo
aggregate(y ~ grupo, FUN = function(v) c(n = length(v), media = mean(v), dp = sd(v)))

# 3. Gráfico
boxplot(y ~ grupo, names = c("Automático (0)", "Manual (1)"),
        col = "gray85", xlab = "", ylab = "Consumo (mpg)")

# 4. Correlação ponto-bisserial: é o Pearson com a variável binária codificada em 0/1
resultado <- cor.test(grupo, y)
print(resultado)

# 5. Cálculo pela fórmula
m1 <- mean(y[grupo == 1]); m0 <- mean(y[grupo == 0])
n1 <- sum(grupo == 1);     n0 <- sum(grupo == 0)
s_n <- sqrt(mean((y - mean(y))^2))            # desvio-padrão com divisor n
(m1 - m0) / s_n * sqrt(n1 * n0 / n^2)

# 6. Equivalência com o teste t de Student (variâncias iguais)
t.test(y ~ grupo, var.equal = TRUE)

# 7. Pressupostos: normalidade em cada grupo e teste t de Welch
tapply(y, grupo, function(v) shapiro.test(v)$p.value)
t.test(y ~ grupo)                             # Welch (variâncias desiguais)
