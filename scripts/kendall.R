# Correlação tau de Kendall: exemplo completo ----
# Pacote: apenas o stats (já vem com o R). Dados: mtcars.

# 1. Dados: potência (hp) e consumo (mpg) de 32 automóveis
x <- mtcars$hp
y <- mtcars$mpg
n <- length(x)

# 2. Contagem dos pares concordantes, discordantes e empatados
pares <- combn(n, 2)
dx <- sign(x[pares[1, ]] - x[pares[2, ]])
dy <- sign(y[pares[1, ]] - y[pares[2, ]])
C  <- sum(dx * dy > 0); D <- sum(dx * dy < 0)
Tx <- sum(dx == 0 & dy != 0); Ty <- sum(dy == 0 & dx != 0)
c(total = ncol(pares), concordantes = C, discordantes = D, empate_x = Tx, empate_y = Ty)

# 3. tau-a (ignora empates) e tau-b (corrige empates)
c(tau_a = (C - D) / ncol(pares),
  tau_b = (C - D) / sqrt((C + D + Tx) * (C + D + Ty)))

# 4. Teste de hipóteses (o R calcula o tau-b)
cor.test(x, y, method = "kendall", exact = FALSE)

# 5. Comparação com Spearman e Pearson
c(spearman = cor(x, y, method = "spearman"), pearson = cor(x, y))

# 6. Intervalo de confiança de 95% por bootstrap (percentil)
set.seed(2026)
boot <- replicate(5000, { i <- sample(n, replace = TRUE)
                          cor(x[i], y[i], method = "kendall") })
quantile(boot, c(0.025, 0.975))

# 7. Gráfico de dispersão
plot(x, y, pch = 19, xlab = "Potência (hp)", ylab = "Consumo (mpg)")
