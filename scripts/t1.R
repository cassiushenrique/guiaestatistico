# Teste t para uma amostra: exemplo completo ----
# Pacote: apenas o stats (já vem com o R).

# 1. Dados: peso líquido de 25 pacotes (g). Valor nominal: 500 g (dados fictícios)
peso <- c(492.2, 493.3, 500.6, 490.3, 499.8, 499.2, 501.3, 496.2, 500.0, 492.3,
          501.6, 500.2, 496.3, 497.2, 493.8, 500.1, 494.6, 504.2, 494.4, 496.0,
          501.6, 496.6, 500.2, 507.5, 496.5)
mu0 <- 500
c(n = length(peso), media = mean(peso), dp = sd(peso), mediana = median(peso))

# 2. Diagnóstico gráfico, antes do teste
par(mfrow = c(1, 2))
hist(peso, col = "gray85", xlab = "Peso (g)", main = ""); abline(v = mu0, col = "red", lwd = 2)
qqnorm(peso, pch = 19); qqline(peso, col = "red", lwd = 2)
par(mfrow = c(1, 1))

# 3. Pressuposto de normalidade
shapiro.test(peso)

# 4. Teste t bilateral
t.test(peso, mu = mu0)

# 5. Cálculo manual
n <- length(peso); ep <- sd(peso) / sqrt(n)
t <- (mean(peso) - mu0) / ep
c(erro_padrao = ep, t = t, gl = n - 1, valor_p = 2 * pt(abs(t), n - 1, lower.tail = FALSE))

# 6. Versão unilateral: o interesse é saber se o peso está ABAIXO do nominal
t.test(peso, mu = mu0, alternative = "less")

# 7. Tamanho do efeito e poder
d <- (mean(peso) - mu0) / sd(peso)
d
# install.packages("pwr")
# pwr::pwr.t.test(n = n, d = d, sig.level = 0.05, type = "one.sample")

# 8. Alternativas não paramétricas
wilcox.test(peso, mu = mu0)
binom.test(sum(peso < mu0), sum(peso != mu0))      # teste do sinal
