# Kolmogorov–Smirnov (uma amostra): exemplo completo ----
# Pacotes: stats (já instalado) e nortest (para a versão de Lilliefors)
# install.packages("nortest")   # execute uma única vez
library(nortest)

# 1. Dados: consumo de 32 automóveis (milhas por galão)
x <- mtcars$mpg
summary(x); length(x)

# 2. ECDF e CDF teórica (H0 totalmente especificada: N(20, 6^2))
plot(ecdf(x), main = "ECDF do consumo versus N(20, 6^2)", xlab = "mpg", col = "darkgreen")
curve(pnorm(x, mean = 20, sd = 6), add = TRUE, col = "red", lwd = 2)

# 3. Teste de KS com parâmetros FIXADOS a priori
ks.test(x, "pnorm", mean = 20, sd = 6, exact = FALSE)

# 4. Quando os parâmetros vêm da amostra: use Lilliefors, nunca o KS clássico
lillie.test(x)                  # consumo (mpg)
lillie.test(mtcars$disp)        # cilindrada (disp)

# 5. Distância D calculada manualmente
xs <- sort(x); n <- length(xs); Fx <- pnorm(xs, 20, 6)
Dmais  <- max((1:n)/n - Fx)
Dmenos <- max(Fx - (0:(n - 1))/n)
c(Dmais = Dmais, Dmenos = Dmenos, D = max(Dmais, Dmenos))
