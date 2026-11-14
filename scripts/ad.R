# Anderson–Darling para normalidade: exemplo completo ----
# Pacote: nortest. Instale uma única vez: install.packages("nortest")
library(nortest)

# 1. Dados
x <- mtcars$disp                    # cilindrada (pol³)
n <- length(x)

# 2. Diagnóstico gráfico
par(mfrow = c(1, 2))
hist(x, breaks = "FD", freq = FALSE, col = "gray85", border = "white",
     main = "Histograma", xlab = "Cilindrada (pol³)")
curve(dnorm(x, mean(x), sd(x)), add = TRUE, col = "red", lwd = 2)
qqnorm(x, pch = 19); qqline(x, col = "red", lwd = 2)
par(mfrow = c(1, 1))

# 3. Teste de Anderson–Darling
resultado <- ad.test(x)
print(resultado)

# 4. Cálculo manual de A^2 e da correção A* (para conferência)
xs <- sort(x); p <- pnorm((xs - mean(xs)) / sd(xs)); i <- 1:n
A2    <- -n - mean((2 * i - 1) * (log(p) + log(1 - rev(p))))
Astar <- A2 * (1 + 0.75 / n + 2.25 / n^2)
c(A2 = A2, A_corrigida = Astar)

# 5. Comparação com outros testes e com os dados transformados
shapiro.test(x)
ad.test(log(x))
ad.test(mtcars$mpg)
