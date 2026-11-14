# Correlação parcial: exemplo completo ----
# Pacote: apenas o stats (já vem com o R). Dados: mtcars.

# 1. Dados: consumo (mpg), potência (hp) e peso (wt)
x <- mtcars$hp; y <- mtcars$mpg; z <- mtcars$wt
n <- length(x)

# 2. Correlações simples (de ordem zero)
cor(cbind(mpg = y, hp = x, wt = z))

# 3. Correlação parcial entre hp e mpg, controlando wt (fórmula)
r_xy <- cor(x, y); r_xz <- cor(x, z); r_yz <- cor(y, z)
r_p  <- (r_xy - r_xz * r_yz) / sqrt((1 - r_xz^2) * (1 - r_yz^2))
k    <- 1                                   # número de variáveis controladas
gl   <- n - 2 - k
t    <- r_p * sqrt(gl / (1 - r_p^2))
c(r_parcial = r_p, t = t, gl = gl, valor_p = 2 * pt(abs(t), gl, lower.tail = FALSE))

# 4. O mesmo resultado pela correlação entre resíduos
e_x <- residuals(lm(x ~ z))
e_y <- residuals(lm(y ~ z))
cor(e_x, e_y)

# 5. Intervalo de confiança de 95% (z de Fisher com n - 3 - k)
zf <- atanh(r_p); ep <- 1 / sqrt(n - 3 - k)
tanh(zf + c(-1, 1) * qnorm(0.975) * ep)

# 6. Matriz de correlações parciais (cada par controlando as demais variáveis)
P <- solve(cor(cbind(mpg = y, hp = x, wt = z)))
parciais <- -cov2cor(P); diag(parciais) <- 1
round(parciais, 4)

# 7. Função pronta no pacote ppcor: install.packages("ppcor")
# ppcor::pcor.test(x, y, z)

# 8. Gráficos
par(mfrow = c(1, 2))
plot(x, y, pch = 19, xlab = "Potência (hp)", ylab = "Consumo (mpg)")
plot(e_x, e_y, pch = 19, col = "darkgreen", xlab = "Resíduo de hp | wt", ylab = "Resíduo de mpg | wt")
abline(lm(e_y ~ e_x), col = "red", lwd = 2)
par(mfrow = c(1, 1))
