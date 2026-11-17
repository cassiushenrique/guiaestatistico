# Regressão polinomial: exemplo completo ----
# Pacotes: nenhum (poly e lm são do R base)
# Dados fictícios: 260 medições de trocas gasosas em folhas de cafeeiro.
# O arquivo fotossintese.csv acompanha o site, na pasta scripts/.
fot <- read.csv("fotossintese.csv")
str(fot)            # A (umol CO2 m-2 s-1), par, temp, vpd
summary(fot)
plot(A ~ par, data = fot, pch = 16, col = "#8aa59680",
     xlab = "PAR (umol m-2 s-1)", ylab = "A (umol CO2 m-2 s-1)")

# O ponto de partida: quanto o modelo LINEAR deixa na mesa
summary(lm(A ~ par + temp + vpd, data = fot))$r.squared   # 0,6747
# O processo gerador verdadeiro daria R² = 0,9514. É esse buraco que o
# grupo inteiro de fichas tenta preencher.

# ---- 1. BASE CRUA contra BASE ORTOGONAL
# As duas geram o MESMO espaço de funções. Em aritmética exata os ajustes
# seriam idênticos; em aritmética de máquina, não.
for (g in c(1, 2, 3, 4, 6, 8, 10, 12)) {
  Xc <- outer(fot$par, 0:g, "^")              # base crua: 1, x, x^2, ...
  Xo <- cbind(1, poly(fot$par, g))            # base ortogonal
  ac <- fitted(lm.fit(Xc, fot$A)); ao <- fitted(lm.fit(Xo, fot$A))
  cat(sprintf("grau %2d: cond(crua) %.2e | maior |dif| nos ajustes %.2e\n",
              g, kappa(Xc, exact = TRUE), max(abs(ac - ao))))
}
# Até o grau 4 as duas concordam até a 10a casa. No grau 8 a base crua
# DEIXA DE FUNCIONAR: os ajustes diferem em 13 umol.

# ---- 2. Os valores-p da base ortogonal SÃO os testes sequenciais
m8 <- lm(A ~ poly(par, 8), data = fot)
summary(m8)
# Cada coeficiente da base ortogonal é independente dos demais, de modo
# que o teste do ultimo termo e o teste sequencial daquele grau.
anova(m8)          # compare com a coluna de valores-p acima

# Na base CRUA os valores-p nao servem para escolher o grau:
summary(lm(A ~ par + I(par^2) + I(par^3) + I(par^4), data = fot))

# ---- 3. A ESCOLHA DO GRAU
library(boot)
set.seed(1)
for (g in 1:12) {
  m <- glm(A ~ poly(par, g), data = fot)
  cv <- cv.glm(fot, m, K = 10)$delta[1]
  cat(sprintf("grau %2d: EQM resub %.4f | EQM 10-fold %.4f | AIC %.2f | BIC %.2f\n",
      g, mean(residuals(m)^2), cv, AIC(m), BIC(m)))
}
# Testes F sequenciais, do grau 1 ao 10:
anova(lm(A ~ poly(par, 10), data = fot))

# ---- 4. A VARIÂNCIA POR REGIÃO: o leque das bordas
set.seed(7)
gx <- seq(min(fot$par), max(fot$par), length.out = 200)
P <- replicate(500, {
  i <- sample(nrow(fot), replace = TRUE)
  predict(lm(A ~ poly(par, 5), data = fot[i, ]), newdata = data.frame(par = gx)) })
va <- apply(P, 1, var)
faixas <- cut(gx, c(0, 200, 500, 900, 1400, 2100))
tapply(va, faixas, mean)
# A variância nas bordas é cerca de 6 vezes a do miolo: é o mesmo
# fenômeno da ficha bias-variance, agora em dados de campo.

# ---- 5. EXTRAPOLAR
nd <- data.frame(par = c(2100, 2200, 2400, 2600, 3000))
for (g in c(2, 4, 6, 8))
  print(round(predict(lm(A ~ poly(par, g), data = fot), newdata = nd), 2))
# A 3.000 umol o grau 8 preve 1.024 umol de fotossintese. Nao extrapole.

# ---- 6. O POLINÔMIO É GLOBAL: uma perturbação local contamina tudo
set.seed(3)
extra <- data.frame(par = runif(25, 1800, 2000))
extra$A <- 26*(1 - exp(-0.0024*extra$par)) -
           1.9e-6*pmax(extra$par - 1000, 0)^2 + 4.0 + rnorm(25, 0, 1.3)
extra$A <- extra$A - mean(26*(1 - exp(-0.0024*fot$par)) -
           1.9e-6*pmax(fot$par - 1000, 0)^2) + mean(fot$A)
fot2 <- rbind(fot[, c("A","par")], extra)
library(splines)
for (p in c(5, 8, 11, 14)) {
  g <- p - 1; K <- p - 4
  nos <- quantile(fot$par, probs = seq(0, 1, length.out = K + 2)[-c(1, K+2)])
  a1 <- predict(lm(A ~ poly(par, g), data = fot),  newdata = data.frame(par = 150))
  a2 <- predict(lm(A ~ poly(par, g), data = fot2), newdata = data.frame(par = 150))
  s1 <- predict(lm(A ~ bs(par, knots = nos), data = fot),  newdata = data.frame(par = 150))
  s2 <- predict(lm(A ~ bs(par, knots = nos), data = fot2), newdata = data.frame(par = 150))
  cat(sprintf("p = %2d: polinomio move %.4f | spline move %.4f\n",
              p, abs(a2 - a1), abs(s2 - s1)))
}
