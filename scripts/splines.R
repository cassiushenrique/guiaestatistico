# Splines de regressão: exemplo completo ----
# Pacotes: splines (bs e ns, que acompanham o R)
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

library(splines)

# ---- 1. A CONTABILIDADE DOS GRAUS DE LIBERDADE
# Uma spline de grau g com K nós internos tem K + g + 1 funções de base
# (contando o intercepto). Confira pelo posto da matriz:
for (g in c(1, 2, 3)) for (K in c(4, 8)) {
  nos <- quantile(fot$par, probs = seq(0, 1, length.out = K + 2)[-c(1, K + 2)])
  B <- cbind(1, bs(fot$par, knots = nos, degree = g))
  cat(sprintf("grau %d, K = %2d nos: K+g+1 = %2d | posto de B = %2d\n",
              g, K, K + g + 1, qr(B)$rank))
}

# ---- 2. As funções da base, e a LOCALIDADE delas
nos <- quantile(fot$par, probs = c(.2, .4, .6, .8))
B <- bs(fot$par, knots = nos, degree = 3)
matplot(sort(fot$par), B[order(fot$par), ], type = "l", lty = 1, lwd = 2,
        xlab = "PAR", ylab = "funcao de base")
# Cada funcao e nao nula em apenas g+2 = 5 intervalos consecutivos.
apply(B, 2, function(z) range(fot$par[z > 1e-12]))

# ---- 3. QUANTOS NÓS? Validação cruzada
library(boot)
set.seed(1)
for (K in 0:20) {
  f <- if (K == 0) A ~ bs(par, degree = 3) else {
    nos <- quantile(fot$par, probs = seq(0, 1, length.out = K + 2)[-c(1, K + 2)])
    as.formula(sprintf("A ~ bs(par, knots = c(%s), degree = 3)",
                       paste(round(nos, 4), collapse = ", "))) }
  m <- glm(f, data = fot)
  cat(sprintf("K = %2d (p = %2d): EQM 10-fold %.4f\n",
              K, K + 4, cv.glm(fot, m, K = 10)$delta[1]))
}

# ---- 4. ONDE PÔR OS NÓS: quantis contra espaçamento igual
for (K in c(2, 4, 6, 8, 12)) {
  nq <- quantile(fot$par, probs = seq(0, 1, length.out = K + 2)[-c(1, K + 2)])
  ne <- seq(min(fot$par), max(fot$par), length.out = K + 2)[-c(1, K + 2)]
  fq <- as.formula(sprintf("A ~ bs(par, knots = c(%s))", paste(round(nq, 4), collapse = ", ")))
  fe <- as.formula(sprintf("A ~ bs(par, knots = c(%s))", paste(round(ne, 4), collapse = ", ")))
  cat(sprintf("K = %2d: quantis %.4f | igualmente espacados %.4f\n", K,
      cv.glm(fot, glm(fq, data = fot), K = 10)$delta[1],
      cv.glm(fot, glm(fe, data = fot), K = 10)$delta[1]))
}

# ---- 5. A SPLINE NATURAL: a restrição que conserta as bordas
# ns() e linear fora dos nos de fronteira; bs() nao e.
nos6 <- quantile(fot$par, probs = seq(0, 1, length.out = 8)[-c(1, 8)])
mb <- lm(A ~ bs(par, knots = nos6), data = fot)
mn <- lm(A ~ ns(par, knots = nos6), data = fot)
mp <- lm(A ~ poly(par, 9), data = fot)
c(bs = length(coef(mb)), ns = length(coef(mn)), poly = length(coef(mp)))

# A variancia do ajuste por regiao, por bootstrap
set.seed(11)
gx <- sort(fot$par)
vari <- function(form) {
  P <- replicate(500, {
    i <- sample(nrow(fot), replace = TRUE)
    predict(lm(form, data = fot[i, ]), newdata = data.frame(par = gx)) })
  tapply(apply(P, 1, var), cut(gx, c(0, 200, 500, 1400, 1800, 2100)), mean) }
rbind(bs = vari(A ~ bs(par, knots = nos6)),
      ns = vari(A ~ ns(par, knots = nos6)),
      poly = vari(A ~ poly(par, 9)))

# ---- 6. EXTRAPOLAR
nd <- data.frame(par = c(2100, 2300, 2600, 3000))
# bs() avisa e extrapola com o polinomio cubico da ultima peca;
# ns() extrapola em LINHA RETA, por construcao.
rbind(bs = predict(mb, newdata = nd), ns = predict(mn, newdata = nd),
      poly = predict(mp, newdata = nd))

# ---- 7. SPLINE CONTRA POLINÔMIO, com a MESMA flexibilidade
for (p in c(5, 6, 8, 10, 12, 15, 20)) {
  g <- p - 1; K <- p - 4
  nos <- quantile(fot$par, probs = seq(0, 1, length.out = K + 2)[-c(1, K + 2)])
  fs <- as.formula(sprintf("A ~ bs(par, knots = c(%s))", paste(round(nos, 4), collapse = ", ")))
  cat(sprintf("p = %2d: polinomio %.4f | spline %.4f\n", p,
      cv.glm(fot, glm(A ~ poly(par, g), data = fot), K = 10)$delta[1],
      cv.glm(fot, glm(fs, data = fot), K = 10)$delta[1]))
}
