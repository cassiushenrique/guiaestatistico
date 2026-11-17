# Regressão local (LOESS): exemplo completo ----
# Pacotes: nenhum (loess e lowess sao do R base)
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

# ---- 1. O ajuste, e o que o span faz
m <- loess(A ~ par, data = fot, span = 0.35, degree = 1)
summary(m)          # traz "equivalent number of parameters": os g.l. efetivos

plot(A ~ par, data = fot, pch = 16, col = "#8aa59680")
gx <- seq(min(fot$par), max(fot$par), length.out = 300)
for (sp in c(0.12, 0.35, 0.90)) {
  mm <- loess(A ~ par, data = fot, span = sp, degree = 1)
  lines(gx, predict(mm, newdata = data.frame(par = gx)), lwd = 2)
}

# ---- 2. A ESCOLHA DO SPAN por validação cruzada
# O loess nao tem cv.glm pronto; faz-se a mao.
set.seed(1)
for (sp in c(0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.7, 0.9, 1.2)) {
  d <- sample(rep(1:10, length.out = nrow(fot))); er <- numeric(nrow(fot))
  for (j in 1:10) {
    mm <- loess(A ~ par, data = fot[d != j, ], span = sp, degree = 1,
                control = loess.control(surface = "direct"))
    er[d == j] <- fot$A[d == j] - predict(mm, newdata = fot[d == j, ]) }
  gl <- loess(A ~ par, data = fot, span = sp, degree = 1)$trace.hat
  cat(sprintf("span %.2f: %3d vizinhos, %5.2f g.l., EQM 10-fold %.4f\n",
              sp, ceiling(sp*nrow(fot)), gl, mean(er^2)))
}

# ---- 3. GRAU 1 contra GRAU 2
for (sp in c(0.15, 0.25, 0.35, 0.5, 0.7)) {
  e <- sapply(1:2, function(g) {
    d <- sample(rep(1:10, length.out = nrow(fot))); er <- numeric(nrow(fot))
    for (j in 1:10) {
      mm <- loess(A ~ par, data = fot[d != j, ], span = sp, degree = g,
                  control = loess.control(surface = "direct"))
      er[d == j] <- fot$A[d == j] - predict(mm, newdata = fot[d == j, ]) }
    mean(er^2) })
  cat(sprintf("span %.2f: grau 1 %.4f | grau 2 %.4f\n", sp, e[1], e[2]))
}

# ---- 4. A VERSÃO ROBUSTA
# family = "symmetric" liga a reponderacao bisquare de Cleveland.
set.seed(19)
fot2 <- fot
idx <- sample(which(fot$par > 400 & fot$par < 1200), 10)
fot2$A[idx] <- fot2$A[idx] + 18          # 10 pontos deslocados

m_g <- loess(A ~ par, data = fot2, span = .35, degree = 1, family = "gaussian")
m_s <- loess(A ~ par, data = fot2, span = .35, degree = 1, family = "symmetric")
plot(A ~ par, data = fot2, pch = 16, col = "#8aa59680")
points(fot2$par[idx], fot2$A[idx], pch = 16, col = "#b5442f")
lines(gx, predict(m_g, data.frame(par = gx)), col = "#bf924c", lwd = 3)
lines(gx, predict(m_s, data.frame(par = gx)), col = "#1f5b43", lwd = 3)
# O ajuste comum e arrastado; o robusto praticamente nao se move.

# ---- 5. O QUE O LOESS NÃO TEM
# (a) formula: nao ha equacao para reportar, so a curva.
# (b) extrapolacao: fora do dominio o R devolve NA por padrao.
predict(m, newdata = data.frame(par = c(1990, 2100, 2400)))
# (c) custo: uma regressao por ponto de predicao. O padrao
#     surface = "interpolate" avalia o ajuste exato em poucos pontos
#     e interpola; surface = "direct" faz a conta em todos.
system.time(loess(A ~ par, data = fot, span = .35,
                  control = loess.control(surface = "direct")))
# (d) nao ha valor-p padronizado. Ha um teste F aproximado entre
#     ajustes aninhados, que supoe o span FIXADO antes de ver os dados:
anova(loess(A ~ par, data = fot, span = .7, degree = 1),
      loess(A ~ par, data = fot, span = .3, degree = 1))

# ---- 6. lowess: a versão mais antiga e mais simples
# Serve para o grafico exploratorio e nada mais: nao aceita newdata,
# nao faz predicao, nao tem erro-padrao.
plot(A ~ par, data = fot, pch = 16, col = "#8aa59680")
lines(lowess(fot$par, fot$A, f = 0.35), col = "#b5442f", lwd = 3)
