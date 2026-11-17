# Splines de suavização: exemplo completo ----
# Pacotes: nenhum (smooth.spline é do R base)
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

# ---- 1. OS DOIS EXTREMOS
# lambda -> 0 interpola; lambda -> infinito devolve a reta de minimos quadrados.
s0 <- smooth.spline(fot$par, fot$A, spar = -1.5)    # quase interpolacao
sI <- smooth.spline(fot$par, fot$A, spar =  2.5)    # quase reta
c(gl_minimo = sI$df, gl_maximo = s0$df, x_distintos = length(unique(fot$par)))
# compare o ajuste com spar alto a reta de minimos quadrados:
max(abs(predict(sI, fot$par)$y - fitted(lm(A ~ par, data = fot))))

# ---- 2. A ESCOLHA DA SUAVIDADE
# Tres caminhos: dar os g.l., deixar o GCV escolher, ou a validacao cruzada.
sg <- smooth.spline(fot$par, fot$A)                 # GCV, o padrao
sg$df; sg$lambda; sg$spar
sc <- smooth.spline(fot$par, fot$A, cv = TRUE)      # deixa-um-de-fora
sc$df
s8 <- smooth.spline(fot$par, fot$A, df = 8)         # g.l. fixados
s8$lambda

# a curva de g.l. contra lambda
for (sp in seq(0, 1.4, by = 0.2)) {
  m <- smooth.spline(fot$par, fot$A, spar = sp)
  cat(sprintf("spar %.1f: lambda %.3e, g.l. %.2f\n", sp, m$lambda, m$df))
}

# ---- 3. O ajuste, com os pontos
plot(A ~ par, data = fot, pch = 16, col = "#8aa59680",
     xlab = "PAR (umol m-2 s-1)", ylab = "A (umol CO2 m-2 s-1)")
lines(sg, col = "#1f5b43", lwd = 3)
lines(smooth.spline(fot$par, fot$A, df = 25), col = "#b5442f", lwd = 2)
lines(smooth.spline(fot$par, fot$A, df = 3),  col = "#bf924c", lwd = 2)

# ---- 4. A LIGAÇÃO COM O RIDGE
# O ajuste e LINEAR em y: f = S_lambda y, e os g.l. efetivos sao tr(S).
# Na base natural com nos em todos os x, S = (W + lambda K)^-1 W, com K
# fixa. Trocar K pela identidade da exatamente o ridge. Os DOIS menores
# autovalores de K sao zero: sao a constante e a reta, que a penalidade
# nao toca - e por isso os g.l. nunca descem abaixo de 2.

# ---- 5. COMPARAÇÃO COM AS DUAS FICHAS ANTERIORES
library(splines); library(boot)
set.seed(1)
for (gl in c(3, 4, 5, 6, 8, 10, 14, 20)) {
  g <- gl - 1; K <- gl - 4
  e1 <- cv.glm(fot, glm(A ~ poly(par, g), data = fot), K = 10)$delta[1]
  e2 <- if (K >= 0) {
    nos <- quantile(fot$par, probs = seq(0, 1, length.out = K + 2)[-c(1, K + 2)])
    f <- as.formula(sprintf("A ~ bs(par, knots = c(%s))", paste(round(nos, 4), collapse = ", ")))
    cv.glm(fot, glm(f, data = fot), K = 10)$delta[1] } else NA
  # validacao cruzada manual para a spline de suavizacao
  d <- sample(rep(1:10, length.out = nrow(fot))); er <- numeric(nrow(fot))
  for (j in 1:10) {
    m <- smooth.spline(fot$par[d != j], fot$A[d != j], df = gl)
    er[d == j] <- fot$A[d == j] - predict(m, fot$par[d == j])$y }
  cat(sprintf("%2d g.l.: polinomio %7.4f | spline c/ nos %7s | suavizacao %7.4f\n",
              gl, e1, ifelse(is.na(e2), "-", sprintf("%.4f", e2)), mean(er^2)))
}

# ---- 6. ERROS CORRELACIONADOS ENGANAM A VALIDAÇÃO
set.seed(5)
for (rho in c(0, 0.5, 0.8)) {
  gl <- replicate(30, {
    e <- numeric(nrow(fot))
    for (i in 2:nrow(fot)) e[i] <- rho*e[i-1] + rnorm(1, 0, 1.3*sqrt(1 - rho^2))
    yy <- 26*(1 - exp(-0.0024*sort(fot$par))) -
          1.9e-6*pmax(sort(fot$par) - 1000, 0)^2 + e
    smooth.spline(sort(fot$par), yy)$df })
  cat(sprintf("rho = %.1f: g.l. escolhidos pelo GCV = %.2f\n", rho, mean(gl)))
}
# A funcao verdadeira e a MESMA nos tres casos. So os erros mudam.
