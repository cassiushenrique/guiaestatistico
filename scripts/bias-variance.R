# Compromisso viés-variância: exemplo completo ----
# Pacotes: glmnet (ridge), FNN ou class (k vizinhos); o resto é R base
# O experimento exige um processo CONHECIDO, para que o viés possa ser
# calculado. Por isso os dados são simulados.

verdade <- function(x) sin(2*x)      # a função que gera y
SIG  <- 0.40                          # desvio-padrão do erro
grade <- seq(0, 3, length.out = 200)  # onde tudo é avaliado
set.seed(1)

# Base polinomial ORTOGONAL (Legendre reescalado): evita o problema
# numérico da base de potências, que é mal condicionada em grau alto.
base <- function(x, g) {
  z <- 2*x/3 - 1                      # leva [0, 3] em [-1, 1]
  B <- matrix(1, length(x), g + 1)
  if (g >= 1) B[, 2] <- z
  for (j in 2:max(g, 2)) if (j <= g)
    B[, j+1] <- ((2*j - 1)*z*B[, j] - (j - 1)*B[, j-1])/j
  B
}

# ---- 1. A decomposição, verificada numericamente
experimento <- function(g, reps = 2000, n = 50) {
  P <- matrix(NA, reps, length(grade))
  for (r in 1:reps) {
    xs <- runif(n, 0, 3); ys <- verdade(xs) + rnorm(n, 0, SIG)
    co <- qr.solve(base(xs, g), ys)
    P[r, ] <- base(grade, g) %*% co
  }
  media <- colMeans(P)
  vies2 <- mean((media - verdade(grade))^2)
  varia <- mean(apply(P, 2, var))
  eqm   <- mean((P - matrix(verdade(grade), reps, length(grade),
                            byrow = TRUE))^2) + SIG^2
  c(vies2 = vies2, variancia = varia, sigma2 = SIG^2,
    soma = vies2 + varia + SIG^2, eqm_simulado = eqm)
}
t(sapply(0:9, experimento))
# A coluna 'soma' e a coluna 'eqm_simulado' coincidem até o erro de
# Monte Carlo: a decomposição é uma IDENTIDADE, não aproximação.

# ---- 2. A complexidade ótima depende de n
for (n in c(15, 25, 50, 100, 400)) {
  tot <- sapply(1:9, function(g) experimento(g, 600, n)["soma"])
  cat(sprintf("n = %3d: melhor grau = %d\n", n, which.min(tot)))
}

# ---- 3. A mesma curva em k vizinhos mais próximos
library(FNN)
exp_knn <- function(k, reps = 1200, n = 80) {
  P <- matrix(NA, reps, length(grade))
  for (r in 1:reps) {
    xs <- runif(n, 0, 3); ys <- verdade(xs) + rnorm(n, 0, SIG)
    P[r, ] <- knn.reg(train = matrix(xs), test = matrix(grade),
                      y = ys, k = k)$pred
  }
  media <- colMeans(P)
  c(vies2 = mean((media - verdade(grade))^2),
    variancia = mean(apply(P, 2, var)))
}
t(sapply(c(1, 2, 3, 5, 10, 20, 40, 80), exp_knn))
# ATENÇÃO: aqui a flexibilidade cresce quando k DIMINUI.

# ---- 4. O erro de treino NÃO tem curva em U
set.seed(3)
xs <- runif(25, 0, 3); ys <- verdade(xs) + rnorm(25, 0, SIG)
for (g in c(0, 1, 2, 3, 5, 9, 15, 20, 24)) {
  co <- qr.solve(base(xs, g), ys)
  treino <- mean((ys - base(xs, g) %*% co)^2)
  predic <- mean((verdade(grade) - base(grade, g) %*% co)^2) + SIG^2
  cat(sprintf("grau %2d: treino %.4f  predição %.4f\n", g, treino, predic))
}
# Com grau 24 e 25 pontos, o erro de treino é ZERO e o de predição
# é o maior de toda a tabela.

# ---- 5. Viés NÃO é sempre ruim: a regressão ridge
library(glmnet)
exp_ridge <- function(lam, reps = 800, n = 25, g = 9) {
  P <- matrix(NA, reps, length(grade))
  for (r in 1:reps) {
    xs <- runif(n, 0, 3); ys <- verdade(xs) + rnorm(n, 0, SIG)
    X <- base(xs, g)[, -1]            # sem o intercepto
    m <- glmnet(X, ys, alpha = 0, lambda = lam, standardize = FALSE)
    P[r, ] <- predict(m, newx = base(grade, g)[, -1])
  }
  media <- colMeans(P)
  c(vies2 = mean((media - verdade(grade))^2),
    variancia = mean(apply(P, 2, var)))
}
t(sapply(c(0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10), exp_ridge))
# O lambda que minimiza o erro total NÃO é zero: aceitar um pouco de
# viés em troca de muito menos variância é o princípio da regularização.

# ---- 6. A decomposição é PONTUAL: o perfil muda ao longo do domínio
P <- matrix(NA, 2000, length(grade))
for (r in 1:2000) {
  xs <- runif(50, 0, 3); ys <- verdade(xs) + rnorm(50, 0, SIG)
  P[r, ] <- base(grade, 5) %*% qr.solve(base(xs, 5), ys)
}
v2 <- (colMeans(P) - verdade(grade))^2; va <- apply(P, 2, var)
borda <- grade < 0.6 | grade > 2.4
c(vies2_borda = mean(v2[borda]),  var_borda = mean(va[borda]))
c(vies2_miolo = mean(v2[!borda]), var_miolo = mean(va[!borda]))
# A variância é bem maior nas bordas: é por isso que extrapolar é
# perigoso mesmo com o modelo certo.
