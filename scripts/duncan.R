# Teste de amplitudes múltiplas de Duncan: exemplo completo ----
# Pacotes: agricolae (duncan.test). Base: stats
# install.packages("agricolae")

# 1. Dados: chickwts, peso de pintinhos por tipo de ração
data(chickwts)
modelo <- aov(weight ~ feed, data = chickwts)
summary(modelo)
k <- nlevels(chickwts$feed); gl <- df.residual(modelo)
QM <- sum(residuals(modelo)^2) / gl

# 2. O nível de proteção que Duncan adota em cada etapa
alfa <- 0.05
p <- 2:k
data.frame(p,
           protecao   = (1 - alfa)^(p - 1),
           alfa_p     = 1 - (1 - alfa)^(p - 1),
           q_duncan   = qtukey((1 - alfa)^(p - 1), p, gl),
           q_tukey    = qtukey(1 - alfa, k, gl),
           AMS_duncan = qtukey((1 - alfa)^(p - 1), p, gl) * sqrt(QM / 12))

# 3. O procedimento
library(agricolae)
duncan.test(modelo, "feed", console = TRUE)

# 4. Lado a lado com os demais, no mesmo pacote
LSD.test(modelo, "feed", console = TRUE)     # sem ajuste
SNK.test(modelo, "feed", console = TRUE)     # Newman-Keuls
REGW.test(modelo, "feed", console = TRUE)    # REGW
HSD.test(modelo, "feed", console = TRUE)     # Tukey

# 5. A simulação que mostra o problema: médias TODAS iguais
set.seed(2026)
simula <- function(k, n = 12, M = 2000, alfa = 0.05) {
  gl <- k * (n - 1)
  q_tuk <- qtukey(1 - alfa, k, gl)
  q_dun <- sapply(2:k, function(p) qtukey((1 - alfa)^(p - 1), p, gl))
  dun <- tuk <- 0
  for (b in 1:M) {
    y <- matrix(rnorm(k * n), nrow = n)
    m <- sort(colMeans(y), decreasing = TRUE)
    ep <- sqrt(mean(apply(y, 2, var)) / n)
    if ((m[1] - m[k]) / ep > q_tuk) tuk <- tuk + 1
    achou <- FALSE; hom <- list()
    for (pp in k:2) for (a in 1:(k - pp + 1)) {
      bl <- a:(a + pp - 1)
      if (any(sapply(hom, function(h) all(bl %in% h)))) next
      if ((m[bl[1]] - m[bl[pp]]) / ep > q_dun[pp - 1]) achou <- TRUE
      else hom[[length(hom) + 1]] <- bl
    }
    if (achou) dun <- dun + 1
  }
  c(duncan = dun / M, tukey = tuk / M)
}
sapply(3:6, function(k) simula(k))   # taxa de erro do conjunto, por número de grupos

# 6. Gráfico
par(mar = c(5, 8, 4, 2))
res <- duncan.test(modelo, "feed")
barplot(res$means[, 1], horiz = TRUE, las = 1, col = "gray85",
        xlim = c(0, 420), xlab = "Peso aos 6 dias (g)")
par(mar = c(5, 4, 4, 2))
