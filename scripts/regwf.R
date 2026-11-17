# Procedimento de Ryan-Einot-Gabriel-Welsch: exemplo completo ----
# Pacotes: agricolae (REGW.test), PMCMRplus, mutoss. Base: stats
# install.packages(c("agricolae", "PMCMRplus", "mutoss"))

# 1. Dados: chickwts, peso de pintinhos por tipo de ração
data(chickwts)
modelo <- aov(weight ~ feed, data = chickwts)
summary(modelo)
k <- nlevels(chickwts$feed); gl <- df.residual(modelo)
QM <- sum(residuals(modelo)^2) / gl
sort(tapply(chickwts$weight, chickwts$feed, mean), decreasing = TRUE)

# 2. Os níveis de significância de cada etapa
alfa <- 0.05
p <- 2:k
alfa_regw <- ifelse(p >= k - 1, alfa, 1 - (1 - alfa)^(p / k))
data.frame(p,
           alfa_regw,
           q_regw   = qtukey(1 - alfa_regw, p, gl),
           alfa_duncan = 1 - (1 - alfa)^(p - 1),
           q_duncan = qtukey((1 - alfa)^(p - 1), p, gl),
           q_tukey  = qtukey(1 - alfa, k, gl))

# 3. O procedimento, com agricolae
library(agricolae)
REGW.test(modelo, "feed", console = TRUE)
# comparando com os demais, no mesmo pacote:
HSD.test(modelo, "feed", console = TRUE)       # Tukey
duncan.test(modelo, "feed", console = TRUE)    # Duncan
LSD.test(modelo, "feed", console = TRUE)       # sem ajuste

# 4. Versão com estatística F (REGWF) e versão com amplitude (REGWQ)
library(PMCMRplus)
summary(regwqTest(weight ~ feed, data = chickwts))   # REGWQ
# a variante REGWF usa a estatística F do subconjunto em vez da amplitude

# 5. Reproduzindo o passo descendente à mão, para entender o mecanismo
medias <- sort(tapply(chickwts$weight, chickwts$feed, mean), decreasing = TRUE)
ns <- table(chickwts$feed)[names(medias)]
for (pp in k:2) {
  for (a in 1:(k - pp + 1)) {
    bloco <- names(medias)[a:(a + pp - 1)]
    i <- bloco[1]; j <- bloco[pp]
    ep <- sqrt(QM / 2 * (1/ns[i] + 1/ns[j]))
    q  <- abs(medias[i] - medias[j]) / ep
    ac <- if (pp >= k - 1) alfa else 1 - (1 - alfa)^(pp / k)
    cat(sprintf("p=%d  %-42s q=%6.3f  critico=%6.3f  %s\n",
                pp, paste(bloco, collapse = " > "), q, qtukey(1 - ac, pp, gl),
                ifelse(q > qtukey(1 - ac, pp, gl), "heterogeneo", "HOMOGENEO")))
  }
}
# Atenção: a regra de parada (não testar subconjuntos contidos em um bloco já
# declarado homogêneo) é essencial e NÃO está neste laço simplificado.

# 6. Gráfico
par(mar = c(5, 8, 4, 2))
bp <- barplot(medias, horiz = TRUE, las = 1, col = "gray85", xlim = c(0, 420),
              xlab = "Peso aos 6 dias (g)")
eps <- sqrt(QM / ns)
arrows(medias - 1.96 * eps, bp, medias + 1.96 * eps, bp, angle = 90, code = 3, length = 0.05)
par(mar = c(5, 4, 4, 2))
