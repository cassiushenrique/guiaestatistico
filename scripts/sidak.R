# Ajuste de Šidák: exemplo completo ----
# Pacotes: stats (base). Opcionais: emmeans, mutoss
# install.packages(c("emmeans", "mutoss"))

# 1. A fórmula, em uma linha
alfa <- 0.05; m <- 15
alfa / m                       # Bonferroni
1 - (1 - alfa)^(1/m)           # Šidák
(1 - (1 - alfa)^(1/m)) / (alfa / m)    # quanto Šidák é menos exigente

# 2. Como as duas se comportam quando m cresce
comparacao <- data.frame(
  m          = c(1, 2, 3, 5, 10, 15, 20, 50, 100, 1000),
  bonferroni = alfa / c(1, 2, 3, 5, 10, 15, 20, 50, 100, 1000),
  sidak      = 1 - (1 - alfa)^(1 / c(1, 2, 3, 5, 10, 15, 20, 50, 100, 1000)))
comparacao$razao <- comparacao$sidak / comparacao$bonferroni
round(comparacao, 8)

# 3. Taxa de erro do conjunto que cada um atinge, SE os testes forem independentes
mm <- c(1, 3, 5, 10, 15, 50)
data.frame(m = mm,
           sem_ajuste  = 1 - (1 - alfa)^mm,
           bonferroni  = 1 - (1 - alfa/mm)^mm,
           sidak       = 1 - (1 - (1 - (1 - alfa)^(1/mm)))^mm)

# 4. Aplicando aos dados: chickwts
data(chickwts)
modelo <- aov(weight ~ feed, data = chickwts)
summary(modelo)

brutos <- with(chickwts, pairwise.t.test(weight, feed, p.adjust.method = "none"))$p.value
p <- as.vector(brutos); p <- p[!is.na(p)]
m <- length(p); m

sidak      <- 1 - (1 - p)^m                  # Šidák de passo único
bonferroni <- p.adjust(p, "bonferroni")
holm       <- p.adjust(p, "holm")
# Holm-Šidák (passo descendente): sequencial, usando a fórmula de Šidák
ordem <- order(p); hs <- numeric(m); acumulado <- 0
for (i in seq_len(m)) {
  acumulado <- max(acumulado, 1 - (1 - p[ordem[i]])^(m - i + 1))
  hs[ordem[i]] <- min(1, acumulado)
}
resultado <- data.frame(p_bruto = p, sidak, bonferroni, holm, holm_sidak = hs)
round(resultado[order(p), ], 5)
colSums(resultado < 0.05)                    # quantas sobrevivem a cada ajuste

# 5. Šidák em emmeans, onde ele aparece com nome próprio
library(emmeans)
em <- emmeans(modelo, ~feed)
pairs(em, adjust = "sidak")
pairs(em, adjust = "bonferroni")
pairs(em, adjust = "tukey")

# 6. Intervalos simultâneos: multiplicadores críticos lado a lado
gl <- df.residual(modelo); k <- nlevels(chickwts$feed)
c(sem_ajuste = qt(1 - alfa/2, gl),
  sidak      = qt(1 - (1 - (1 - alfa)^(1/m))/2, gl),
  bonferroni = qt(1 - alfa/(2*m), gl),
  tukey      = qtukey(1 - alfa, k, gl) / sqrt(2),
  scheffe    = sqrt((k - 1) * qf(1 - alfa, k - 1, gl)))

# 7. Gráfico: as duas curvas de nível individual
curve(0.05/x, 1, 50, log = "y", col = "red", lwd = 2,
      xlab = "Número de testes (m)", ylab = "Nível individual")
curve(1 - 0.95^(1/x), 1, 50, col = "darkgreen", lwd = 2, lty = 2, add = TRUE)
legend("topright", c("Bonferroni", "Šidák"), col = c("red", "darkgreen"), lwd = 2, lty = 1:2)
