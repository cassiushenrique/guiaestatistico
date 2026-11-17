# Ajuste de Bonferroni: exemplo completo ----
# Pacotes: stats (base). Opcionais: multcomp, emmeans, rstatix
# install.packages(c("multcomp", "emmeans"))

# 1. Dados: chickwts, peso de pintinhos por tipo de ração
data(chickwts)
table(chickwts$feed)
modelo <- aov(weight ~ feed, data = chickwts)
summary(modelo)

# 2. A conta que justifica o ajuste
k <- nlevels(chickwts$feed); m <- k * (k - 1) / 2
m                                   # 15 comparações
1 - 0.95^m                          # risco do conjunto SE os testes fossem independentes

# 3. A forma mais direta: p.adjust() sobre valores-p brutos
with(chickwts, pairwise.t.test(weight, feed, p.adjust.method = "none"))
with(chickwts, pairwise.t.test(weight, feed, p.adjust.method = "bonferroni"))
with(chickwts, pairwise.t.test(weight, feed, p.adjust.method = "holm"))

# 4. p.adjust() funciona com QUALQUER vetor de valores-p
p <- c(0.001, 0.013, 0.021, 0.049, 0.310)
p.adjust(p, method = "bonferroni")
p.adjust(p, method = "holm")        # mais poderoso, mesma proteção
data.frame(bruto = p,
           bonferroni = p.adjust(p, "bonferroni"),
           holm = p.adjust(p, "holm"),
           BH_fdr = p.adjust(p, "BH"))

# 5. Intervalos de confiança simultâneos de Bonferroni
alfa <- 0.05
gl <- df.residual(modelo); QM <- sum(residuals(modelo)^2) / gl
tcrit <- qt(1 - alfa / (2 * m), gl)
medias <- tapply(chickwts$weight, chickwts$feed, mean)
ns <- table(chickwts$feed)
pares <- combn(levels(chickwts$feed), 2)
ic <- t(apply(pares, 2, function(p_) {
  d <- medias[p_[2]] - medias[p_[1]]
  ep <- sqrt(QM * (1/ns[p_[1]] + 1/ns[p_[2]]))
  c(diferenca = d, inferior = d - tcrit * ep, superior = d + tcrit * ep)
}))
rownames(ic) <- apply(pares, 2, paste, collapse = " - ")
round(ic, 2)

# 6. Bonferroni não é só para comparações pareadas: contrastes planejados
library(multcomp)
levels(chickwts$feed)   # a ordem das colunas: casein horsebean linseed meatmeal soybean sunflower
contrastes <- rbind(
  "vegetais - animais"     = c(-1/2,  1/3,  1/3, -1/2,  1/3,  0  ),
  "sunflower - as demais"  = c(-1/5, -1/5, -1/5, -1/5, -1/5,  1  ),
  "horsebean - as demais"  = c(-1/5,  1,   -1/5, -1/5, -1/5, -1/5))
rowSums(contrastes)      # cada linha deve somar zero
summary(glht(modelo, linfct = mcp(feed = contrastes)), test = adjusted("bonferroni"))
confint(glht(modelo, linfct = mcp(feed = contrastes)))

# 7. Comparação com Tukey, construído para este conjunto de comparações
TukeyHSD(modelo)

# 8. Gráfico
par(mar = c(5, 10, 4, 2))
plot(TukeyHSD(modelo), las = 1, cex.axis = 0.7)
par(mar = c(5, 4, 4, 2))
