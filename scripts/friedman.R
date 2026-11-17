# Teste de Friedman: exemplo completo ----
# Pacotes: stats (base), PMCMRplus ou rstatix (comparações múltiplas)
# install.packages(c("PMCMRplus", "rstatix"))

# 1. Dados: 10 provadores avaliaram 4 formulações de um produto (escala 0 a 10)
#    Cada provador avaliou TODAS as formulações: delineamento em blocos.
notas <- matrix(c(4.8, 5.6, 7.2, 5.8,
                  7.4, 8.2, 8.4, 7.5,
                  6.5, 6.5, 7.8, 6.0,
                  7.2, 7.1, 8.0, 7.0,
                  7.2, 7.4, 9.1, 7.4,
                  6.0, 7.2, 8.0, 5.5,
                  7.7, 7.6, 7.4, 7.2,
                  5.9, 6.1, 7.8, 6.4,
                  4.9, 5.2, 6.5, 6.4,
                  5.4, 6.2, 6.9, 6.3),
                nrow = 10, byrow = TRUE,
                dimnames = list(paste("provador", 1:10), c("A", "B", "C", "D")))
notas
apply(notas, 2, function(x) c(media = mean(x), mediana = median(x)))

# 2. Os postos são calculados DENTRO de cada bloco (cada provador)
postos <- t(apply(notas, 1, rank))
postos
colSums(postos)            # R_j: soma dos postos de cada formulação
colMeans(postos)           # posto médio; sob H0, todos valeriam (k+1)/2 = 2,5

# 3. O teste
friedman.test(notas)

# 4. Equivalente com os dados em formato longo
longo <- data.frame(nota = as.vector(notas),
                    formulacao = rep(colnames(notas), each = nrow(notas)),
                    provador = factor(rep(rownames(notas), times = ncol(notas))))
friedman.test(nota ~ formulacao | provador, data = longo)

# 5. Concordância entre os provadores: W de Kendall
Q <- friedman.test(notas)$statistic
n <- nrow(notas); k <- ncol(notas)
Q / (n * (k - 1))
# equivalente: rstatix::friedman_effsize(longo, nota ~ formulacao | provador)

# 6. Comparações múltiplas
PMCMRplus::frdAllPairsConoverTest(notas, p.adjust.method = "bonferroni")
PMCMRplus::frdAllPairsNemenyiTest(notas)
pairwise.wilcox.test(longo$nota, longo$formulacao, paired = TRUE, p.adjust.method = "bonferroni")

# 7. Comparação com a ANOVA de medidas repetidas
anova_mr <- aov(nota ~ formulacao + Error(provador / formulacao), data = longo)
summary(anova_mr)
shapiro.test(residuals(lm(nota ~ formulacao + provador, data = longo)))

# 8. Gráficos
par(mfrow = c(1, 2))
matplot(t(notas), type = "b", pch = 19, lty = 1, col = "gray60", xaxt = "n",
        xlab = "Formulação", ylab = "Nota")
axis(1, at = 1:4, labels = colnames(notas))
lines(colMeans(notas), type = "b", pch = 18, col = "red", lwd = 3)
barplot(colMeans(postos), col = "gray85", xlab = "Formulação", ylab = "Posto médio")
abline(h = (ncol(notas) + 1) / 2, col = "red", lwd = 2, lty = 2)
par(mfrow = c(1, 1))
