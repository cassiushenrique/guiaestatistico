# Teste de Kruskal-Wallis: exemplo completo ----
# Pacotes: stats (base), car (Brown-Forsythe), FSA ou dunn.test (comparações múltiplas)
# install.packages(c("car", "FSA"))
library(car)
library(FSA)

# 1. Dados: contagem de insetos sobreviventes com seis inseticidas
#    InsectSprays vem com o R; fonte original: Beall (1942).
data(InsectSprays)
str(InsectSprays)
aggregate(count ~ spray, data = InsectSprays,
          FUN = function(x) c(n = length(x), media = mean(x), mediana = median(x), var = var(x)))

# 2. Por que não a ANOVA clássica: variâncias muito diferentes
leveneTest(count ~ spray, data = InsectSprays, center = median)   # Brown-Forsythe
tapply(InsectSprays$count, InsectSprays$spray, var)
by(InsectSprays$count, InsectSprays$spray, shapiro.test)

# 3. O teste
kruskal.test(count ~ spray, data = InsectSprays)

# 4. Os postos por trás do resultado
InsectSprays$posto <- rank(InsectSprays$count)
tapply(InsectSprays$posto, InsectSprays$spray, mean)        # postos médios
mean(InsectSprays$posto)                                   # posto médio geral = (N+1)/2

# 5. Comparações múltiplas de Dunn
dunnTest(count ~ spray, data = InsectSprays, method = "bonferroni")
# alternativa: PMCMRplus::kwAllPairsDunnTest(count ~ spray, data = InsectSprays, p.adjust.method = "bonferroni")

# 6. Tamanho de efeito (epsilon ao quadrado)
H <- kruskal.test(count ~ spray, data = InsectSprays)$statistic
N <- nrow(InsectSprays)
H / ((N^2 - 1) / (N + 1))
# equivalente: rstatix::kruskal_effsize(InsectSprays, count ~ spray)

# 7. Comparação com a ANOVA de Welch e com a ANOVA clássica
oneway.test(count ~ spray, data = InsectSprays, var.equal = FALSE)
summary(aov(count ~ spray, data = InsectSprays))

# 8. Gráficos
par(mfrow = c(1, 2))
boxplot(count ~ spray, data = InsectSprays, col = "gray85",
        xlab = "Inseticida", ylab = "Contagem de insetos")
points(jitter(as.numeric(InsectSprays$spray), 0.6), InsectSprays$count, pch = 19, col = "gray40")
barplot(tapply(InsectSprays$posto, InsectSprays$spray, mean), col = "gray85",
        xlab = "Inseticida", ylab = "Posto médio")
abline(h = (nrow(InsectSprays) + 1) / 2, col = "red", lwd = 2, lty = 2)
par(mfrow = c(1, 1))
