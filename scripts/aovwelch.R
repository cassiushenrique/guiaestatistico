# ANOVA de Welch: exemplo completo ----
# Pacotes: stats (base), car (Brown-Forsythe), rstatix ou PMCMRplus (Games-Howell)
# install.packages(c("car", "rstatix"))
library(car)

# 1. Dados fictícios: resistência à compressão (MPa) de corpos de prova de concreto
#    de quatro centrais dosadoras, com números de amostras diferentes.
resistencia <- c(57.0, 54.0, 51.5, 31.1, 55.9, 32.9, 60.2, 57.5, 59.5, 59.4,
                 54.7, 47.2, 49.2, 65.6, 46.8, 49.9, 45.5, 47.3, 49.2, 54.3,
                 48.4, 48.3, 47.8, 49.2, 46.3, 49.1, 51.6, 47.5, 48.9, 54.6,
                 50.5, 48.9, 51.9, 57.5, 58.1, 50.0, 54.4, 53.0, 51.5, 48.1, 55.6, 50.8,
                 55.6, 55.2, 58.7, 57.4, 52.7, 54.5, 57.1, 52.3, 54.2, 55.5)
central <- factor(rep(c("A", "B", "C", "D"), times = c(20, 10, 12, 10)))
dados <- data.frame(central, resistencia)
aggregate(resistencia ~ central, data = dados,
          FUN = function(x) c(n = length(x), media = mean(x), dp = sd(x), var = var(x)))

# 2. O pressuposto que decide qual ANOVA usar
leveneTest(resistencia ~ central, data = dados, center = median)   # Brown-Forsythe
bartlett.test(resistencia ~ central, data = dados)
v <- tapply(dados$resistencia, dados$central, var)
max(v) / min(v)                                   # razão entre as variâncias
boxplot(resistencia ~ central, data = dados, col = "gray85")

# 3. As duas ANOVAs, lado a lado
oneway.test(resistencia ~ central, data = dados, var.equal = TRUE)    # clássica
oneway.test(resistencia ~ central, data = dados, var.equal = FALSE)   # de Welch
summary(aov(resistencia ~ central, data = dados))                     # igual à clássica

# 4. Normalidade: examine POR GRUPO, não nos resíduos agrupados
by(dados$resistencia, dados$central, shapiro.test)
shapiro.test(residuals(aov(resistencia ~ central, data = dados)))

# 5. Comparações múltiplas compatíveis com variâncias desiguais
library(rstatix)
games_howell_test(dados, resistencia ~ central)
# alternativas: PMCMRplus::gamesHowellTest(resistencia ~ central, data = dados)
#               pairwise.t.test(dados$resistencia, dados$central, pool.sd = FALSE,
#                               p.adjust.method = "holm")
TukeyHSD(aov(resistencia ~ central, data = dados))   # para comparar: usa variância agrupada

# 6. Tamanho de efeito
tab <- summary(aov(resistencia ~ central, data = dados))[[1]]
tab[1, "Sum Sq"] / sum(tab[, "Sum Sq"])            # eta ao quadrado

# 7. Gráficos
par(mfrow = c(1, 2))
medias <- tapply(dados$resistencia, dados$central, mean)
eps <- tapply(dados$resistencia, dados$central, function(x) sd(x) / sqrt(length(x)))
plot(1:4, medias, ylim = range(dados$resistencia), pch = 18, cex = 2, xaxt = "n",
     xlab = "Central", ylab = "Resistência (MPa)")
axis(1, at = 1:4, labels = levels(dados$central))
arrows(1:4, medias - 1.96 * eps, 1:4, medias + 1.96 * eps, angle = 90, code = 3, length = 0.1)
boxplot(resistencia ~ central, data = dados, col = "gray85",
        xlab = "Central", ylab = "Resistência (MPa)")
par(mfrow = c(1, 1))
