# Games-Howell: exemplo completo ----
# Pacotes: rstatix ou PMCMRplus. Opcional: userfriendlyscience (descontinuado)
# install.packages(c("rstatix", "PMCMRplus", "car"))

# 1. Dados fictícios: resistência à compressão (MPa) de corpos de prova de concreto
#    de quatro centrais dosadoras. É o conjunto da ficha da ANOVA de Welch.
resistencia <- c(57.0,54.0,51.5,31.1,55.9,32.9,60.2,57.5,59.5,59.4,
                 54.7,47.2,49.2,65.6,46.8,49.9,45.5,47.3,49.2,54.3,
                 48.4,48.3,47.8,49.2,46.3,49.1,51.6,47.5,48.9,54.6,
                 50.5,48.9,51.9,57.5,58.1,50.0,54.4,53.0,51.5,48.1,55.6,50.8,
                 55.6,55.2,58.7,57.4,52.7,54.5,57.1,52.3,54.2,55.5)
central <- factor(rep(c("A", "B", "C", "D"), times = c(20, 10, 12, 10)))
dados <- data.frame(central, resistencia)
aggregate(resistencia ~ central, data = dados,
          FUN = function(x) c(n = length(x), media = mean(x), dp = sd(x), var = var(x)))

# 2. O pressuposto que decide entre Tukey e Games-Howell
library(car)
leveneTest(resistencia ~ central, data = dados, center = median)
v <- tapply(dados$resistencia, dados$central, var); max(v) / min(v)
boxplot(resistencia ~ central, data = dados, col = "gray85")

# 3. O teste global compatível: ANOVA de Welch, não a clássica
oneway.test(resistencia ~ central, data = dados, var.equal = FALSE)
oneway.test(resistencia ~ central, data = dados, var.equal = TRUE)   # para comparar

# 4. Games-Howell
library(rstatix)
games_howell_test(dados, resistencia ~ central)
# alternativas:
# PMCMRplus::gamesHowellTest(resistencia ~ central, data = dados)
# pairwise.t.test(dados$resistencia, dados$central, pool.sd = FALSE, p.adjust.method = "holm")

# 5. Para comparar: Tukey, que usa a variância agrupada
TukeyHSD(aov(resistencia ~ central, data = dados))

# 6. O cálculo passo a passo de um par, para entender a fórmula
m <- tapply(dados$resistencia, dados$central, mean)
s2 <- tapply(dados$resistencia, dados$central, var)
n <- table(dados$central)
i <- "B"; j <- "D"
dif <- m[j] - m[i]
ep  <- sqrt((s2[i]/n[i] + s2[j]/n[j]) / 2)              # erro-padrão na escala de q
gl  <- (s2[i]/n[i] + s2[j]/n[j])^2 /
       ((s2[i]/n[i])^2/(n[i]-1) + (s2[j]/n[j])^2/(n[j]-1))   # Welch-Satterthwaite
q   <- abs(dif) / ep
c(diferenca = dif, ep = ep, gl = gl, q = q,
  p = ptukey(q, nmeans = 4, df = gl, lower.tail = FALSE),
  inferior = dif - qtukey(0.95, 4, gl) * ep,
  superior = dif + qtukey(0.95, 4, gl) * ep)

# 7. Gráfico: médias com o intervalo de cada grupo, não com o agrupado
medias <- m; eps <- sqrt(s2 / n)
plot(1:4, medias, ylim = c(44, 60), pch = 18, cex = 2, xaxt = "n",
     xlab = "Central dosadora", ylab = "Resistência (MPa)")
axis(1, at = 1:4, labels = levels(dados$central))
arrows(1:4, medias - 1.96 * eps, 1:4, medias + 1.96 * eps, angle = 90, code = 3, length = 0.08)
