# Dunnett C: exemplo completo ----
# Pacotes: stats (base). O SPSS traz o Dunnett C pronto; no R, a conta é curta.
# Para comparar: rstatix (Games-Howell), PMCMRplus, car
# install.packages(c("rstatix", "car"))

# 1. Dados fictícios: resistência à compressão (MPa) de corpos de prova de concreto
#    de quatro centrais dosadoras. É o conjunto das fichas da ANOVA de Welch e de
#    Games-Howell.
resistencia <- c(57.0,54.0,51.5,31.1,55.9,32.9,60.2,57.5,59.5,59.4,
                 54.7,47.2,49.2,65.6,46.8,49.9,45.5,47.3,49.2,54.3,
                 48.4,48.3,47.8,49.2,46.3,49.1,51.6,47.5,48.9,54.6,
                 50.5,48.9,51.9,57.5,58.1,50.0,54.4,53.0,51.5,48.1,55.6,50.8,
                 55.6,55.2,58.7,57.4,52.7,54.5,57.1,52.3,54.2,55.5)
central <- factor(rep(c("A", "B", "C", "D"), times = c(20, 10, 12, 10)))
dados <- data.frame(central, resistencia)
aggregate(resistencia ~ central, data = dados,
          FUN = function(x) c(n = length(x), media = mean(x), var = var(x)))

# 2. O pressuposto que justifica o procedimento
library(car)
leveneTest(resistencia ~ central, data = dados, center = median)
v <- tapply(dados$resistencia, dados$central, var); max(v) / min(v)

# 3. O teste global compatível: ANOVA de Welch
oneway.test(resistencia ~ central, data = dados, var.equal = FALSE)

# 4. Dunnett C, implementado diretamente
alfa <- 0.05
m  <- tapply(dados$resistencia, dados$central, mean)
s2 <- tapply(dados$resistencia, dados$central, var)
n  <- table(dados$central)
k  <- nlevels(dados$central)
q_ind <- sapply(n, function(ni) qtukey(1 - alfa, k, ni - 1))   # q(k, n_i - 1)
q_ind

pares <- combn(levels(dados$central), 2)
resultado <- t(apply(pares, 2, function(p_) {
  i <- p_[1]; j <- p_[2]
  si <- s2[i]/n[i]; sj <- s2[j]/n[j]
  dif <- m[j] - m[i]
  ep  <- sqrt((si + sj) / 2)                       # erro-padrão na escala de q
  qc  <- (q_ind[i] * si + q_ind[j] * sj) / (si + sj)   # valor crítico PONDERADO
  c(diferenca = dif, ep = ep, q_obs = abs(dif)/ep, q_critico = qc,
    inferior = dif - qc * ep, superior = dif + qc * ep,
    difere = as.numeric(abs(dif)/ep > qc))
}))
rownames(resultado) <- apply(pares, 2, paste, collapse = " - ")
round(resultado, 4)

# 5. Para comparar: Games-Howell e Tukey-Kramer
library(rstatix)
games_howell_test(dados, resistencia ~ central)
TukeyHSD(aov(resistencia ~ central, data = dados))

# 6. Por que o valor crítico é uma média ponderada
#    Quando as duas variâncias são iguais, qc é a média simples de q(k, n_i-1) e
#    q(k, n_j-1). Quando uma delas domina, qc tende ao q daquele grupo:
si <- s2["A"]/n["A"]; sj <- s2["D"]/n["D"]
c(peso_A = si/(si+sj), peso_D = sj/(si+sj),
  q_A = q_ind["A"], q_D = q_ind["D"],
  q_ponderado = (q_ind["A"]*si + q_ind["D"]*sj)/(si+sj))

# 7. Gráfico: as médias com o intervalo de cada grupo
eps <- sqrt(s2 / n)
plot(1:k, m, ylim = c(44, 60), pch = 18, cex = 2, xaxt = "n",
     xlab = "Central dosadora", ylab = "Resistência (MPa)")
axis(1, at = 1:k, labels = levels(dados$central))
arrows(1:k, m - 1.96*eps, 1:k, m + 1.96*eps, angle = 90, code = 3, length = 0.08)
