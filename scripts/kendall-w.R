# Coeficiente de concordância de Kendall (W): exemplo completo ----
# Pacotes: irr (kendall), DescTools (KendallW), vegan
# install.packages(c("irr", "DescTools"))

# Os mesmos dados das demais fichas de concordância.
sev <- matrix(c(
 3,3,4,3,4, 5,4,4,4,5, 4,4,5,3,5, 5,5,5,4,5, 5,5,5,4,3,
 1,2,2,1,1, 2,2,3,2,4, 1,2,1,1,1, 3,3,4,3,4, 3,4,5,4,2,
 4,3,4,3,4, 3,4,4,3,3, 1,1,1,1,1, 3,3,5,3,3, 2,3,2,2,1,
 5,5,5,5,4, 5,5,4,3,5, 4,5,5,4,4, 2,2,5,2,2, 3,4,5,4,3,
 4,4,4,4,1, 1,2,3,1,3, 3,3,3,3,3, 2,3,5,2,3, 2,1,2,2,2,
 2,1,3,2,3, 3,3,4,3,1, 2,1,1,1,5, 3,2,4,5,4, 1,1,1,1,2,
 4,5,5,4,4, 2,2,4,2,1, 1,1,4,1,3, 1,1,2,1,1, 3,3,3,2,3,
 3,3,3,3,3, 3,5,4,3,2, 2,2,4,2,1, 4,2,5,4,4, 5,5,5,5,2),
 ncol = 5, byrow = TRUE,
 dimnames = list(NULL, c("A","B","C","D","E")))

library(irr)
kendall(sev, correct = TRUE)      # W com correção para empates
kendall(sev, correct = FALSE)     # sem correção

library(DescTools)
KendallW(sev, correct = TRUE, test = TRUE)
KendallW(sev, correct = TRUE, test = TRUE, conf.level = 0.95)

# A conta à mão
n <- nrow(sev); m <- ncol(sev)
R  <- apply(sev, 2, rank)         # postos DENTRO de cada avaliador
Rt <- rowSums(R)                  # soma dos postos de cada folha
mean(Rt)                          # é sempre m(n+1)/2
S  <- sum((Rt - mean(Rt))^2); S
TT <- sum(apply(R, 2, function(r) {
  t <- table(r); sum(t^3 - t) }))  # correção para empates
TT
12*S/(m^2*(n^3 - n) - m*TT)

# As três leituras equivalentes de W
# (a) proporção da variação máxima de Rt
Smax <- (m^2*(n^3 - n) - m*TT)/12
S/Smax

# (b) função da correlação de Spearman média
pares <- combn(m, 2)
rs <- apply(pares, 2, function(p) cor(sev[, p[1]], sev[, p[2]], method = "spearman"))
rbar <- mean(rs); rbar
((m - 1)*rbar + 1)/m              # é exatamente W

# (c) relação com o teste de Friedman
W <- 12*S/(m^2*(n^3 - n) - m*TT)
m*(n - 1)*W                       # a estatística de Friedman
friedman.test(t(sev))             # os mesmos números
pchisq(m*(n-1)*W, df = n - 1, lower.tail = FALSE)

# ATENÇÃO aos papéis: em friedman.test(), as COLUNAS são os tratamentos
# e as LINHAS os blocos. Para o W de Kendall das folhas, as folhas fazem
# o papel de tratamentos e os avaliadores o de blocos: por isso t(sev).

# O que W NÃO mede: deslocamentos constantes
sev2 <- sev; sev2[, "C"] <- pmin(sev2[, "C"] + 1, 6)
kendall(sev2)$value               # praticamente igual
# E o que ele mede: inversões de ordem
sev3 <- sev; sev3[, "C"] <- 6 - sev3[, "C"]
kendall(sev3)$value               # cai muito

# Quem destoa na ordenação
sapply(1:m, function(j) kendall(sev[, -j])$value)

# Valor-p por permutação, indispensável com n pequeno
set.seed(4)
permuta <- function(X, B = 10000) {
  Wobs <- kendall(X)$value
  mean(replicate(B, kendall(apply(X, 2, sample))$value) >= Wobs)
}
permuta(sev[1:6, ])               # com 6 folhas, a aproximação falha

# Gráficos
matplot(apply(sev, 2, rank)[order(rowSums(apply(sev, 2, rank))), ],
        type = "l", lty = 1, col = 1:5, xlab = "Folhas (ordenadas)",
        ylab = "Posto atribuído")
legend("topleft", colnames(sev), col = 1:5, lty = 1, bty = "n")
