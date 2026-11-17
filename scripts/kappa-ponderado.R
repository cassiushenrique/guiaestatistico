# Kappa ponderado: exemplo completo ----
# Pacotes: irr (kappa2 com pesos), vcd (Kappa), psych, DescTools
# install.packages(c("irr", "vcd", "psych"))

# Os mesmos dados da ficha do kappa de Cohen: 40 folhas, cinco avaliadores,
# severidade de ferrugem em escala ORDINAL de 1 a 5.
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

tab <- table(A = factor(sev[,"A"], 1:5), B = factor(sev[,"B"], 1:5))
tab
sum(diag(tab))                              # acordo exato
sum(tab[abs(row(tab) - col(tab)) == 1])     # desacordo de uma classe

# Os três kappas
library(irr)
kappa2(sev[, c("A","B")], weight = "unweighted")
kappa2(sev[, c("A","B")], weight = "equal")      # pesos LINEARES
kappa2(sev[, c("A","B")], weight = "squared")    # pesos QUADRÁTICOS

library(vcd)
Kappa(tab)                                  # devolve as duas de uma vez
confint(Kappa(tab))

# As matrizes de peso, explicitamente (no R, pesos de DESACORDO)
k <- 5; i <- row(matrix(0, k, k)); j <- col(matrix(0, k, k))
w_nom <- 1 * (i != j)
w_lin <- abs(i - j)/(k - 1)
w_qua <- (i - j)^2/(k - 1)^2
round(w_lin, 4); round(w_qua, 4)

# A conta à mão, com pesos quadráticos
P  <- prop.table(tab)
E  <- outer(rowSums(P), colSums(P))
po <- 1 - sum(w_qua * P); pe <- 1 - sum(w_qua * E)
(po - pe)/(1 - pe)

# Intervalo de confiança por bootstrap das unidades
set.seed(3)
bs <- replicate(5000, {
  idx <- sample(nrow(sev), replace = TRUE)
  kappa2(sev[idx, c("A","B")], weight = "squared")$value
})
quantile(bs, c(0.025, 0.975)); sd(bs)

# A identidade de Fleiss e Cohen (1973):
# kappa quadrático = ICC(2,1) sobre os mesmos dados
library(irr)
icc(sev[, c("A","B")], model = "twoway", type = "agreement", unit = "single")
kappa2(sev[, c("A","B")], weight = "squared")$value
# As duas quantidades coincidem até a terceira casa decimal.

# Todos os pares, nos três esquemas
pares <- combn(colnames(sev), 2)
res <- t(apply(pares, 2, function(p) c(
  nominal    = kappa2(sev[, p], weight = "unweighted")$value,
  linear     = kappa2(sev[, p], weight = "equal")$value,
  quadratico = kappa2(sev[, p], weight = "squared")$value)))
rownames(res) <- apply(pares, 2, paste, collapse = "-")
round(res, 4)
cor(res[, "nominal"], res[, "quadratico"], method = "spearman")

# Gráficos
library(vcd)
agreementplot(tab, main = "Concordância entre A e B",
              weights = c(1, 0.75))   # crédito parcial ao desacordo de 1 classe
