# Kappa de Cohen: exemplo completo ----
# Pacotes: irr (kappa2), vcd (Kappa com IC), psych, DescTools
# install.packages(c("irr", "vcd", "psych", "DescTools"))

# Dados fictícios: 40 folhas de cafeeiro, severidade de ferrugem de 1 a 5,
# avaliadas por cinco avaliadores. O mesmo conjunto das demais fichas de
# concordância.
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
 dimnames = list(paste0("folha", 1:40), c("A","B","C","D","E")))
head(sev)
apply(sev, 2, function(x) table(factor(x, levels = 1:5)))
colMeans(sev)

# O kappa de Cohen é para DOIS avaliadores e escala NOMINAL.
# Dicotomizando em severidade alta (4 e 5) contra baixa (1 a 3):
alta <- ifelse(sev >= 4, "alta", "baixa")
tab  <- table(A = alta[, "A"], B = alta[, "B"])
tab
addmargins(tab)
prop.table(tab)

library(irr)
kappa2(alta[, c("A", "B")])          # kappa, z e valor-p

library(vcd)
Kappa(tab)                            # kappa não ponderado e ponderado
confint(Kappa(tab))                   # intervalos de confiança

# A conta à mão
P  <- prop.table(tab)
po <- sum(diag(P)); po
pe <- sum(rowSums(P) * colSums(P)); pe
(po - pe) / (1 - pe)

# Erro-padrão e IC
library(psych)
cohen.kappa(alta[, c("A", "B")])

# Os dois paradoxos de Feinstein e Cicchetti
t_prev <- matrix(c(85, 5, 5, 5), 2, byrow = TRUE)     # prevalência extrema
t_vies1 <- matrix(c(45, 15, 25, 15), 2, byrow = TRUE) # marginais parecidas
t_vies2 <- matrix(c(25, 35, 5, 35), 2, byrow = TRUE)  # marginais diferentes
sapply(list(t_prev, t_vies1, t_vies2), function(T) {
  P <- prop.table(T); po <- sum(diag(P))
  pe <- sum(rowSums(P) * colSums(P))
  c(po = po, pe = pe, kappa = (po - pe)/(1 - pe), pabak = 2*po - 1,
    ip = abs(T[1,1] - T[2,2])/sum(T), iv = abs(T[1,2] - T[2,1])/sum(T))
})

# Todos os pares, na escala completa de cinco classes
pares <- combn(colnames(sev), 2)
res <- t(apply(pares, 2, function(p) {
  k <- kappa2(sev[, p])
  c(kappa = k$value, z = k$statistic, p = k$p.value,
    concordancia = mean(sev[, p[1]] == sev[, p[2]]))
}))
rownames(res) <- apply(pares, 2, paste, collapse = "-")
round(res, 4)
mean(res[, "kappa"])                  # kappa de Light

# Gráficos
library(vcd)
agreementplot(tab, main = "Concordância entre A e B")
mosaic(table(A = sev[,"A"], B = sev[,"B"]), shade = TRUE)
