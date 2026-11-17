# Coeficiente AC1 de Gwet: exemplo completo ----
# Pacotes: irrCAC (do próprio Gwet), irr, DescTools
# install.packages(c("irrCAC", "irr"))

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

library(irrCAC)
df <- as.data.frame(sev)
gwet.ac1.raw(df)                   # AC1, erro-padrão e IC
fleiss.kappa.raw(df)               # kappa de Fleiss, para comparar
krippen.alpha.raw(df)              # alfa de Krippendorff
pa.coeff.raw(df)                   # concordância bruta

# Versão PONDERADA (AC2), para escala ordinal
gwet.ac1.raw(df, weights = "ordinal")
gwet.ac1.raw(df, weights = "linear")
gwet.ac1.raw(df, weights = "quadratic")
# Atenção: em irrCAC os pesos são de CONCORDÂNCIA (1 na diagonal),
# ao contrário de irr::kappa2(), em que são de DESACORDO.

# A conta à mão
n <- nrow(sev); m <- ncol(sev); k <- 5
N  <- t(apply(sev, 1, function(r) table(factor(r, levels = 1:k))))
pi_ <- colSums(N)/(n*m)
po <- mean((rowSums(N^2) - m)/(m*(m-1)))
pe_gwet   <- sum(pi_*(1 - pi_))/(k - 1)
pe_fleiss <- sum(pi_^2)
c(po = po,
  AC1    = (po - pe_gwet)/(1 - pe_gwet),
  Fleiss = (po - pe_fleiss)/(1 - pe_fleiss))

# O paradoxo da prevalência, lado a lado
tabelas <- list(equilibrada = matrix(c(45,5,5,45), 2, byrow = TRUE),
                desequil    = matrix(c(80,5,5,10), 2, byrow = TRUE),
                extrema     = matrix(c(92,4,4,0),  2, byrow = TRUE))
t(sapply(tabelas, function(T) {
  P <- prop.table(T); po <- sum(diag(P))
  pe_k <- sum(rowSums(P)*colSums(P))
  pbar <- (rowSums(P) + colSums(P))/2
  pe_g <- 2*pbar[1]*pbar[2]
  c(po = po, pe_kappa = pe_k, kappa = (po - pe_k)/(1 - pe_k),
    pe_ac1 = pe_g, ac1 = (po - pe_g)/(1 - pe_g), pabak = 2*po - 1)
}))

# O mesmo cálculo em escalas colapsadas
s1 <- ifelse(sev == 1, 1, 2)          # sem sintoma x com sintoma
df1 <- as.data.frame(s1)
gwet.ac1.raw(df1)$est
fleiss.kappa.raw(df1)$est

# Gráficos
p <- seq(0.5, 0.99, length.out = 200)
plot(p, p^2 + (1-p)^2, type = "l", col = "firebrick", lwd = 2, ylim = c(0, 1),
     xlab = "Prevalência da categoria dominante", ylab = "pe")
lines(p, 2*p*(1-p), col = "darkgreen", lwd = 2)
legend("left", c("pe do kappa", "pe do AC1"),
       col = c("firebrick", "darkgreen"), lwd = 2, bty = "n")
