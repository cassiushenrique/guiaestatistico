# Coeficiente de correlação intraclasse: exemplo completo ----
# Pacotes: irr (icc), psych (ICC), lme4 (via modelo misto), blandr
# install.packages(c("irr", "psych", "lme4"))

# Dados fictícios: as mesmas 40 folhas das demais fichas de concordância,
# agora com a severidade medida em ESCALA CONTÍNUA (% de área lesionada),
# estimada por quatro avaliadores.
area <- matrix(c(
  1.8, 8.3,26.3,11.1, 32.4,20.9,32.8,26.8, 37.8,34.4,39.3,36.4,
 34.1,34.5,40.7,38.5, 31.6,32.0,41.3,31.0,  2.2, 0.1,11.7, 0.2,
 20.8,10.2,32.3,28.5,  0.3, 8.2,23.8, 6.2, 13.1, 8.0,28.8,33.8,
 24.0,22.3,29.6,17.4, 49.6,38.6,56.5,52.8, 20.3,20.0,27.4,25.8,
  3.3, 0.1, 5.0, 0.1, 20.3,15.7,28.8,19.8, 10.0, 9.1,13.5, 0.1,
 33.0,25.5,39.9,29.9, 27.9,22.2,49.3,42.6, 28.1,21.6,35.1,22.1,
 13.3, 7.8,19.7,15.9, 31.9,10.0,37.0,29.6,  6.2,11.0,22.9,10.6,
  3.6, 0.1,26.4,24.1, 22.3, 9.8,26.1,18.6,  4.7, 4.3,22.0, 7.3,
  0.1, 0.1,17.1, 4.7,  2.5, 8.5,14.5, 0.5,  9.2,18.4,12.8,23.6,
 12.6, 1.5,13.4,16.8, 37.4,30.3,57.1,38.9,  3.5, 0.1,10.0, 4.2,
 21.5,23.1,39.7,32.5, 10.8,10.1,19.4, 2.1,  9.5, 0.1,10.4, 8.2,
  0.1, 6.7, 5.5, 0.1, 14.7,11.3,28.4,18.9, 19.2,10.2,25.1, 8.6,
 19.7,10.0,15.5,27.9,  1.3, 0.1,11.3,12.7, 36.4,38.8,57.4,58.7,
 13.9,12.6,20.1,30.7),
 ncol = 4, byrow = TRUE, dimnames = list(NULL, c("A","B","C","D")))
colMeans(area); apply(area, 2, sd)

# A ANOVA de duas vias que está por trás
longo <- data.frame(y = as.vector(area),
                    folha = factor(rep(1:40, times = 4)),
                    aval  = factor(rep(colnames(area), each = 40)))
anova(aov(y ~ folha + aval, data = longo))
# MSR = QM de folha, MSC = QM de avaliador, MSE = QM do resíduo

# As seis formas de Shrout e Fleiss
library(psych)
ICC(area)            # devolve as seis de uma vez, com IC e teste F

library(irr)
icc(area, model = "oneway",                      type = "consistency", unit = "single")
icc(area, model = "twoway", type = "agreement",   unit = "single")     # ICC(2,1)
icc(area, model = "twoway", type = "consistency", unit = "single")     # ICC(3,1)
icc(area, model = "twoway", type = "agreement",   unit = "average")    # ICC(2,k)

# A conta à mão
n <- nrow(area); k <- ncol(area); gm <- mean(area)
MSR <- k*sum((rowMeans(area) - gm)^2)/(n - 1)
MSC <- n*sum((colMeans(area) - gm)^2)/(k - 1)
MSE <- sum((area - rowMeans(area) - rep(colMeans(area), each = n) + gm)^2)/((n-1)*(k-1))
MSW <- sum((area - rowMeans(area))^2)/(n*(k-1))
c(ICC11 = (MSR - MSW)/(MSR + (k-1)*MSW),
  ICC21 = (MSR - MSE)/(MSR + (k-1)*MSE + k*(MSC - MSE)/n),
  ICC31 = (MSR - MSE)/(MSR + (k-1)*MSE))

# Pelo modelo misto, que é a formulação moderna
library(lme4)
m <- lmer(y ~ (1 | folha) + (1 | aval), data = longo)
vc <- as.data.frame(VarCorr(m)); vc
s2f <- vc$vcov[vc$grp == "folha"]; s2a <- vc$vcov[vc$grp == "aval"]
s2e <- vc$vcov[vc$grp == "Residual"]
s2f/(s2f + s2a + s2e)      # ICC(2,1): concordância absoluta
s2f/(s2f + s2e)            # ICC(3,1): consistência

# Erro-padrão da medida e diferença mínima detectável
EPM <- sqrt(MSE); EPM
1.96*sqrt(2)*EPM

# Bland-Altman para cada par
pares <- combn(colnames(area), 2)
t(apply(pares, 2, function(p) {
  d <- area[, p[1]] - area[, p[2]]
  c(vies = mean(d), dp = sd(d),
    inf = mean(d) - 1.96*sd(d), sup = mean(d) + 1.96*sd(d))
}))
# install.packages("blandr"); blandr::blandr.draw(area[,"B"], area[,"C"])

# Spearman-Brown: quanto se ganha com mais avaliadores
r1 <- icc(area, model = "twoway", type = "agreement", unit = "single")$value
sapply(1:10, function(kk) kk*r1/(1 + (kk-1)*r1))

# O ICC depende da heterogeneidade da amostra
sub <- area[rowMeans(area) < 25, ]
icc(sub, model = "twoway", type = "agreement", unit = "single")$value
sqrt(sum((sub - rowMeans(sub) - rep(colMeans(sub), each = nrow(sub))
          + mean(sub))^2)/((nrow(sub)-1)*(k-1)))   # o EPM mal muda
