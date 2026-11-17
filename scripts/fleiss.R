# Kappa de Fleiss: exemplo completo ----
# Pacotes: irr (kappam.fleiss, kappam.light), DescTools, irrCAC
# install.packages(c("irr", "DescTools"))

# Os mesmos dados das demais fichas de concordância: 40 folhas, cinco
# avaliadores, severidade de ferrugem de 1 a 5.
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
kappam.fleiss(sev)                  # o kappa global, com z e valor-p
kappam.fleiss(sev, detail = TRUE)   # kappa de cada categoria
kappam.light(sev)                   # média dos kappas pareados

# A conta à mão
n <- nrow(sev); m <- ncol(sev); k <- 5
N <- t(apply(sev, 1, function(r) table(factor(r, levels = 1:k))))
head(N)                              # contagens por folha e classe
pj <- colSums(N)/(n*m); pj           # prevalência de cada classe
Pe <- sum(pj^2); Pe
Pi <- (rowSums(N^2) - m)/(m*(m-1))   # concordância dentro de cada folha
Pbar <- mean(Pi); Pbar
(Pbar - Pe)/(1 - Pe)

# Erro-padrão de Fleiss (válido sob H0: kappa = 0) e teste
num <- 2*(Pe - (2*m-3)*Pe^2 + 2*(m-2)*sum(pj^3))
ep0 <- sqrt(num/(n*m*(m-1)))/(1-Pe); ep0
kap <- (Pbar - Pe)/(1 - Pe); kap/ep0
2*pnorm(abs(kap/ep0), lower.tail = FALSE)

# Para o INTERVALO, use bootstrap das unidades (o EP acima só vale sob H0)
set.seed(7)
bs <- replicate(5000, {
  idx <- sample(n, replace = TRUE)
  kappam.fleiss(sev[idx, ])$value
})
quantile(bs, c(0.025, 0.975)); sd(bs)

# Quem derruba a concordância: deixa-um-de-fora
sapply(1:m, function(j) kappam.fleiss(sev[, -j])$value)
colMeans(sev)                        # o avaliador C é o mais severo

# O efeito de colapsar categorias
s3 <- ifelse(sev <= 2, 1, ifelse(sev == 3, 2, 3))
s2 <- ifelse(sev <= 3, 1, 2)
c(cinco = kappam.fleiss(sev)$value,
  tres  = kappam.fleiss(s3)$value,
  duas  = kappam.fleiss(s2)$value)

# Alternativas para o mesmo delineamento
library(DescTools)
KappaM(sev, method = "Fleiss", conf.level = 0.95)
# irrCAC::gwet.ac1.raw(as.data.frame(sev))      # AC1 de Gwet
# irr::kripp.alpha(t(sev), method = "nominal")  # alfa de Krippendorff

# Gráficos
barplot(colMeans(sev), ylab = "Classe média atribuída", col = "darkseagreen")
boxplot(as.vector(sev) ~ rep(colnames(sev), each = nrow(sev)),
        xlab = "Avaliador", ylab = "Classe")
