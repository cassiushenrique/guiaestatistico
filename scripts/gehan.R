# Teste generalizado de Wilcoxon-Gehan: exemplo completo ----
# Pacotes: survival (base/recomendado), survminer (gráficos)

library(survival)

# PARTE 1 - os dados de Freireich, os mesmos das duas fichas anteriores
tempo  <- c(6,6,6,6,7,9,10,10,11,13,16,17,19,20,22,23,25,32,32,34,35,
            1,1,2,2,3,4,4,5,5,8,8,8,8,11,11,12,12,15,17,22,23)
evento <- c(0,1,1,1,1,0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
            rep(1, 21))
grupo  <- factor(rep(c("6-MP", "placebo"), each = 21))
dados  <- data.frame(tempo, evento, grupo)

# O teste de Gehan, com peso igual ao número em risco.
# No R, a forma direta é pelo pacote coin; com survdiff, o mais próximo
# disponível é o Peto-Peto (rho = 1), que usa S(t-) como peso.
survdiff(Surv(tempo, evento) ~ grupo, data = dados, rho = 0)     # log-rank
survdiff(Surv(tempo, evento) ~ grupo, data = dados, rho = 1)     # Peto-Peto

# install.packages("coin")
library(coin)
logrank_test(Surv(tempo, evento) ~ grupo, data = dados, type = "Gehan-Breslow")
logrank_test(Surv(tempo, evento) ~ grupo, data = dados, type = "Tarone-Ware")
logrank_test(Surv(tempo, evento) ~ grupo, data = dados, type = "Peto-Peto")
logrank_test(Surv(tempo, evento) ~ grupo, data = dados, type = "logrank")

# A estatística de Gehan na forma ORIGINAL: contagem de pares
U <- 0; indecisos <- 0
for (i in which(grupo == "6-MP")) for (j in which(grupo == "placebo")) {
  ti <- tempo[i]; ei <- evento[i]; tj <- tempo[j]; ej <- evento[j]
  if      (ti > tj && ej == 1)                 U <- U + 1
  else if (tj > ti && ei == 1)                 U <- U - 1
  else if (ti == tj && ei == 0 && ej == 1)     U <- U + 1
  else if (ti == tj && ei == 1 && ej == 0)     U <- U - 1
  else indecisos <- indecisos + 1
}
c(U = U, indecisos = indecisos, total = 21*21)
# Sem censura, essa contagem é exatamente a estatística de Mann-Whitney.

# PARTE 2 - o caso em que a escolha do peso muda a conclusão (dados fictícios)
A_t <- c(1,1,1,1,2,2,3,3,3,3,4,4,5,6,6,7,8,9,9,11,19,19,20,22,24,24,25,31,36,36,36,36)
A_e <- c(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0)
B_t <- c(6,7,9,11,11,12,12,13,13,14,14,14,14,15,16,16,16,18,19,20,20,20,20,21,21,22,24,25,25,28,31,33)
B_e <- rep(1, 32)
cult <- data.frame(tempo = c(A_t, B_t), evento = c(A_e, B_e),
                   cultivar = factor(rep(c("A", "B"), each = 32)))
fit2 <- survfit(Surv(tempo, evento) ~ cultivar, data = cult)
summary(fit2, times = c(3, 6, 12, 18, 24, 30, 36))

survdiff(Surv(tempo, evento) ~ cultivar, data = cult, rho = 0)   # NÃO rejeita
survdiff(Surv(tempo, evento) ~ cultivar, data = cult, rho = 1)   # rejeita
library(coin)
logrank_test(Surv(tempo, evento) ~ cultivar, data = cult, type = "Gehan-Breslow")
logrank_test(Surv(tempo, evento) ~ cultivar, data = cult, type = "Tarone-Ware")

# Quando as curvas se cruzam, relate o tempo médio restrito
print(fit2, rmean = 12)
print(fit2, rmean = 24)
print(fit2, rmean = 36)
# install.packages("survRM2")
library(survRM2)
rmst2(cult$tempo, cult$evento, as.integer(cult$cultivar) - 1, tau = 12)
rmst2(cult$tempo, cult$evento, as.integer(cult$cultivar) - 1, tau = 36)

# Gráficos
library(survminer)
ggsurvplot(fit2, data = cult, conf.int = FALSE, risk.table = TRUE,
           xlab = "Meses", legend.title = "Cultivar")
plot(fit2, fun = "cloglog", col = c("darkgreen", "firebrick"), lwd = 2)
# Linhas que se cruzam: riscos NÃO proporcionais.
