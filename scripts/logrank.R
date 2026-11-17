# Teste log-rank: exemplo completo ----
# Pacotes: survival (base/recomendado), survminer (gráficos)

library(survival)
tempo  <- c(6,6,6,6,7,9,10,10,11,13,16,17,19,20,22,23,25,32,32,34,35,
            1,1,2,2,3,4,4,5,5,8,8,8,8,11,11,12,12,15,17,22,23)
evento <- c(0,1,1,1,1,0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
            rep(1, 21))
grupo  <- factor(rep(c("6-MP", "placebo"), each = 21))
dados  <- data.frame(tempo, evento, grupo)

# O teste
survdiff(Surv(tempo, evento) ~ grupo, data = dados)
# N, Observed, Expected, (O-E)^2/E e (O-E)^2/V, com o qui-quadrado e o valor-p

sd <- survdiff(Surv(tempo, evento) ~ grupo, data = dados)
sd$obs; sd$exp                       # observados e esperados por grupo
sd$chisq
pchisq(sd$chisq, df = 1, lower.tail = FALSE)

# A tabela, tempo a tempo, feita à mão
tev <- sort(unique(dados$tempo[dados$evento == 1]))
tab <- t(sapply(tev, function(x) {
  n  <- sum(dados$tempo >= x);  d  <- sum(dados$tempo == x & dados$evento == 1)
  n1 <- sum(dados$tempo >= x & dados$grupo == "6-MP")
  d1 <- sum(dados$tempo == x & dados$evento == 1 & dados$grupo == "6-MP")
  E1 <- d * n1 / n
  V1 <- d * (n1/n) * (1 - n1/n) * (n - d) / max(n - 1, 1)
  c(t = x, n = n, d = d, n1 = n1, d1 = d1, E1 = E1, dif = d1 - E1, V = V1)
}))
round(tab, 4)
colSums(tab[, c("d", "d1", "E1", "dif", "V")])
sum(tab[, "dif"])^2 / sum(tab[, "V"])            # a estatística Q

# Razão de riscos
sd$obs / sd$exp                                  # O/E por grupo
(sd$obs[2]/sd$exp[2]) / (sd$obs[1]/sd$exp[1])    # razão O/E
# Estimativa de Peto, a partir do próprio log-rank:
OE <- sum(tab[, "dif"]); V <- sum(tab[, "V"])
exp(-OE/V); exp(-OE/V + c(-1, 1) * 1.96 / sqrt(V))
# Comparação com a regressão de Cox (verossimilhança parcial):
summary(coxph(Surv(tempo, evento) ~ grupo, data = dados))

# A família de testes ponderados: rho = 0 é o log-rank, rho = 1 é Peto-Peto
survdiff(Surv(tempo, evento) ~ grupo, data = dados, rho = 0)
survdiff(Surv(tempo, evento) ~ grupo, data = dados, rho = 1)
survdiff(Surv(tempo, evento) ~ grupo, data = dados, rho = 0.5)

# Mais de dois grupos e versão estratificada
# survdiff(Surv(tempo, evento) ~ grupo + strata(centro), data = dados)

# Verificar riscos proporcionais, que é quando o log-rank tem poder máximo
fit <- survfit(Surv(tempo, evento) ~ grupo, data = dados)
plot(fit, fun = "cloglog", col = c("darkgreen", "firebrick"), lwd = 2,
     xlab = "Semanas (log)", ylab = "log(-log S(t))")
# Linhas aproximadamente paralelas indicam riscos proporcionais.
cox.zph(coxph(Surv(tempo, evento) ~ grupo, data = dados))

library(survminer)
ggsurvplot(fit, data = dados, pval = TRUE, pval.method = TRUE,
           risk.table = TRUE, conf.int = TRUE, xlab = "Semanas")
