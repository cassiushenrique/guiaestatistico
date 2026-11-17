# Estimador de Kaplan-Meier: exemplo completo ----
# Pacotes: survival (base/recomendado), survminer (gráficos)
# install.packages(c("survminer"))

library(survival)

# Freireich et al. (1963): tempo até a recaída, em semanas.
# O sinal + na impressão marca as observações CENSURADAS.
tempo  <- c(6,6,6,6,7,9,10,10,11,13,16,17,19,20,22,23,25,32,32,34,35,
            1,1,2,2,3,4,4,5,5,8,8,8,8,11,11,12,12,15,17,22,23)
evento <- c(0,1,1,1,1,0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
            rep(1, 21))
grupo  <- factor(rep(c("6-MP", "placebo"), each = 21))
dados  <- data.frame(tempo, evento, grupo)
table(dados$grupo, dados$evento)

# O objeto Surv: tempo + indicador de evento
Surv(dados$tempo, dados$evento)[1:10]

# O ajuste
fit <- survfit(Surv(tempo, evento) ~ grupo, data = dados)
fit                       # n, eventos, mediana e IC da mediana
summary(fit)              # a tabela de vida completa, linha a linha
summary(fit, times = c(5, 10, 15, 20, 23))   # S(t) em tempos escolhidos

# A conta à mão, no grupo 6-MP
d6 <- subset(dados, grupo == "6-MP")
tev <- sort(unique(d6$tempo[d6$evento == 1]))
nj  <- sapply(tev, function(x) sum(d6$tempo >= x))
dj  <- sapply(tev, function(x) sum(d6$tempo == x & d6$evento == 1))
data.frame(tempo = tev, n = nj, d = dj,
           fator = round(1 - dj/nj, 4), S = round(cumprod(1 - dj/nj), 4))

# Erro-padrão de Greenwood e intervalos de confiança
summary(survfit(Surv(tempo, evento) ~ 1, data = d6))$std.err
survfit(Surv(tempo, evento) ~ grupo, data = dados, conf.type = "log")      # padrão
survfit(Surv(tempo, evento) ~ grupo, data = dados, conf.type = "log-log")
survfit(Surv(tempo, evento) ~ grupo, data = dados, conf.type = "plain")

# Quantis de sobrevivência, com intervalo
quantile(fit, probs = c(0.25, 0.50, 0.75))

# Tempo médio restrito de sobrevivência (RMST) até tau = 23
print(fit, rmean = 23)

# O que acontece quando a censura é tratada errado
median(d6$tempo[d6$evento == 1])          # descartar os censurados
median(d6$tempo)                          # censura tratada como recaída
# A mediana correta, 23, sai do Kaplan-Meier e de mais nenhum desses atalhos.

# Gráficos
plot(fit, col = c("darkgreen", "firebrick"), lwd = 2, mark.time = TRUE,
     xlab = "Semanas", ylab = "S(t)", conf.int = TRUE)
legend("topright", levels(dados$grupo), col = c("darkgreen", "firebrick"), lwd = 2)
abline(h = 0.5, lty = 2)

library(survminer)
ggsurvplot(fit, data = dados, conf.int = TRUE, risk.table = TRUE,
           pval = TRUE, surv.median.line = "hv",
           xlab = "Semanas", legend.title = "Tratamento")
ggsurvplot(fit, data = dados, fun = "cumhaz")   # risco acumulado
