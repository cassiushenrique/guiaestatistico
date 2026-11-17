# Teste de Yuen para médias aparadas: exemplo completo ----
# Pacote: WRS2. Instale uma única vez: install.packages("WRS2")
library(WRS2)

# 1. Dados: tempo de ciclo (s) de duas máquinas, 20 medições cada (dados fictícios)
maquina1 <- c(51.6, 49.5, 52.6, 50.8, 49.8, 51.2, 28.4, 48.8, 50.2, 52.0,
              48.0, 53.7, 46.3, 74.8, 48.4, 31.2, 49.2, 47.4, 50.5, 70.6)
maquina2 <- c(30.1, 55.8, 57.2, 33.5, 52.3, 56.5, 78.2, 54.0, 54.4, 56.2,
              58.6, 55.5, 55.2, 56.8, 53.4, 57.6, 59.7, 54.8, 83.0, 58.0)
dados <- data.frame(tempo = c(maquina1, maquina2),
                    maquina = factor(rep(c("M1", "M2"), each = 20)))
rbind(media = tapply(dados$tempo, dados$maquina, mean),
      aparada20 = tapply(dados$tempo, dados$maquina, mean, trim = 0.2),
      mediana = tapply(dados$tempo, dados$maquina, median),
      dp = tapply(dados$tempo, dados$maquina, sd))

# 2. Diagnóstico: caudas pesadas nos dois grupos
par(mfrow = c(1, 2))
boxplot(tempo ~ maquina, data = dados, col = "gray85", ylab = "Tempo de ciclo (s)")
qqnorm(maquina1, pch = 19); qqline(maquina1, col = "red")
par(mfrow = c(1, 1))
tapply(dados$tempo, dados$maquina, function(v) shapiro.test(v)$p.value)

# 3. Teste t de Welch (usa a média e o desvio-padrão completos)
t.test(tempo ~ maquina, data = dados)

# 4. Teste de Yuen com 20% de aparagem
yuen(tempo ~ maquina, data = dados, tr = 0.2)

# 5. Intervalo de confiança e tamanho do efeito robusto
yuen(tempo ~ maquina, data = dados, tr = 0.2)$conf.int
yuen.effect.ci(tempo ~ maquina, data = dados, tr = 0.2)

# 6. Efeito da proporção de aparagem
for (tr in c(0.05, 0.10, 0.20, 0.25)) {
  r <- yuen(tempo ~ maquina, data = dados, tr = tr)
  cat("aparagem", tr, ": t =", round(r$test, 4), " gl =", round(r$df, 3),
      " p =", format.pval(r$p.value, digits = 3), "\n")
}

# 7. Versão por bootstrap, recomendada para amostras pequenas
set.seed(2026)
yuenbt(tempo ~ maquina, data = dados, tr = 0.2, nboot = 2000)

# 8. Alternativa por postos
wilcox.test(tempo ~ maquina, data = dados)
