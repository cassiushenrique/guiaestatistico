# Regressão de Cox: exemplo completo ----
# Pacote: survival (acompanha a instalação padrão do R)
library(survival)

# 1. Dados: ensaio de confiabilidade com 30 bombas centrífugas (dados fictícios)
dados <- data.frame(
  tempo = c(1347, 790, 748, 1424, 827, 744, 265, 811, 257, 247, 557, 802, 59, 444, 655,
            266, 678, 932, 136, 249, 23, 83, 80, 1151, 24, 477, 46, 326, 378, 332),
  falha = c(0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
  carga = factor(rep(c("normal", "severa"), each = 15)),
  temp  = c(65, 67, 72, 70, 72, 72, 81, 58, 77, 80, 76, 67, 83, 56, 79,
            73, 56, 65, 66, 60, 74, 68, 76, 66, 57, 76, 77, 57, 69, 69))
dados$temp_c <- dados$temp - 70        # temperatura centrada em 70 °C
table(dados$carga, dados$falha)

# 2. Kaplan-Meier por grupo e teste log-rank
km <- survfit(Surv(tempo, falha) ~ carga, data = dados)
summary(km, times = c(250, 500, 750, 1000))
survdiff(Surv(tempo, falha) ~ carga, data = dados)

# 3. Modelo de Cox
modelo <- coxph(Surv(tempo, falha) ~ carga + temp_c, data = dados)
summary(modelo)

# 4. Razões de risco para variações relevantes
exp(10 * coef(modelo)["temp_c"])                   # efeito de +10 °C
exp(coef(modelo)["cargasevera"] + 20 * coef(modelo)["temp_c"])

# 5. Curvas de sobrevivência previstas pelo modelo
perfis <- data.frame(carga = factor(c("normal", "normal", "severa", "severa"),
                                    levels = c("normal", "severa")),
                     temp_c = c(-10, 10, -10, 10))
plot(survfit(modelo, newdata = perfis), col = 1:4, lwd = 2,
     xlab = "Tempo (h)", ylab = "Sobrevivência estimada")
legend("topright", c("normal 60 °C", "normal 80 °C", "severa 60 °C", "severa 80 °C"),
       col = 1:4, lwd = 2, bty = "n")

# 6. Verificação de riscos proporcionais (resíduos de Schoenfeld)
zph <- cox.zph(modelo)
print(zph)
plot(zph)

# 7. Observações influentes (resíduos dfbeta)
plot(residuals(modelo, type = "dfbeta")[, 1], ylab = "dfbeta (carga)"); abline(h = 0, col = "red")
