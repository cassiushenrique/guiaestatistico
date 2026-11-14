# Mauchly (esfericidade): exemplo completo ----
# Pacote: apenas o stats (já vem com o R).

# 1. Dados: tempo de reação (ms) de 12 participantes em três condições de ruído
dados <- data.frame(
  silencio = c(412, 385, 440, 398, 455, 372, 420, 401, 433, 390, 447, 408),
  moderado = c(428, 402, 447, 421, 460, 395, 431, 426, 441, 409, 452, 430),
  intenso  = c(471, 455, 462, 502, 489, 468, 444, 517, 470, 446, 495, 483))
colMeans(dados)

# 2. Variâncias das diferenças entre pares de condições
c(sil_mod = var(dados$silencio - dados$moderado),
  sil_int = var(dados$silencio - dados$intenso),
  mod_int = var(dados$moderado - dados$intenso))

# 3. Teste de Mauchly (modelo multivariado apenas com intercepto)
modelo <- lm(as.matrix(dados) ~ 1)
mauchly.test(modelo, X = ~ 1)

# 4. ANOVA de medidas repetidas com as correções de Greenhouse-Geisser e Huynh-Feldt
anova(modelo, X = ~ 1, test = "Spherical")

# 5. Gráficos: perfis individuais e diferenças entre condições
par(mfrow = c(1, 2))
matplot(t(dados), type = "b", pch = 19, lty = 1, col = "gray50", xaxt = "n",
        xlab = "Condição", ylab = "Tempo de reação (ms)")
axis(1, at = 1:3, labels = names(dados))
lines(colMeans(dados), col = "red", lwd = 3)
boxplot(list("S - M" = dados$silencio - dados$moderado,
             "S - I" = dados$silencio - dados$intenso,
             "M - I" = dados$moderado - dados$intenso),
        col = "gray85", ylab = "Diferença (ms)")
par(mfrow = c(1, 1))
