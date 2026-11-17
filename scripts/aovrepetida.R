# ANOVA de medidas repetidas: exemplo completo ----
# Pacotes: stats (base), ez ou rstatix (esfericidade e correções), emmeans (comparações)
# install.packages(c("ez", "rstatix", "emmeans"))

# 1. Dados fictícios: tempo de reação (ms) de 12 participantes em três condições de ruído.
#    É o mesmo conjunto usado na ficha do teste de esfericidade de Mauchly.
tempos <- matrix(c(412, 428, 471, 385, 402, 455, 440, 447, 462, 398, 421, 502,
                   455, 460, 489, 372, 395, 468, 420, 431, 444, 401, 426, 517,
                   433, 441, 470, 390, 409, 446, 447, 452, 495, 408, 430, 483),
                 nrow = 12, byrow = TRUE,
                 dimnames = list(paste0("p", 1:12),
                                 c("silencio", "moderado", "intenso")))
tempos
colMeans(tempos)
apply(tempos, 2, sd)

# 2. Formato longo, que é o exigido pelas funções de modelagem
longo <- data.frame(
  participante = factor(rep(rownames(tempos), times = 3)),
  condicao     = factor(rep(colnames(tempos), each = 12),
                        levels = c("silencio", "moderado", "intenso")),
  tempo        = as.vector(tempos))
str(longo)

# 3. O modelo: o termo Error() declara que as medidas são do mesmo participante
modelo <- aov(tempo ~ condicao + Error(participante / condicao), data = longo)
summary(modelo)

# 4. Esfericidade: teste de Mauchly e correções
#    Rota 1 - com o pacote ez, que entrega tudo em uma tabela
library(ez)
ezANOVA(data = longo, dv = tempo, wid = participante, within = condicao, detailed = TRUE)

#    Rota 2 - com stats, pela abordagem multivariada
mlm <- lm(tempos ~ 1)
desenho <- data.frame(condicao = factor(colnames(tempos), levels = colnames(tempos)))
library(car)
av <- Anova(mlm, idata = desenho, idesign = ~condicao, type = "III")
summary(av, multivariate = FALSE)     # traz Mauchly, GG e HF

#    Rota 3 - com rstatix
library(rstatix)
anova_test(data = longo, dv = tempo, wid = participante, within = condicao)

# 5. Tamanho de efeito
#    eta parcial = SQ_condicao / (SQ_condicao + SQ_erro)
#    eta generalizado leva também a variabilidade entre participantes
# rstatix::anova_test() já devolve o eta generalizado (ges)

# 6. Comparações múltiplas
with(longo, pairwise.t.test(tempo, condicao, paired = TRUE, p.adjust.method = "holm"))
library(emmeans)
emmeans(modelo, ~condicao) |> pairs(adjust = "holm")

# 7. Alternativas
friedman.test(tempos)                                 # não paramétrica
library(lme4)
mm <- lmer(tempo ~ condicao + (1 | participante), data = longo)   # modelo misto
anova(mm)

# 8. Gráficos
par(mfrow = c(1, 2))
matplot(t(tempos), type = "b", pch = 19, lty = 1, col = "gray65", xaxt = "n",
        xlab = "Condição", ylab = "Tempo de reação (ms)")
axis(1, at = 1:3, labels = colnames(tempos))
lines(colMeans(tempos), type = "b", pch = 18, col = "red", lwd = 3, cex = 1.8)
boxplot(tempo ~ condicao, data = longo, col = "gray85",
        xlab = "Condição", ylab = "Tempo de reação (ms)")
par(mfrow = c(1, 1))
