# Qui-quadrado de aderência: exemplo completo ----
# Pacotes: stats (base). Opcionais: vcd (gráficos), DescTools
# install.packages(c("vcd", "DescTools"))

# PARTE 1 - proporções dadas pela teoria: as ervilhas de Mendel
observado <- c("lisa amarela" = 315, "lisa verde" = 108,
               "rugosa amarela" = 101, "rugosa verde" = 32)
proporcoes <- c(9, 3, 3, 1) / 16
chisq.test(observado, p = proporcoes)

teste <- chisq.test(observado, p = proporcoes)
teste$expected                  # frequências esperadas
teste$residuals                 # resíduos de Pearson: (O - E)/raiz(E)
teste$residuals^2               # contribuição de cada categoria
sum(teste$residuals^2)          # = a estatística

# O teste tem DOIS lados: valores muito pequenos indicam ajuste bom demais
pchisq(teste$statistic, teste$parameter)            # probabilidade de um ajuste PIOR
pchisq(41.6, 84)                # a crítica de Fisher a todos os dados de Mendel

# PARTE 2 - ajuste de uma DISTRIBUIÇÃO, com parâmetro estimado
# Mortes por coice de cavalo no exército prussiano (Bortkiewicz, 1898)
mortes  <- 0:4
corpos  <- c(109, 65, 22, 3, 1)
n <- sum(corpos)
(lambda <- sum(mortes * corpos) / n)          # parâmetro ESTIMADO dos dados
esperado <- c(dpois(0:3, lambda), ppois(3, lambda, lower.tail = FALSE)) * n
round(esperado, 3)
min(esperado)                                  # abaixo de 1: é preciso agrupar

# Agrupando a cauda e corrigindo os graus de liberdade
obs2 <- c(109, 65, 22, 4)
esp2 <- c(dpois(0:2, lambda), ppois(2, lambda, lower.tail = FALSE)) * n
X2 <- sum((obs2 - esp2)^2 / esp2)
gl <- length(obs2) - 1 - 1                     # UM a menos: lambda foi estimado
c(qui_quadrado = X2, gl = gl, p = pchisq(X2, gl, lower.tail = FALSE))
# ATENÇÃO: chisq.test() NÃO desconta o parâmetro estimado. Com
# chisq.test(obs2, p = esp2/n) o valor-p sairia errado (gl = 3 em vez de 2).

# PARTE 3 - um caso que rejeita, com dados fictícios
defeitos <- c(dimensional = 58, acabamento = 41, montagem = 35,
              material = 32, outros = 14)
t3 <- chisq.test(defeitos)                     # H0: as cinco categorias são equiprováveis
t3
t3$expected
round(t3$residuals, 3)                         # ONDE está a discrepância
sqrt(t3$statistic / sum(defeitos))             # w de Cohen

# Valor-p por simulação, sem depender da aproximação
chisq.test(defeitos, simulate.p.value = TRUE, B = 100000)

# Intervalos de confiança simultâneos para as proporções (método de Goodman)
library(DescTools)
MultinomCI(defeitos, conf.level = 0.95, method = "goodman")

# Gráficos
par(mfrow = c(1, 2))
bp <- barplot(rbind(defeitos, t3$expected), beside = TRUE, col = c("gray40", "gray85"),
              legend.text = c("observado", "esperado"), ylab = "Peças refugadas")
barplot(t3$residuals, col = ifelse(abs(t3$residuals) > 2, "firebrick", "gray70"),
        ylab = "Resíduo padronizado", las = 2)
abline(h = c(-2, 2), lty = 2)
par(mfrow = c(1, 1))
