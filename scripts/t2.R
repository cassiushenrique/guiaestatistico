# Teste t de Student para dois grupos independentes: exemplo completo ----
# Pacote: apenas o stats (já vem com o R). Dados: PlantGrowth.

# 1. Dados: peso seco de plantas (g) no grupo controle e no tratamento 2
dados <- subset(PlantGrowth, group %in% c("ctrl", "trt2"))
dados$group <- droplevels(dados$group)
aggregate(weight ~ group, data = dados, FUN = function(v) c(n = length(v), media = mean(v), dp = sd(v)))

# 2. Diagnóstico gráfico
boxplot(weight ~ group, data = dados, col = "gray85", ylab = "Peso seco (g)")

# 3. Pressupostos: normalidade em cada grupo e igualdade de variâncias
tapply(dados$weight, dados$group, function(v) shapiro.test(v)$p.value)
# install.packages("car")
# car::leveneTest(weight ~ group, data = dados)
var.test(weight ~ group, data = dados)        # razão de variâncias (exige normalidade)

# 4. Teste t de Student (supõe variâncias iguais)
t.test(weight ~ group, data = dados, var.equal = TRUE)

# 5. Cálculo manual da variância combinada
g <- split(dados$weight, dados$group)
n1 <- length(g$ctrl); n2 <- length(g$trt2)
s2p <- ((n1 - 1) * var(g$ctrl) + (n2 - 1) * var(g$trt2)) / (n1 + n2 - 2)
ep <- sqrt(s2p * (1/n1 + 1/n2))
c(s2_combinada = s2p, erro_padrao = ep,
  t = (mean(g$ctrl) - mean(g$trt2)) / ep, gl = n1 + n2 - 2)

# 6. Teste t de Welch (não supõe variâncias iguais) — é o padrão do R
t.test(weight ~ group, data = dados)

# 7. Tamanho do efeito
d <- (mean(g$ctrl) - mean(g$trt2)) / sqrt(s2p)
c(d_de_Cohen = d, g_de_Hedges = d * (1 - 3 / (4 * (n1 + n2) - 9)))

# 8. Alternativa não paramétrica
wilcox.test(weight ~ group, data = dados)

# 9. Gráfico das médias com intervalos de confiança
medias <- tapply(dados$weight, dados$group, mean)
erros  <- tapply(dados$weight, dados$group, function(v) qt(0.975, length(v) - 1) * sd(v) / sqrt(length(v)))
bp <- barplot(medias, ylim = c(0, 7), ylab = "Peso seco médio (g)", col = "gray85")
arrows(bp, medias - erros, bp, medias + erros, angle = 90, code = 3, length = 0.1)
