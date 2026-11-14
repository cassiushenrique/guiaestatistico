# Levene–Brown–Forsythe: exemplo completo ----
# Pacote: car. Instale uma única vez: install.packages("car")
library(car)

# 1. Dados: consumo (mpg) por número de cilindros
dados <- mtcars
dados$cil <- factor(dados$cyl)
aggregate(mpg ~ cil, data = dados, FUN = function(v) c(n = length(v), media = mean(v), mediana = median(v), dp = sd(v)))

# 2. Diagnóstico gráfico
boxplot(mpg ~ cil, data = dados, col = "gray85", xlab = "Cilindros", ylab = "mpg")

# 3. Brown–Forsythe (centro = mediana, padrão do pacote car)
leveneTest(mpg ~ cil, data = dados)

# 4. Levene clássico (centro = média)
leveneTest(mpg ~ cil, data = dados, center = mean)

# 5. Cálculo manual: ANOVA sobre os desvios absolutos em torno da mediana
dados$Z <- abs(dados$mpg - ave(dados$mpg, dados$cil, FUN = median))
anova(lm(Z ~ cil, data = dados))

# 6. Bartlett para comparação (sensível a não normalidade)
bartlett.test(mpg ~ cil, data = dados)

# 7. Em modelos: homogeneidade dos resíduos entre grupos
modelo <- lm(mpg ~ cil, data = dados)
leveneTest(modelo)
