# Análise log-linear: exemplo completo ----
# Pacotes: stats e MASS (base/recomendados), vcd (gráficos de mosaico)
# install.packages(c("vcd"))

# Dados: UCBAdmissions, candidaturas à pós-graduação em Berkeley em 1973,
# classificadas por admissão (A), sexo (S) e departamento (D).
data(UCBAdmissions)
UCBAdmissions
dim(UCBAdmissions)                 # 2 x 2 x 6 = 24 células
sum(UCBAdmissions)                 # 4 526 candidaturas
ftable(UCBAdmissions)

# Formato longo: uma linha por célula, com a contagem
df <- as.data.frame(UCBAdmissions)
names(df) <- c("situacao", "sexo", "dep", "n")
head(df)

# Os modelos log-lineares, ajustados como GLM de Poisson
m0 <- glm(n ~ situacao + sexo + dep, family = poisson, data = df)
m1 <- glm(n ~ situacao * dep + sexo, family = poisson, data = df)
m2 <- glm(n ~ sexo * dep + situacao, family = poisson, data = df)
m3 <- glm(n ~ situacao * dep + sexo * dep, family = poisson, data = df)
m4 <- glm(n ~ (situacao + sexo + dep)^2, family = poisson, data = df)
ms <- glm(n ~ situacao * sexo * dep, family = poisson, data = df)

# Desvios e graus de liberdade de cada um
sapply(list(m0, m1, m2, m3, m4, ms),
       function(m) c(gl = m$df.residual, G2 = deviance(m)))
anova(m3, m4, test = "Chisq")       # o termo situação:sexo vale a pena?

# O coeficiente de interesse: a razão de chances comum
summary(m4)$coefficients["situacaoRejected:sexoFemale", ]
exp(coef(m4)["situacaoRejected:sexoFemale"])

# Seleção automática
library(MASS)
step(ms, direction = "backward", trace = 1)
AIC(m0, m1, m2, m3, m4, ms)
BIC(m0, m1, m2, m3, m4, ms)

# A via clássica equivalente, para tabelas: loglin()
loglin(UCBAdmissions, margin = list(c(1, 3), c(2, 3)))   # modelo (AD, SD)

# Resíduos de Pearson do modelo de independência condicional
r <- residuals(m3, type = "pearson")
cbind(df, ajustado = round(fitted(m3), 2), residuo = round(r, 3))[abs(r) > 1, ]

# Por que a tabela agregada engana
agregada <- margin.table(UCBAdmissions, c(1, 2))
agregada
chisq.test(agregada)                # significativo
prop.table(agregada, 2)
# Mas departamento por departamento:
apply(UCBAdmissions, 3, function(t) round(chisq.test(t)$p.value, 5))
apply(UCBAdmissions, 3, function(t) round((t[1,1]*t[2,2])/(t[1,2]*t[2,1]), 4))

# Gráficos
library(vcd)
mosaic(UCBAdmissions, shade = TRUE, legend = TRUE,
       main = "Resíduos do modelo de independência")
mosaic(~ situacao + sexo | dep, data = UCBAdmissions, shade = TRUE)
fourfoldplot(UCBAdmissions, std = "ind.max")
