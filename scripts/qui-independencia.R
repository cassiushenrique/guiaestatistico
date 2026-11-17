# Qui-quadrado de independência: exemplo completo ----
# Pacotes: stats (base), vcd (gráficos e medidas), DescTools
# install.packages(c("vcd", "DescTools"))

# 1. Dados: UCBAdmissions, admissões à pós-graduação em Berkeley (1973)
data(UCBAdmissions)
dimnames(UCBAdmissions)
agregada <- margin.table(UCBAdmissions, c(2, 1))   # sexo x situação
agregada
prop.table(agregada, 1)                            # taxas por sexo

# 2. O teste na tabela agregada
chisq.test(agregada)                     # com correção de Yates (padrão em 2x2)
chisq.test(agregada, correct = FALSE)    # sem correção

teste <- chisq.test(agregada, correct = FALSE)
teste$expected                           # frequências esperadas
round(teste$residuals, 3)                # resíduos de Pearson
round(teste$stdres, 3)                   # resíduos padronizados ajustados

# 3. Medidas de associação
library(DescTools)
CramerV(agregada, conf.level = 0.95)
Phi(agregada)
OddsRatio(agregada, conf.level = 0.95)
RelRisk(agregada, conf.level = 0.95)
fisher.test(agregada)                    # exato, para comparar

# 4. A MESMA pergunta, dentro de cada departamento
apply(UCBAdmissions, 3, function(t) {
  tt <- t(t)                             # linhas = sexo
  c(taxa_H = tt[1,1]/sum(tt[1,]), taxa_M = tt[2,1]/sum(tt[2,]),
    RC = tt[1,1]*tt[2,2]/(tt[1,2]*tt[2,1]),
    p = chisq.test(tt, correct = FALSE)$p.value)
})

# 5. A associação comum, controlando o departamento
mantelhaen.test(UCBAdmissions)           # estatística e RC de Mantel-Haenszel
# teste de homogeneidade das razões de chances (Breslow-Day)
library(DescTools)
BreslowDayTest(UCBAdmissions)

# 6. A explicação: taxa de admissão do departamento x proporção de candidatas
taxa  <- apply(UCBAdmissions, 3, function(t) sum(t["Admitted",]) / sum(t))
pmulh <- apply(UCBAdmissions, 3, function(t) sum(t[, "Female"]) / sum(t))
cbind(taxa_admissao = round(100*taxa, 2), perc_mulheres = round(100*pmulh, 2))
cor(taxa, pmulh)

# 7. Modelo log-linear, que generaliza o teste
longo <- as.data.frame(UCBAdmissions)
mod_ind <- glm(Freq ~ Gender + Admit, data = longo, family = poisson)       # independência
mod_sat <- glm(Freq ~ Gender * Admit, data = longo, family = poisson)       # saturado
anova(mod_ind, mod_sat, test = "Chisq")        # equivale ao teste de independência
mod_dep <- glm(Freq ~ Dept * Gender + Dept * Admit, data = longo, family = poisson)
mod_tudo <- glm(Freq ~ Dept * Gender + Dept * Admit + Gender * Admit, data = longo, family = poisson)
anova(mod_dep, mod_tudo, test = "Chisq")       # associação sexo-admissão, dado o depto

# 8. Gráficos
library(vcd)
mosaic(agregada, shade = TRUE, legend = TRUE)
mosaic(~ Gender + Admit | Dept, data = UCBAdmissions, shade = TRUE)
assoc(agregada, shade = TRUE)
