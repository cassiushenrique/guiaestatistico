# Teste de McNemar: exemplo completo ----
# Pacotes: stats (base), exact2x2 (versão exata), DescTools (kappa, IC)
# install.packages(c("exact2x2", "DescTools"))

# PARTE 1 - o exemplo padrão: duas pesquisas com as MESMAS pessoas
# Fonte: Agresti (1990), p. 350; é o exemplo da documentação de mcnemar.test()
Desempenho <- matrix(c(794, 86, 150, 570), nrow = 2,
  dimnames = list("1ª pesquisa" = c("aprova", "desaprova"),
                  "2ª pesquisa" = c("aprova", "desaprova")))
Desempenho
addmargins(Desempenho)
prop.table(margin.table(Desempenho, 1))      # aprovação na 1ª
prop.table(margin.table(Desempenho, 2))      # aprovação na 2ª

# O teste
mcnemar.test(Desempenho)                     # com correção de continuidade (padrão)
mcnemar.test(Desempenho, correct = FALSE)
b <- Desempenho[1, 2]; c_ <- Desempenho[2, 1]
binom.test(b, b + c_, p = 0.5)               # versão exata

# A conta, à mão
(abs(b - c_) - 1)^2 / (b + c_)               # com correção
(b - c_)^2 / (b + c_)                        # sem correção

# Medidas de efeito
b / c_                                        # razão de chances de mudança
exp(log(b/c_) + c(-1, 1) * 1.96 * sqrt(1/b + 1/c_))
n <- sum(Desempenho)
(b - c_) / n                                  # diferença de proporções
# IC da diferença (método de Wald para dados pareados):
dif <- (b - c_)/n
ep  <- sqrt(b + c_ - (b - c_)^2/n) / n
dif + c(-1, 1) * 1.96 * ep

# Por que o qui-quadrado de independência seria a pergunta errada
chisq.test(Desempenho, correct = FALSE)      # responde a outra coisa

# PARTE 2 - concordância entre dois métodos (dados fictícios)
metodos <- matrix(c(46, 7, 22, 45), nrow = 2,
  dimnames = list("método A" = c("detecta", "não detecta"),
                  "método B" = c("detecta", "não detecta")))
metodos
mcnemar.test(metodos)
mcnemar.test(metodos, correct = FALSE)
binom.test(metodos[1,2], metodos[1,2] + metodos[2,1], p = 0.5)   # exato
library(exact2x2)
mcnemar.exact(metodos)

# Concordância, que é outra pergunta
library(DescTools)
CohenKappa(metodos, conf.level = 0.95)
sum(diag(metodos)) / sum(metodos)            # concordância bruta

# PARTE 3 - extensões
# Tabelas k x k pareadas: teste de homogeneidade marginal de Stuart-Maxwell
# DescTools::StuartMaxwellTest(tabela)
# Três ou mais medidas binárias: teste Q de Cochran (ver a ficha correspondente)

# Gráficos
par(mfrow = c(1, 2))
barplot(c(Desempenho[1,2], Desempenho[2,1]), names.arg = c("aprova -> desaprova", "desaprova -> aprova"),
        col = c("firebrick", "darkorange"), ylab = "Entrevistados", main = "Só as discordâncias")
mosaicplot(Desempenho, color = TRUE, main = "Tabela completa")
par(mfrow = c(1, 1))
