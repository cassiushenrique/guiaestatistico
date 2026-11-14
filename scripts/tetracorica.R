# Correlação tetracórica: exemplo completo ----
# Pacote: psych. Instale uma única vez: install.packages("psych")
library(psych)

# 1. Dados: 200 alunos, aprovação (nota >= 60) em duas provas
tab <- matrix(c(70, 30,
                20, 80), nrow = 2, byrow = TRUE,
              dimnames = list(Prova1 = c("aprovado", "reprovado"),
                              Prova2 = c("aprovado", "reprovado")))
tab
prop.table(tab)

# 2. Correlação tetracórica (rho) e limiares (tau)
tetrachoric(tab)

# 3. Comparação com o coeficiente phi e a razão de chances
n11 <- tab[1, 1]; n10 <- tab[1, 2]; n01 <- tab[2, 1]; n00 <- tab[2, 2]
phi <- (n11 * n00 - n10 * n01) / sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
rc  <- (n11 * n00) / (n10 * n01)
c(phi = phi, razao_de_chances = rc, aprox_cosseno = cos(pi / (1 + sqrt(rc))))

# 4. Teste de associação na tabela
chisq.test(tab, correct = FALSE)

# 5. Com dados brutos (uma linha por aluno, colunas 0/1), a chamada é a mesma
dados <- data.frame(p1 = rep(c(1, 1, 0, 0), c(n11, n10, n01, n00)),
                    p2 = rep(c(1, 0, 1, 0), c(n11, n10, n01, n00)))
tetrachoric(dados)

# 6. Gráfico da tabela
mosaicplot(tab, color = c("darkseagreen", "gray85"), main = "Aprovação nas duas provas")
