# Teste exato de Fisher: exemplo completo ----
# Pacotes: stats (base). Opcionais: exact2x2 (mid-p e variantes), DescTools
# install.packages(c("exact2x2", "DescTools"))

# PARTE 1 - a prova do chá (Fisher, 1935)
cha <- matrix(c(4, 0, 0, 4), nrow = 2,
              dimnames = list("verdade" = c("leite 1º", "chá 1º"),
                              "declarou" = c("leite 1º", "chá 1º")))
cha
fisher.test(cha, alternative = "greater")     # unilateral: p = 1/70 = 0,0143
fisher.test(cha)                               # bilateral
choose(4, 4) * choose(4, 0) / choose(8, 4)     # a conta, à mão

# a distribuição completa do número de acertos
data.frame(acertos = 0:4,
           combinacoes = choose(4, 0:4) * choose(4, 4:0),
           prob = dhyper(0:4, 4, 4, 4))
fisher.test(matrix(c(3, 1, 1, 3), 2), alternative = "greater")$p.value   # 3 acertos: 0,243

# PARTE 2 - contagens pequenas: 23 lotes de sementes, dois fungicidas (dados fictícios)
tab <- matrix(c(2, 8, 10, 3), nrow = 2,
              dimnames = list(fungicida = c("A", "B"),
                              doenca = c("com", "sem")))
tab
prop.table(tab, 1)

chisq.test(tab)                 # o R avisa: aproximação pode ser incorreta
chisq.test(tab)$expected        # a menor esperada é 4,78
chisq.test(tab, correct = FALSE)
fisher.test(tab)                # o teste exato

# De onde vem o valor-p: as margens são FIXAS e só há 11 tabelas possíveis
n1 <- sum(tab[1, ]); n2 <- sum(tab[2, ]); m1 <- sum(tab[, 1]); N <- sum(tab)
a <- max(0, m1 - n2):min(n1, m1)
prob <- dhyper(a, n1, n2, m1)
obs  <- dhyper(tab[1, 1], n1, n2, m1)
data.frame(a, prob, inclui = prob <= obs * (1 + 1e-7))
sum(prob[prob <= obs * (1 + 1e-7)])            # = o valor-p bilateral

# Razão de chances: a condicional (que o R devolve) e a amostral
fisher.test(tab)$estimate                      # condicional, por máxima verossimilhança
tab[1,1]*tab[2,2] / (tab[1,2]*tab[2,1])        # amostral
fisher.test(tab)$conf.int

# PARTE 3 - mid-p, menos conservador
library(exact2x2)
fisher.exact(tab, midp = TRUE)
# ou, à mão: p_exato - 0,5 * P(tabela observada)

# PARTE 4 - tabelas maiores que 2x2
tab3 <- matrix(c(8, 4, 1, 3, 6, 4, 1, 2, 7), nrow = 3, byrow = TRUE,
               dimnames = list(cultivar = c("A", "B", "C"),
                               nota = c("baixa", "média", "alta")))
tab3
fisher.test(tab3)                               # enumeração exata
chisq.test(tab3)                                # aproximação, com aviso
chisq.test(tab3, simulate.p.value = TRUE, B = 200000)
# para tabelas grandes, fisher.test pode exigir mais memória:
# fisher.test(tab3, workspace = 2e8)  ou  fisher.test(tab3, simulate.p.value = TRUE)

# Gráfico
par(mfrow = c(1, 2))
barplot(dhyper(0:4, 4, 4, 4), names.arg = 0:4, col = c("firebrick", rep("gray85", 3), "firebrick"),
        xlab = "Acertos", ylab = "Probabilidade", main = "Prova do chá")
barplot(prob, names.arg = a, col = ifelse(prob <= obs*(1+1e-7), "firebrick", "gray85"),
        xlab = "Lotes com doença no fungicida A", ylab = "Probabilidade", main = "Fungicidas")
par(mfrow = c(1, 1))
