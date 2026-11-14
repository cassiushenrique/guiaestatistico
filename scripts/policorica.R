# Correlação policórica: exemplo completo ----
# Pacote: psych. Instale uma única vez: install.packages("psych")
library(psych)

# 1. Dados: 250 clientes; satisfação com o produto e avaliação do atendimento
tab <- matrix(c(40, 25, 10,
                20, 45, 30,
                 5, 25, 50), nrow = 3, byrow = TRUE,
              dimnames = list(satisfacao  = c("baixa", "media", "alta"),
                              atendimento = c("ruim", "regular", "bom")))
tab

# 2. Dados brutos: uma linha por cliente, categorias codificadas de 1 a 3
dados <- data.frame(satisfacao  = rep(rep(1:3, each = 3), times = as.vector(t(tab))),
                    atendimento = rep(rep(1:3, times = 3), times = as.vector(t(tab))))
table(dados)

# 3. Correlação policórica (rho) e limiares (tau)
polychoric(dados)

# 4. Comparação com coeficientes calculados sobre os códigos
c(pearson  = cor(dados$satisfacao, dados$atendimento),
  spearman = cor(dados$satisfacao, dados$atendimento, method = "spearman"),
  kendall  = cor(dados$satisfacao, dados$atendimento, method = "kendall"))

# 5. Teste de associação
chisq.test(tab)

# 6. Erro-padrão e teste do modelo normal latente (pacote polycor)
# install.packages("polycor")
# polycor::polychor(tab, ML = TRUE, std.err = TRUE)

# 7. Gráfico da tabela
mosaicplot(tab, color = c("gray85", "darkseagreen", "darkgreen"), main = "Satisfação e atendimento")
