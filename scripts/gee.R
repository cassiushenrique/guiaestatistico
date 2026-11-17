# Equações de estimação generalizadas (GEE): exemplo completo ----
# Pacote: geepack. Instale uma única vez: install.packages("geepack")
library(geepack)

# 1. Dados: 30 operadores executam as mesmas 4 tarefas (dados fictícios)
treino_operador <- c(1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0,
                     1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0)
sucesso <- c(1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
             1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0,
             1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0,
             1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1)
dados <- data.frame(
  operador    = factor(rep(1:30, each = 4)),
  treinamento = rep(treino_operador, each = 4),
  dificuldade = rep(1:4, times = 30) - 2.5,
  sucesso     = sucesso)
xtabs(~ treinamento + sucesso, data = dados)

# 2. Análise ingênua: logística que ignora o agrupamento por operador
m_naive <- glm(sucesso ~ treinamento + dificuldade, data = dados, family = binomial)
summary(m_naive)

# 3. GEE com correlação de trabalho permutável (exchangeable)
#    IMPORTANTE: os dados precisam estar ordenados por unidade (id)
m_gee <- geeglm(sucesso ~ treinamento + dificuldade, id = operador,
                data = dados, family = binomial, corstr = "exchangeable")
summary(m_gee)

# 4. Correlação de trabalho estimada
summary(m_gee)$corr

# 5. Razões de chances e intervalos de confiança de 95%
b <- coef(m_gee); ep <- summary(m_gee)$coefficients[, 2]
round(exp(cbind(OR = b, inferior = b - 1.96 * ep, superior = b + 1.96 * ep)), 4)

# 6. Outras estruturas de trabalho e comparação por QIC
m_ind <- geeglm(sucesso ~ treinamento + dificuldade, id = operador,
                data = dados, family = binomial, corstr = "independence")
m_ar1 <- geeglm(sucesso ~ treinamento + dificuldade, id = operador,
                data = dados, family = binomial, corstr = "ar1")
rbind(independencia = QIC(m_ind), permutavel = QIC(m_gee), ar1 = QIC(m_ar1))

# 7. Comparação com o modelo misto (efeito aleatório por operador)
# install.packages("lme4")
# m_misto <- lme4::glmer(sucesso ~ treinamento + dificuldade + (1 | operador),
#                        data = dados, family = binomial)
# O GEE estima o efeito MÉDIO NA POPULAÇÃO; o misto, o efeito PARA UM MESMO operador.

# 8. Gráfico: taxas observadas
taxas <- tapply(dados$sucesso, list(dados$dificuldade, dados$treinamento), mean)
matplot(1:4, taxas, type = "b", pch = 19, lty = 1, lwd = 2, ylim = c(0, 1),
        xlab = "Dificuldade da tarefa", ylab = "Proporção de sucesso")
legend("bottomleft", c("sem treinamento", "com treinamento"), col = 1:2, lwd = 2, bty = "n")
