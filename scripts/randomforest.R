# Random forests: exemplo completo ----
# Pacotes: randomForest (ou ranger, mais rápido) e pdp (dependência parcial, opcional)
# Dados fictícios: 600 lotes de café (arquivo cafe.csv, pasta scripts/ do site).
# Baixe o arquivo para a pasta de trabalho, ou leia direto do endereço:
url <- "https://cassiushenrique.github.io/ModelosEstatisticos/scripts/cafe.csv"
cafe <- if (file.exists("cafe.csv")) read.csv("cafe.csv") else read.csv(url)
str(cafe)        # nota, fino (0/1), alt, temp, chuva, somb, nfol, ph
cafe$fino_f <- factor(cafe$fino, levels = c(0, 1), labels = c("nao", "fino"))
table(cafe$fino_f)

library(randomForest)
f <- nota ~ alt + temp + chuva + somb + nfol + ph

# ---- 1. A floresta padrão (mtry = p/3 na regressão, raiz de p na classificação)
set.seed(1)
rf <- randomForest(f, data = cafe, ntree = 500, importance = TRUE)
rf                       # EQM OOB e % de variância explicada
plot(rf)                 # erro OOB em função do número de árvores

# ---- 2. O único parâmetro que importa: mtry
oob <- sapply(1:6, function(m) {
  set.seed(1); tail(randomForest(f, data = cafe, mtry = m, ntree = 500)$mse, 1) })
plot(1:6, oob, type = "b", xlab = "mtry", ylab = "EQM OOB")
# tuneRF() faz uma busca automática
set.seed(1); tuneRF(cafe[, c("alt","temp","chuva","somb","nfol","ph")], cafe$nota, ntreeTry = 300)

# ---- 3. Importância das variáveis: permutação (%IncMSE) e impureza (IncNodePurity)
importance(rf)
varImpPlot(rf)
# Calibrando o olho: acrescente colunas de puro ruído e veja a importância delas
set.seed(4)
cafe2 <- transform(cafe, ruido = rnorm(nrow(cafe)), codigo = sample(0:499, nrow(cafe), TRUE))
rf2 <- randomForest(nota ~ alt + temp + chuva + somb + nfol + ph + ruido + codigo,
                    data = cafe2, mtry = 3, importance = TRUE)
importance(rf2)

# ---- 4. Dependência parcial: a forma do efeito de cada variável
partialPlot(rf, pred.data = cafe, x.var = "alt")
partialPlot(rf, pred.data = cafe, x.var = "somb")
# Interação: dependência parcial em duas variáveis (pacote pdp)
# install.packages("pdp")
library(pdp)
pd2 <- partial(rf, pred.var = c("temp", "somb"), train = cafe)
plotPartial(pd2)

# ---- 5. Classificação
set.seed(1)
rfc <- randomForest(fino_f ~ alt + temp + chuva + somb + nfol + ph, data = cafe, ntree = 500)
rfc                      # matriz de confusão OOB
head(predict(rfc, type = "prob"))     # probabilidades OOB

# ---- 6. ranger: a mesma floresta, muito mais rápida, com importância por permutação
# install.packages("ranger")
library(ranger)
rg <- ranger(f, data = cafe, num.trees = 500, mtry = 3, importance = "permutation")
rg$prediction.error; sort(rg$variable.importance, decreasing = TRUE)
