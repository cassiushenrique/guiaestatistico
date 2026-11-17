# Gradient boosting de árvores: exemplo completo ----
# Pacotes: gbm (clássico) e xgboost (opcional, mais rápido)
# Dados fictícios: 600 lotes de café (arquivo cafe.csv, pasta scripts/ do site).
# Baixe o arquivo para a pasta de trabalho, ou leia direto do endereço:
url <- "https://cassiushenrique.github.io/ModelosEstatisticos/scripts/cafe.csv"
cafe <- if (file.exists("cafe.csv")) read.csv("cafe.csv") else read.csv(url)
str(cafe)        # nota, fino (0/1), alt, temp, chuva, somb, nfol, ph
cafe$fino_f <- factor(cafe$fino, levels = c(0, 1), labels = c("nao", "fino"))
table(cafe$fino_f)

library(gbm)
f <- nota ~ alt + temp + chuva + somb + nfol + ph

# ---- 1. Boosting à mão, com tocos (árvores de um corte) e nu = 0,1
library(rpart)
F <- rep(mean(cafe$nota), nrow(cafe)); nu <- 0.1
for (m in 1:300) {
  r <- cafe$nota - F                                   # resíduo atual
  h <- rpart(r ~ alt + temp + chuva + somb + nfol + ph, data = cbind(cafe, r = r),
             control = rpart.control(maxdepth = 1, cp = 0, xval = 0))
  F <- F + nu * predict(h, cafe)
}
mean((cafe$nota - F)^2)                                # EQM de treino

# ---- 2. gbm, com o número de árvores escolhido por validação cruzada
set.seed(1)
gb <- gbm(f, data = cafe, distribution = "gaussian",
          n.trees = 3000, shrinkage = 0.05, interaction.depth = 3,
          bag.fraction = 0.8, cv.folds = 10, n.minobsinnode = 10)
M <- gbm.perf(gb, method = "cv")       # gráfico e melhor número de árvores
M
min(gb$cv.error)                       # EQM validado no melhor M

# ---- 3. A profundidade é a ordem das interações
cv_prof <- sapply(1:6, function(k) {
  set.seed(1)
  g <- gbm(f, data = cafe, distribution = "gaussian", n.trees = 2000,
           shrinkage = 0.05, interaction.depth = k, cv.folds = 10)
  min(g$cv.error) })
plot(1:6, cv_prof, type = "b", xlab = "interaction.depth", ylab = "EQM validado")

# ---- 4. Influência relativa e dependência parcial
summary(gb, n.trees = M)                         # importância
plot(gb, i.var = "alt", n.trees = M)             # efeito da altitude
plot(gb, i.var = c("temp", "somb"), n.trees = M) # a interação, em duas variáveis
interact.gbm(gb, data = cafe, i.var = c("temp", "somb"), n.trees = M)  # estatística H

# ---- 5. Classificação (perda logística = "bernoulli")
set.seed(1)
gbc <- gbm(fino ~ alt + temp + chuva + somb + nfol + ph, data = cafe,
           distribution = "bernoulli", n.trees = 2000, shrinkage = 0.05,
           interaction.depth = 3, cv.folds = 10)
Mc <- gbm.perf(gbc, method = "cv")
p <- predict(gbc, cafe, n.trees = Mc, type = "response")

# ---- 6. xgboost (opcional): a mesma ideia, com regularização e muito mais rápido
# install.packages("xgboost")
library(xgboost)
X <- as.matrix(cafe[, c("alt","temp","chuva","somb","nfol","ph")])
set.seed(1)
cvx <- xgb.cv(params = list(objective = "reg:squarederror", eta = 0.05, max_depth = 3,
                            subsample = 0.8), data = xgb.DMatrix(X, label = cafe$nota),
              nrounds = 2000, nfold = 10, early_stopping_rounds = 50, verbose = 0)
cvx$best_iteration
