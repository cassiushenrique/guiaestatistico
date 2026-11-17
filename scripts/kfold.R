# Validação cruzada k-fold: exemplo completo ----
# Pacotes: caret, boot (cv.glm), rsample/tune (tidymodels)
# install.packages(c("caret", "boot"))

# Dados fictícios: 120 talhões de milho. Produtividade (t/ha) em função da
# dose de nitrogênio, densidade de plantio, matéria orgânica e precipitação.
# O arquivo scripts/milho.csv acompanha o repositório.
milho <- read.csv("milho.csv")
str(milho); summary(milho$y)
colMeans(milho)

# O modelo usado em todas as fichas deste grupo
f_completo <- y ~ N + I(N^2) + densidade + I(densidade^2) + mo + chuva

eqm <- function(obs, prev) mean((obs - prev)^2)
n <- nrow(milho)

# ---- A conta, à mão, com k = 5
set.seed(0)
k <- 5
dobra <- sample(rep(1:k, length.out = n))
erros <- numeric(n)
for (j in 1:k) {
  mm <- lm(f_completo, data = milho[dobra != j, ])
  erros[dobra == j] <- (milho$y[dobra == j] -
                        predict(mm, newdata = milho[dobra == j, ]))^2
}
tapply(erros, dobra, mean)      # EQM de cada dobra
mean(erros)                     # a estimativa
eqm(milho$y, predict(lm(f_completo, data = milho)))   # resubstituição

# ---- Com o pacote boot
library(boot)
g <- glm(f_completo, data = milho)
cv.glm(milho, g, K = 5)$delta     # [1] bruto, [2] com correção de viés
cv.glm(milho, g, K = 10)$delta
cv.glm(milho, g)$delta            # K = n: deixa-um-de-fora

# ---- Com o caret, que é o caminho usual
library(caret)
set.seed(4)
ctrl <- trainControl(method = "cv", number = 10)
train(f_completo, data = milho, method = "lm", trControl = ctrl)

# Validação cruzada REPETIDA: elimina a variabilidade do sorteio
ctrl_r <- trainControl(method = "repeatedcv", number = 10, repeats = 20)
train(f_completo, data = milho, method = "lm", trControl = ctrl_r)

# ---- A escolha de k
for (k in c(2, 3, 5, 10, 20, 60, n)) {
  v <- replicate(if (k < n) 200 else 1, {
    d <- sample(rep(1:k, length.out = n)); e <- numeric(n)
    for (j in 1:k) {
      mm <- lm(f_completo, data = milho[d != j, ])
      e[d == j] <- (milho$y[d == j] - predict(mm, newdata = milho[d == j, ]))^2
    }
    mean(e) })
  cat(sprintf("k = %3d: EQM %.4f, DP entre sorteios %.4f\n", k, mean(v), sd(v)))
}

# ---- Comparando os seis modelos candidatos
cands <- list("so N"      = y ~ N,
              "N+dens"    = y ~ N + densidade,
              "4 lineares"= y ~ N + densidade + mo + chuva,
              "com N2"    = y ~ N + I(N^2) + densidade + mo + chuva,
              "com dens2" = y ~ N + densidade + I(densidade^2) + mo + chuva,
              "ambos"     = f_completo)
set.seed(4)
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 20)
res <- lapply(cands, function(f) train(f, data = milho, method = "lm",
                                       trControl = ctrl))
sapply(res, function(m) c(RMSE = m$results$RMSE, EQM = m$results$RMSE^2,
                          SD = m$results$RMSESD))

# Regra de UM ERRO-PADRÃO: entre os modelos cujo EQM não excede
# (menor EQM + 1 erro-padrão), escolha o mais simples.

# ---- Validação cruzada ANINHADA (seleção DENTRO do laço)
set.seed(31)
kext <- 5; dext <- sample(rep(1:kext, length.out = n)); erros <- numeric(n)
for (j in 1:kext) {
  tr <- milho[dext != j, ]; te <- milho[dext == j, ]
  # laço INTERNO: escolhe o modelo usando SÓ o treino externo
  eint <- sapply(cands, function(f) {
    gg <- glm(f, data = tr); cv.glm(tr, gg, K = 5)$delta[1] })
  melhor <- cands[[which.min(eint)]]
  mm <- lm(melhor, data = tr)
  erros[dext == j] <- (te$y - predict(mm, newdata = te))^2
}
mean(erros)    # estimativa HONESTA, que inclui o custo da seleção

# ---- Séries temporais e dados agrupados pedem outros esquemas
# trainControl(method = "timeslice", initialWindow = 60, horizon = 12)
# groupKFold(milho$fazenda, k = 5)
