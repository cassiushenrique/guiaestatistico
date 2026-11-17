# Separação treino-teste (holdout): exemplo completo ----
# Pacotes: caret, rsample (tidymodels); base R também resolve
# install.packages(c("caret", "rsample"))

# Dados fictícios: 120 talhões de milho. Produtividade (t/ha) em função da
# dose de nitrogênio, densidade de plantio, matéria orgânica e precipitação.
# O arquivo scripts/milho.csv acompanha o repositório.
milho <- read.csv("milho.csv")
str(milho); summary(milho$y)
colMeans(milho)

# O modelo usado em todas as fichas deste grupo
f_completo <- y ~ N + I(N^2) + densidade + I(densidade^2) + mo + chuva

# ---- UMA separação 70/30
set.seed(1)
n <- nrow(milho)
tr <- sample(n, round(0.70*n))
treino <- milho[tr, ]; teste <- milho[-tr, ]
c(treino = nrow(treino), teste = nrow(teste))

m <- lm(f_completo, data = treino)
eqm <- function(obs, prev) mean((obs - prev)^2)
eqm(treino$y, predict(m))                     # EQM de treino
eqm(teste$y,  predict(m, newdata = teste))    # EQM de teste
# O de treino é sempre menor: é o otimismo da resubstituição.

# ---- A estimativa depende do sorteio
set.seed(7)
est <- replicate(2000, {
  i <- sample(n, round(0.70*n))
  mm <- lm(f_completo, data = milho[i, ])
  eqm(milho$y[-i], predict(mm, newdata = milho[-i, ]))
})
c(media = mean(est), dp = sd(est), min = min(est), max = max(est))
hist(est, breaks = 50, col = "darkseagreen",
     xlab = "EQM de teste", main = "2.000 separações 70/30")

# ---- A proporção de treino
for (p in c(0.5, 0.6, 0.7, 0.8, 0.9)) {
  v <- replicate(1000, {
    i <- sample(n, round(p*n))
    mm <- lm(f_completo, data = milho[i, ])
    eqm(milho$y[-i], predict(mm, newdata = milho[-i, ]))
  })
  cat(sprintf("treino %2.0f%%: EQM medio %.4f, DP %.4f\n", 100*p, mean(v), sd(v)))
}

# ---- Separação em TRÊS partes, quando há seleção de modelo
set.seed(3)
i <- sample(n)
treino <- milho[i[1:72], ]; valid <- milho[i[73:96], ]; teste <- milho[i[97:120], ]
cands <- list(y ~ N,
              y ~ N + densidade,
              y ~ N + densidade + mo + chuva,
              y ~ N + I(N^2) + densidade + mo + chuva,
              y ~ N + densidade + I(densidade^2) + mo + chuva,
              f_completo)
ev <- sapply(cands, function(f) {
  mm <- lm(f, data = treino); eqm(valid$y, predict(mm, newdata = valid)) })
j <- which.min(ev)
mm <- lm(cands[[j]], data = treino)
c(validacao = ev[j], teste = eqm(teste$y, predict(mm, newdata = teste)))
# O EQM de validação é otimista; o de teste é a estimativa honesta.

# ---- Separação ESTRATIFICADA, com classes raras
library(rsample)
# div <- initial_split(dados, prop = 0.7, strata = classe)
# treino <- training(div); teste <- testing(div)
library(caret)
# idx <- createDataPartition(dados$classe, p = 0.7, list = FALSE)

# ---- A alternativa: validação cruzada (ver as fichas kfold e loocv)
library(caret)
ctrl <- trainControl(method = "cv", number = 10)
train(f_completo, data = milho, method = "lm", trControl = ctrl)
