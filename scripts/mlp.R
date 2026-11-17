# Rede neural multicamadas (MLP): exemplo completo ----
# Pacote: nnet (incluído no R). Para redes maiores, keras3 ou torch.
# Dados fictícios: 600 lotes de café (arquivo cafe.csv, pasta scripts/ do site).
# Baixe o arquivo para a pasta de trabalho, ou leia direto do endereço:
url <- "https://cassiushenrique.github.io/ModelosEstatisticos/scripts/cafe.csv"
cafe <- if (file.exists("cafe.csv")) read.csv("cafe.csv") else read.csv(url)
str(cafe)        # nota, fino (0/1), alt, temp, chuva, somb, nfol, ph
cafe$fino_f <- factor(cafe$fino, levels = c(0, 1), labels = c("nao", "fino"))
table(cafe$fino_f)

library(nnet)
vars <- c("alt", "temp", "chuva", "somb", "nfol", "ph")
Z <- data.frame(scale(cafe[, vars]), fino_f = cafe$fino_f)

# ---- 1. Uma camada oculta com H neurônios logísticos
set.seed(1)
m <- nnet(fino_f ~ ., data = Z, size = 4, decay = 1, maxit = 2000)
summary(m)                         # todos os pesos: 6x4 + 4 (camada oculta) + 4 + 1 (saída)

# ---- 2. Escolher H (size) e o decaimento (decay) por validação cruzada
set.seed(1)
d <- sample(rep(1:10, length.out = nrow(Z)))
grade <- expand.grid(size = c(1, 2, 4, 8, 16), decay = c(0.0001, 0.01, 0.1, 1, 3))
grade$erro <- apply(grade, 1, function(g) {
  pr <- character(nrow(Z))
  for (j in 1:10) {
    mj <- nnet(fino_f ~ ., data = Z[d != j, ], size = g["size"], decay = g["decay"],
               maxit = 2000, trace = FALSE)
    pr[d == j] <- as.character(predict(mj, Z[d == j, ], type = "class")) }
  mean(pr != Z$fino_f) })
grade[order(grade$erro), ][1:5, ]
# (o caret faz o mesmo com train(method = "nnet", tuneGrid = ...))

# ---- 3. Mínimos locais: o mesmo modelo, dez pontos de partida
perdas <- sapply(1:10, function(s) { set.seed(s)
  nnet(fino_f ~ ., data = Z, size = 16, decay = 0.0001, maxit = 3000, trace = FALSE)$value })
perdas
# Média das probabilidades de várias redes (avNNet, do caret, faz isso)
p_media <- rowMeans(sapply(1:5, function(s) { set.seed(s)
  predict(nnet(fino_f ~ ., data = Z, size = 16, decay = 3, maxit = 3000, trace = FALSE), Z) }))

# ---- 4. Regressão da nota (saída linear) e a interação sombra x temperatura
Zr <- data.frame(scale(cafe[, vars]), nota = cafe$nota)
set.seed(1)
mr <- nnet(nota ~ ., data = Zr, size = 8, decay = 1, linout = TRUE, maxit = 5000)
efeito <- function(T) {
  base <- cafe[, vars]; base$temp <- T
  a <- base; a$somb <- 10; b <- base; b$somb <- 60
  sc <- function(x) scale(x, attr(scale(cafe[, vars]), "scaled:center"),
                             attr(scale(cafe[, vars]), "scaled:scale"))
  mean(predict(mr, sc(b)) - predict(mr, sc(a))) }
c(frio = efeito(18.5), quente = efeito(22.5))
