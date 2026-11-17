# Bagging (bootstrap aggregation): exemplo completo ----
# Pacotes: rpart (incluído no R) e randomForest
# Dados fictícios: 600 lotes de café (arquivo cafe.csv, pasta scripts/ do site).
# Baixe o arquivo para a pasta de trabalho, ou leia direto do endereço:
url <- "https://cassiushenrique.github.io/ModelosEstatisticos/scripts/cafe.csv"
cafe <- if (file.exists("cafe.csv")) read.csv("cafe.csv") else read.csv(url)
str(cafe)        # nota, fino (0/1), alt, temp, chuva, somb, nfol, ph
cafe$fino_f <- factor(cafe$fino, levels = c(0, 1), labels = c("nao", "fino"))
table(cafe$fino_f)

library(rpart)
f <- nota ~ alt + temp + chuva + somb + nfol + ph

# ---- 1. Bagging à mão: B árvores profundas, cada uma numa reamostra
set.seed(7)
i_te <- sample(nrow(cafe), 200); treino <- cafe[-i_te, ]; teste <- cafe[i_te, ]
B <- 500
P <- sapply(1:B, function(b) {
  i <- sample(nrow(treino), replace = TRUE)
  a <- rpart(f, data = treino[i, ], control = rpart.control(cp = 0, minsplit = 2, minbucket = 1, xval = 0))
  predict(a, teste)
})
eqm <- function(o, p) mean((o - p)^2)
c(uma_arvore = mean(apply(P, 2, function(p) eqm(teste$nota, p))),
  bagging    = eqm(teste$nota, rowMeans(P)))
# EQM do bagging em função de B
plot(sapply(1:B, function(b) eqm(teste$nota, rowMeans(P[, 1:b, drop = FALSE]))),
     type = "l", log = "x", xlab = "B", ylab = "EQM no teste")

# ---- 2. O mesmo com randomForest: bagging é a floresta com mtry = p
library(randomForest)
set.seed(1)
bag <- randomForest(f, data = cafe, mtry = 6, ntree = 500, importance = TRUE)
bag                     # traz o EQM fora da sacola (OOB) e o % de variância explicada
plot(bag)               # EQM OOB em função do número de árvores

# ---- 3. Bagging só ajuda procedimentos instáveis
set.seed(1)
d <- sample(rep(1:10, length.out = nrow(cafe)))
cv <- function(ajusta) {
  e <- numeric(nrow(cafe))
  for (j in 1:10) { tr <- cafe[d != j, ]; te <- cafe[d == j, ]; e[d == j] <- te$nota - ajusta(tr, te) }
  mean(e^2)
}
bag_de <- function(ajuste, B = 200) function(tr, te)
  rowMeans(sapply(1:B, function(b) ajuste(tr[sample(nrow(tr), replace = TRUE), ], te)))
lin <- function(tr, te) predict(lm(f, data = tr), te)
arv <- function(tr, te) predict(rpart(f, data = tr, control = rpart.control(cp = 0, minsplit = 2, xval = 0)), te)
c(linear = cv(lin), linear_bag = cv(bag_de(lin)),
  arvore = cv(arv), arvore_bag = cv(bag_de(arv)))

# ---- 4. Classificação: média das probabilidades das árvores
set.seed(1)
bag_c <- randomForest(fino_f ~ alt + temp + chuva + somb + nfol + ph, data = cafe,
                      mtry = 6, ntree = 500)
bag_c$err.rate[500, "OOB"]   # erro fora da sacola
