# Deixa-um-de-fora (LOOCV): exemplo completo ----
# Pacotes: boot (cv.glm), caret; o atalho sai do R base
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

# ---- Força bruta: n ajustes
e_bruta <- numeric(n)
for (i in 1:n) {
  mm <- lm(f_completo, data = milho[-i, ])
  e_bruta[i] <- (milho$y[i] - predict(mm, newdata = milho[i, ]))^2
}
mean(e_bruta)

# ---- O ATALHO: um ajuste só, pela alavancagem
m <- lm(f_completo, data = milho)
h <- hatvalues(m)                    # diagonal da matriz chapéu
r <- residuals(m)
e_atalho <- (r/(1 - h))^2
mean(e_atalho)
max(abs(e_bruta - e_atalho))         # praticamente zero: é identidade

# A soma das residuais deixa-um-de-fora é a estatística PRESS
PRESS <- sum(e_atalho)*1; PRESS
sum(h)                                # é sempre p
mean(h)                               # p/n
sum(h > 2*length(coef(m))/n)          # observações de alta alavancagem

# As cinco de maior alavancagem
o <- order(h, decreasing = TRUE)[1:5]
data.frame(obs = o, h = round(h[o], 4), residuo = round(r[o], 4),
           res_loo = round(r[o]/(1 - h[o]), 4))

# ---- Com o pacote boot
library(boot)
g <- glm(f_completo, data = milho)
cv.glm(milho, g)$delta               # K = n por padrão
cv.glm(milho, g, K = 10)$delta

# ---- Com o caret
library(caret)
train(f_completo, data = milho, method = "lm",
      trControl = trainControl(method = "LOOCV"))

# ---- Comparação com os demais k
for (k in c(2, 3, 5, 10, 20, 40, n)) {
  v <- if (k == n) mean(e_atalho) else mean(replicate(200, {
    d <- sample(rep(1:k, length.out = n)); e <- numeric(n)
    for (j in 1:k) {
      mm <- lm(f_completo, data = milho[d != j, ])
      e[d == j] <- (milho$y[d == j] - predict(mm, newdata = milho[d == j, ]))^2
    }
    mean(e) }))
  cat(sprintf("k = %3d: EQM %.4f\n", k, v))
}

# ---- Seleção de modelo: LOOCV, AIC e BIC
cands <- list("so N"       = y ~ N,
              "N+dens"     = y ~ N + densidade,
              "4 lineares" = y ~ N + densidade + mo + chuva,
              "com N2"     = y ~ N + I(N^2) + densidade + mo + chuva,
              "com dens2"  = y ~ N + densidade + I(densidade^2) + mo + chuva,
              "ambos"      = f_completo)
t(sapply(cands, function(f) {
  mm <- lm(f, data = milho); hh <- hatvalues(mm); rr <- residuals(mm)
  c(resub = mean(rr^2), loocv = mean((rr/(1 - hh))^2),
    AIC = AIC(mm), BIC = BIC(mm)) }))
# Para modelos lineares, LOOCV e AIC são assintoticamente equivalentes
# (Stone, 1977), e aqui escolhem o mesmo modelo.

# ---- Quando a deixa-um-de-fora FALHA (exemplo construído)
# Classificador que prevê a classe majoritária do treino, em dados
# sem sinal e com classes 50/50: a deixa-um-de-fora estima acurácia ZERO.
set.seed(9)
nn <- 50; yy <- sample(rep(0:1, each = nn/2))
err <- sum(sapply(1:nn, function(i)
  (as.numeric(mean(yy[-i]) > 0.5) != yy[i])))
1 - err/nn       # acurácia estimada pela deixa-um-de-fora
