# Seleção exaustiva de subconjuntos: exemplo completo ----
# Pacotes: leaps (regsubsets)
# Dados fictícios: 90 amostras de solo, 12 candidatas.
# O arquivo solo.csv acompanha o site, na pasta scripts/.
solo <- read.csv("solo.csv")
str(solo)          # y (t/ha) e as 12 candidatas
dim(solo)
# O modelo VERDADEIRO (conhecido por construção) usa só quatro:
#   y = 4,00 + 0,80*mo + 0,0250*pres + 0,0300*v + 0,0200*argila + erro
cand <- c("ph","mo","pres","k","ca","mg","al","ctc","v","argila","silte","areia")
verdadeiras <- c("mo","pres","v","argila")

# A colinearidade, que é o problema central do grupo
library(car)
vif(lm(y ~ ., data = solo))
round(cor(solo[, cand]), 2)

# ---- 1. A enumeração exaustiva: todos os 2^12 = 4.096 subconjuntos
library(leaps)
ex <- regsubsets(y ~ ., data = solo, nvmax = 12, really.big = FALSE)
s  <- summary(ex)
s                            # a tabela de asteriscos: o melhor de cada tamanho

# O leaps NÃO testa os 4.096: usa ramificação e poda (Furnival e Wilson, 1974)
# e devolve o mesmo resultado examinando uma fração deles.

# ---- 2. Os quatro critérios, lado a lado
data.frame(tam = 1:12, R2 = round(s$rsq, 4), R2aj = round(s$adjr2, 4),
           Cp = round(s$cp, 2), BIC = round(s$bic, 2))

which.max(s$adjr2)           # o tamanho que o R² ajustado escolhe
which.min(s$bic)             # o que o BIC escolhe
# Para o Cp a regra é o MENOR modelo com Cp <= p (e NÃO "o Cp mais próximo
# de p": o Cp do modelo completo é sempre exatamente p, de modo que essa
# leitura escolheria trivialmente o modelo completo).
min(which(s$cp <= (1:12) + 1))

# As variáveis de cada escolha
coef(ex, which.max(s$adjr2))
coef(ex, which.min(s$bic))
coef(ex, min(which(s$cp <= (1:12) + 1)))

# ---- 3. O modelo escolhido recupera o VERDADEIRO?
names(coef(ex, 4))[-1]
verdadeiras
# A areia entra no lugar da argila, porque argila = 100 - silte - areia
cor(solo$argila, solo$areia)
cor(solo$argila, solo$silte + solo$areia)

# ---- 4. A MALDIÇÃO DO VENCEDOR: y permutado, nenhum sinal
set.seed(5)
inf <- t(replicate(200, {
  yp <- sample(solo$y)
  sp <- summary(regsubsets(yp ~ ., data = solo[, cand], nvmax = 6))
  sp$rsq[1:6] }))
rbind(esperado = (1:6)/(nrow(solo) - 1), melhor = colMeans(inf))
# O R² esperado de UM subconjunto fixo de tamanho k é k/(n-1). O do MELHOR
# de todos os subconjuntos desse tamanho é MUITO maior. A diferença é
# inflação vinda da própria busca.

# ---- 5. INSTABILIDADE: a escolha muda de amostra para amostra
set.seed(11)
cont <- numeric(12); names(cont) <- cand
for (b in 1:500) {
  i  <- sample(nrow(solo), replace = TRUE)
  sb <- summary(regsubsets(y ~ ., data = solo[i, ], nvmax = 4))
  sel <- cand[sb$which[4, -1]]
  cont[sel] <- cont[sel] + 1
}
sort(round(100*cont/500, 1), decreasing = TRUE)

# ---- 6. O ERRO DE PREDIÇÃO, honestamente: validação cruzada ANINHADA
# A busca precisa ser REFEITA dentro de cada dobra. Avaliar o subconjunto
# escolhido na amostra inteira é otimista.
set.seed(3)
k <- 10; d <- sample(rep(1:k, length.out = nrow(solo)))
er_an <- er_fix <- numeric(nrow(solo))
fixo <- names(coef(ex, 4))[-1]          # escolhido na amostra INTEIRA
for (j in 1:k) {
  tr <- solo[d != j, ]; te <- solo[d == j, ]
  sb <- summary(regsubsets(y ~ ., data = tr, nvmax = 12))
  mel <- which.min(sb$bic)
  vs  <- cand[sb$which[mel, -1]]
  m1  <- lm(reformulate(vs, "y"), data = tr)
  er_an[d == j] <- te$y - predict(m1, newdata = te)
  m2  <- lm(reformulate(fixo, "y"), data = tr)
  er_fix[d == j] <- te$y - predict(m2, newdata = te)
}
mean(er_fix^2)      # OTIMISTA: a escolha viu os dados de teste
mean(er_an^2)       # honesto
mean((solo$y - fitted(lm(reformulate(verdadeiras, "y"), data = solo)))^2)

# ---- 7. Quais subconjuntos cada dobra escolhe?
for (j in 1:k) {
  sb <- summary(regsubsets(y ~ ., data = solo[d != j, ], nvmax = 12))
  cat(sprintf("dobra %2d: %s\n", j,
      paste(cand[sb$which[which.min(sb$bic), -1]], collapse = ", ")))
}
