# Regressão lasso: exemplo completo ----
# Pacotes: glmnet
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

library(glmnet)
Xm <- as.matrix(solo[, cand]); y <- solo$y

# ---- 1. O lambda acima do qual TUDO é zero tem forma fechada
Xp <- scale(Xm); n <- nrow(solo)
max(abs(t(Xp) %*% (y - mean(y))))/n        # lambda_max
# O glmnet constrói a grade a partir desse valor, para baixo.

# ---- 2. O caminho do lasso (alpha = 1 é o padrão)
la <- glmnet(Xm, y, alpha = 1)
plot(la, xvar = "lambda", label = TRUE)
print(la)                                  # Df, %Dev e Lambda por passo

# A ORDEM DE ENTRADA das variáveis
for (j in seq_along(cand)) {
  b <- as.vector(coef(la)[j + 1, ])
  k <- which(abs(b) > 1e-8)[1]
  cat(sprintf("%-8s entra em lambda = %.5f\n", cand[j],
              if (is.na(k)) NA else la$lambda[k]))
}

# ---- 3. A escolha de lambda por validação cruzada
set.seed(1)
cvl <- cv.glmnet(Xm, y, alpha = 1, nfolds = 10)
plot(cvl)
cvl$lambda.min; cvl$lambda.1se
coef(cvl, s = "lambda.min")      # os coeficientes; os zeros aparecem como "."
coef(cvl, s = "lambda.1se")
sum(coef(cvl, s = "lambda.min")[-1] != 0)   # quantas ficaram ativas
verdadeiras                                  # para comparar

# ---- 4. O PROBLEMA DOS GRUPOS CORRELACIONADOS
cor(solo$argila, solo$areia)
cor(solo$argila, solo$silte + solo$areia)    # -0,99: substituto quase perfeito
# A argila ESTÁ no modelo verdadeiro e o lasso escolhe silte e areia.

# Instabilidade do conjunto ativo
set.seed(41)
cont <- numeric(12); names(cont) <- cand; nat <- numeric(1000)
for (b in 1:1000) {
  i  <- sample(nrow(solo), replace = TRUE)
  m  <- glmnet(Xm[i, ], y[i], alpha = 1, lambda = cvl$lambda.min)
  vs <- cand[as.vector(coef(m))[-1] != 0]
  cont[vs] <- cont[vs] + 1; nat[b] <- length(vs)
}
sort(round(100*cont/1000, 1), decreasing = TRUE)
mean(nat); range(nat)

# ---- 5. REDE ELÁSTICA: a correção para grupos correlacionados
set.seed(2)
for (al in c(1, 0.7, 0.4, 0.1)) {
  cj <- cv.glmnet(Xm, y, alpha = al, nfolds = 10)
  vs <- cand[as.vector(coef(cj, s = "lambda.min"))[-1] != 0]
  cat(sprintf("alpha = %.1f: EQM %.4f, %d ativas: %s\n", al,
      min(cj$cvm), length(vs), paste(vs, collapse = ", ")))
}
# Quando alpha diminui, os grupos colineares entram JUNTOS.

# ---- 6. Com p > n, o lasso ativa no máximo n variáveis
# Dados fictícios: espectro NIR, 60 amostras, 200 comprimentos de onda.
# O arquivo espectro.csv acompanha o site, na pasta scripts/.
esp <- read.csv("espectro.csv")
dim(esp)                      # 60 x 201: proteina + nm1100 ... nm2300
Xe <- as.matrix(esp[, -1]); ye <- esp$proteina
qr(scale(Xe))$rank            # 59: o posto é menor que o número de colunas

set.seed(7)
cve <- cv.glmnet(Xe, ye, alpha = 1, nfolds = 10)
cve$lambda.min
sum(coef(cve, s = "lambda.min")[-1] != 0)   # muito menos que 200
min(cve$cvm); 1 - min(cve$cvm)/var(ye)
max(cve$nzero)                               # nunca passa de n = 60

# ---- 7. O LASSO RELAXADO: reajustar por MQO nas selecionadas
vs <- cand[as.vector(coef(cvl, s = "lambda.min"))[-1] != 0]
cbind(lasso    = as.vector(coef(cvl, s = "lambda.min"))[-1][
                    as.vector(coef(cvl, s = "lambda.min"))[-1] != 0],
      relaxado = coef(lm(reformulate(vs, "y"), data = solo))[-1])
# O glmnet faz isso diretamente com relax = TRUE:
cvr <- cv.glmnet(Xm, y, alpha = 1, relax = TRUE)
print(cvr)

# ---- 8. NÃO reporte valores-p do modelo do lasso
# A seleção usou os dados; os p do lm reajustado são inválidos pelas
# mesmas razões da ficha stepwise. Para inferência pós-seleção, veja
# selectiveInference::fixedLassoInf.
