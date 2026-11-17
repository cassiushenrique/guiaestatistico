# Regressão ridge: exemplo completo ----
# Pacotes: glmnet (Friedman, Hastie e Tibshirani), car (vif)
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

# ---- 1. O diagnóstico: a colinearidade é severa
Xp <- scale(as.matrix(solo[, cand]))
sv <- svd(Xp)$d; sv                       # valores singulares
max(sv)/min(sv)                           # número de condição de X
kappa(Xp)                                 # idem, pelo R
sort(vif(lm(y ~ ., data = solo)), decreasing = TRUE)

# A consequência: coeficientes instáveis
set.seed(31)
B <- t(replicate(2000, {
  i <- sample(nrow(solo), replace = TRUE)
  coef(lm(y ~ ., data = solo[i, ]))[-1] }))
data.frame(MQO = round(coef(lm(y ~ ., data = solo))[-1], 4),
           EP_boot = round(apply(B, 2, sd), 4),
           troca_sinal = round(100*apply(B, 2, function(z)
               min(mean(z > 0), mean(z < 0))), 1))

# ---- 2. O ridge
library(glmnet)
Xm <- as.matrix(solo[, cand]); y <- solo$y
# alpha = 0 é o ridge; o glmnet PADRONIZA por padrão
rg <- glmnet(Xm, y, alpha = 0)
plot(rg, xvar = "lambda", label = TRUE)
print(rg)

# Os graus de liberdade efetivos
gl <- sapply(rg$lambda, function(l) sum(sv^2/(sv^2 + l*nrow(solo))))
cbind(lambda = round(rg$lambda, 5), gl_efetivos = round(gl, 3))[c(1,20,40,60,80,100), ]

# ---- 3. O ENCOLHIMENTO por direção principal
l0 <- 0.03162
data.frame(componente = 1:12, d = round(sv, 3),
           fator = round(sv^2/(sv^2 + l0*nrow(solo)), 4))
# As direções de MAIOR variância quase não são tocadas; as de menor,
# onde a colinearidade concentra o ruído, são eliminadas.

# ---- 4. A escolha de lambda por validação cruzada
set.seed(1)
cvr <- cv.glmnet(Xm, y, alpha = 0, nfolds = 10)
plot(cvr)
cvr$lambda.min            # menor erro
cvr$lambda.1se            # regra do erro-padrão único
coef(cvr, s = "lambda.min")
coef(cvr, s = "lambda.1se")

# Comparação honesta com o MQO (lambda reescolhido em cada dobra)
set.seed(3)
k <- 10; d <- sample(rep(1:k, length.out = nrow(solo)))
er_r <- er_o <- numeric(nrow(solo))
for (j in 1:k) {
  tr <- d != j; te <- d == j
  cj <- cv.glmnet(Xm[tr, ], y[tr], alpha = 0, nfolds = 10)
  er_r[te] <- y[te] - predict(cj, newx = Xm[te, ], s = "lambda.min")
  er_o[te] <- y[te] - predict(lm(y ~ ., data = solo[tr, ]), newdata = solo[te, ])
}
mean(er_r^2); mean(er_o^2)

# ---- 5. NENHUM coeficiente chega a zero
b <- as.vector(coef(cvr, s = "lambda.min"))[-1]
sum(b == 0)                 # zero: o ridge não seleciona
min(abs(b))
# Para zerar coeficientes é preciso a penalidade L1 do lasso.

# ---- 6. O REGIME EM QUE O RIDGE É INDISPENSÁVEL: p > n
# Dados fictícios: espectro NIR, 60 amostras, 200 comprimentos de onda.
# O arquivo espectro.csv acompanha o site, na pasta scripts/.
esp <- read.csv("espectro.csv")
dim(esp)                      # 60 x 201: proteina + nm1100 ... nm2300
Xe <- as.matrix(esp[, -1]); ye <- esp$proteina
qr(scale(Xe))$rank            # 59: o posto é menor que o número de colunas

m_imp <- lm(ye ~ Xe)
sum(is.na(coef(m_imp)))     # 141 coeficientes NA: X'X é singular
summary(m_imp)$r.squared    # 1: ajuste perfeito e inútil

set.seed(7)
cve <- cv.glmnet(Xe, ye, alpha = 0, nfolds = 10)
cve$lambda.min
min(cve$cvm)                               # EQM validado
1 - min(cve$cvm)/var(ye)                   # R² validado
# Com p > n o ridge continua tendo solução única, porque X'X + n*lambda*I
# é sempre invertível para lambda > 0.

# ---- 7. PADRONIZAR NÃO É OPCIONAL
# A penalidade soma os QUADRADOS dos coeficientes, de modo que depende da
# escala. Mudar a unidade de UMA variável muda o ajuste de TODAS.
solo2 <- solo; solo2$pres <- solo2$pres/1000     # mg/dm3 -> g/dm3
b1 <- coef(glmnet(as.matrix(solo[, cand]),  y, alpha = 0, lambda = 0.05,
                  standardize = FALSE))
b2 <- coef(glmnet(as.matrix(solo2[, cand]), y, alpha = 0, lambda = 0.05,
                  standardize = FALSE))
cbind(mg_dm3 = round(as.vector(b1), 6), g_dm3 = round(as.vector(b2), 6))
# Com standardize = TRUE (o padrão) o problema desaparece.
