# Regressão por componentes principais (PCR): exemplo completo ----
# Pacotes: pls (função pcr)
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

# ---- 1. Os componentes principais
Xp <- scale(as.matrix(solo[, cand]))
sv <- svd(Xp)
pv <- sv$d^2/sum(sv$d^2)                   # proporção da variância de X
Tm <- sv$u %*% diag(sv$d)                  # os escores
data.frame(CP = 1:12, d = round(sv$d, 3),
           var_pct = round(100*pv, 2), acum = round(100*cumsum(pv), 2),
           r_com_y = round(abs(cor(Tm, solo$y)), 4))
# Os componentes são ORTOGONAIS: todo VIF é exatamente 1.
round(cor(Tm[, 1:4]), 10)

# ---- 2. A PCR com o pacote pls
library(pls)
mp <- pcr(y ~ ., data = solo, scale = TRUE, validation = "CV", segments = 10)
summary(mp)
validationplot(mp, val.type = "MSEP")
RMSEP(mp)
# O m escolhido:
which.min(RMSEP(mp)$val[1, 1, -1])
coef(mp, ncomp = 7)                        # na escala das variáveis originais

# ---- 3. A REGRA DOS "80% DA VARIÂNCIA" É UM MAU CRITÉRIO
for (alvo in c(0.70, 0.80, 0.90, 0.95)) {
  m <- which(cumsum(pv) >= alvo)[1]
  cat(sprintf("%2.0f%% da variância -> m = %d\n", 100*alvo, m))
}
# O critério só olha X e ignora y por completo. Use validação cruzada.

# ---- 4. PCR com m = p é EXATAMENTE o MQO
max(abs(coef(pcr(y ~ ., data = solo, scale = TRUE), ncomp = 12) -
        coef(lm(y ~ ., data = solo))[-1]*apply(solo[, cand], 2, sd)))

# ---- 5. O caso em que a ordem da variância ENGANA: p > n
# Dados fictícios: espectro NIR, 60 amostras, 200 comprimentos de onda.
# O arquivo espectro.csv acompanha o site, na pasta scripts/.
esp <- read.csv("espectro.csv")
dim(esp)                      # 60 x 201: proteina + nm1100 ... nm2300
Xe <- as.matrix(esp[, -1]); ye <- esp$proteina
qr(scale(Xe))$rank            # 59: o posto é menor que o número de colunas

Ep <- scale(Xe); sve <- svd(Ep)
Te <- sve$u %*% diag(sve$d)
pve <- sve$d^2/sum(sve$d^2)
data.frame(CP = 1:10, var_pct = round(100*pve[1:10], 2),
           r_com_y = round(abs(cor(Te[, 1:10], ye)), 4))
# O CP1 explica 83,5% da variância do espectro e tem r de só 0,151 com a
# proteína. O CP3 explica 3,5% e tem r de 0,599.

me <- pcr(ye ~ Ep, validation = "CV", segments = 10)
summary(me)
validationplot(me, val.type = "R2")
# Com UM componente o R² validado é NEGATIVO: o CP1 prediz pior que a
# média de y, apesar de carregar 83,5% da variância do espectro.

# ---- 6. A PCR É UM RIDGE DISCRETO
l0 <- 0.03162
data.frame(CP = 1:12,
           ridge = round(sv$d^2/(sv$d^2 + l0*nrow(solo)), 4),
           pcr_m7 = c(rep(1, 7), rep(0, 5)))
# O ridge multiplica cada direção por um fator CONTÍNUO; a PCR, por 1 ou 0.

# ---- 7. As cargas: interpretação, não resultado estatístico
round(sve$v[, 1:4], 3)                     # espectro
round(svd(Xp)$v[, 1:4], 3)                 # solo
