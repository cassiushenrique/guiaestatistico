# Regressão por mínimos quadrados parciais (PLS): exemplo completo ----
# Pacotes: pls (função plsr)
# Dados fictícios: espectro NIR, 60 amostras, 200 comprimentos de onda.
# O arquivo espectro.csv acompanha o site, na pasta scripts/.
esp <- read.csv("espectro.csv")
dim(esp)                      # 60 x 201: proteina + nm1100 ... nm2300
Xe <- as.matrix(esp[, -1]); ye <- esp$proteina
qr(scale(Xe))$rank            # 59: o posto é menor que o número de colunas

# ---- 1. A PLS no espectro, com validação cruzada
library(pls)
dados <- data.frame(proteina = ye, Xe = I(Xe))
ml <- plsr(proteina ~ Xe, data = dados, validation = "CV", segments = 10)
summary(ml)        # DUAS linhas de variância explicada: X e proteina
validationplot(ml, val.type = "R2")
RMSEP(ml)
which.min(RMSEP(ml)$val[1, 1, -1])      # o m escolhido

# A comparação que interessa: PLS contra PCR, no mesmo conjunto
mp <- pcr(proteina ~ Xe, data = dados, validation = "CV", segments = 10)
cbind(PLS = drop(RMSEP(ml)$val)[1:13], PCR = drop(RMSEP(mp)$val)[1:13])
# A PLS chega a um erro útil com cerca de METADE dos componentes.

# ---- 2. O algoritmo NIPALS, passo a passo (um componente)
Ep <- scale(Xe); yc <- ye - mean(ye)
w1 <- t(Ep) %*% yc; w1 <- w1/sqrt(sum(w1^2))    # direção de maior COVARIÂNCIA
t1 <- Ep %*% w1
p1 <- t(Ep) %*% t1/sum(t1^2)                    # a carga
q1 <- sum(yc*t1)/sum(t1^2)
cor(t1, ye)                                      # |r| do escore com y
sum(t1^2*sum(p1^2))/sum(Ep^2)                    # fração da variância de X
# Deflação, e repete-se nos resíduos:
Ep2 <- Ep - t1 %*% t(p1); yc2 <- yc - q1*t1

# Para comparar, o primeiro componente PRINCIPAL do mesmo espectro:
cp1 <- svd(Ep)$u[, 1]
cor(cp1, ye)                                     # muito menor
# A PLS abre mão de variância em X para ganhar correlação com y.

# ---- 3. PLS com m = posto(X) é o MQO
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

Xs <- scale(as.matrix(solo[, cand]))
ms <- plsr(solo$y ~ Xs, ncomp = 12)
max(abs(drop(coef(ms, ncomp = 12)) - coef(lm(y ~ ., data = solo))[-1]*apply(solo[, cand], 2, sd)))

# ---- 4. E quando a colinearidade é apenas moderada? (solo)
m1 <- plsr(y ~ ., data = solo, scale = TRUE, validation = "CV", segments = 10)
m2 <- pcr(y ~ ., data = solo, scale = TRUE, validation = "CV", segments = 10)
cbind(PLS = drop(RMSEP(m1)$val)^2, PCR = drop(RMSEP(m2)$val)^2)
# Aqui a PLS não tem vantagem: o MQO já funciona.

# ---- 5. OS PESOS: coeficientes contra VIP
b <- drop(coef(ml, ncomp = 5))
W <- ml$loading.weights[, 1:5]; Tt <- ml$scores[, 1:5]
q <- drop(ml$Yloadings)[1:5]
ss <- q^2*colSums(Tt^2)
VIP <- sqrt(ncol(Xe)*rowSums(sweep(W^2, 2, ss, "*"))/sum(ss))
lam <- as.numeric(sub("nm", "", colnames(Xe)))

# As bandas CONSTRUÍDAS: 1.350 e 1.920 determinam a proteína; 1.680 e
# 2.180 não. Qual das duas medidas acerta?
for (r in list(c(1100,1260,"nao"), c(1260,1440,"SIM"), c(1440,1560,"nao"),
               c(1560,1800,"nao"), c(1800,2040,"SIM"), c(2040,2300,"nao"))) {
  m <- lam >= as.numeric(r[1]) & lam < as.numeric(r[2])
  cat(sprintf("%4s-%4s nm: VIP medio %.3f, acima de 1: %2d, soma|coef| %.3f  determina: %s\n",
      r[1], r[2], mean(VIP[m]), sum(VIP[m] > 1), sum(abs(b[m])), r[3]))
}
sum(VIP > 1)      # 85 das 200: a regra "VIP > 1" não discrimina aqui
mean(VIP^2)       # é sempre 1 por construção

# ---- 6. Para classificação e múltiplas respostas
# plsr aceita matriz de respostas (PLS2); para discriminação, ver
# caret::train(method = "pls") ou o pacote mixOmics (PLS-DA).
