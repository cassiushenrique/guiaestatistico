# Métricas de classificação: exemplo completo ----
# Pacotes: pROC (ROC e AUC), PRROC (precisão-revocação), caret (matriz)
# Dados fictícios: 400 cafeeiros avaliados para doença foliar.
# O arquivo plantas.csv acompanha o site, na pasta scripts/.
plantas <- read.csv("plantas.csv")
str(plantas)        # doente (0/1), lesoes, umidade, clorofila
table(plantas$doente)
mean(plantas$doente)                        # prevalência = 0,125

# O modelo: regressão logística com as três medidas
mod <- glm(doente ~ lesoes + umidade + clorofila,
           data = plantas, family = binomial)
summary(mod)
p <- fitted(mod)                            # escore de risco de cada planta

# ---- 1. A matriz de confusão no corte 0,5
pred <- as.integer(p >= 0.5)
tab  <- table(Real = plantas$doente, Predito = pred)
tab

library(caret)
confusionMatrix(factor(pred, levels = c(1, 0)),
                factor(plantas$doente, levels = c(1, 0)),
                positive = "1")
# A saída traz acurácia, sensibilidade, especificidade, VPP, VPN,
# acurácia balanceada e o teste de McNemar de uma só vez.

# ---- 2. As métricas, uma a uma, pela definição
vp <- sum(pred == 1 & plantas$doente == 1)
fp <- sum(pred == 1 & plantas$doente == 0)
fn <- sum(pred == 0 & plantas$doente == 1)
vn <- sum(pred == 0 & plantas$doente == 0)
c(VP = vp, FP = fp, FN = fn, VN = vn)

(vp + vn)/length(p)                      # acurácia
vp/(vp + fn)                             # sensibilidade (revocação)
vn/(vn + fp)                             # especificidade
vp/(vp + fp)                             # valor preditivo positivo (precisão)
vn/(vn + fn)                             # valor preditivo negativo
2*vp/(2*vp + fp + fn)                    # F1
(vp/(vp+fn) + vn/(vn+fp))/2              # acurácia balanceada
(vp*vn - fp*fn)/sqrt((vp+fp)*(vp+fn)*(vn+fp)*(vn+fn))  # Matthews

# ---- 3. O PARADOXO DA ACURÁCIA
mean(plantas$doente == 0)                # acurácia do classificador trivial
# O trivial declara TODAS sadias: acurácia 0,875, sensibilidade 0.

# ---- 4. O corte não tem nada de especial: é decisão de custo
for (k in c(0.05, 0.10, 0.125, 0.20, 0.30, 0.40, 0.50, 0.70)) {
  pr <- as.integer(p >= k)
  v <- sum(pr == 1 & plantas$doente == 1); f <- sum(pr == 1 & plantas$doente == 0)
  m <- sum(pr == 0 & plantas$doente == 1); n <- sum(pr == 0 & plantas$doente == 0)
  cat(sprintf("corte %.3f: VP %3d FP %3d FN %3d sens %.4f VPP %.4f\n",
              k, v, f, m, v/(v + m), v/(v + f)))
}

# Se um falso negativo custa C vezes um falso positivo, o corte que
# minimiza o custo ESPERADO é 1/(1 + C)
for (C in c(1, 2, 5, 10, 20, 50)) {
  k <- 1/(1 + C); pr <- as.integer(p >= k)
  custo <- C*sum(pr == 0 & plantas$doente == 1) +
               sum(pr == 1 & plantas$doente == 0)
  cat(sprintf("C = %2d: corte %.4f, custo %.1f\n", C, k, custo))
}

# ---- 5. Curva ROC e AUC
library(pROC)
r <- roc(plantas$doente, p)
r                                         # AUC
auc(r); ci.auc(r)                         # IC por DeLong
plot(r, legacy.axes = TRUE, print.auc = TRUE,
     xlab = "1 - especificidade", ylab = "Sensibilidade")

# A AUC é a probabilidade de uma doente receber escore maior que uma sadia
mean(outer(p[plantas$doente == 1], p[plantas$doente == 0], ">") +
     0.5*outer(p[plantas$doente == 1], p[plantas$doente == 0], "=="))

# AUC de cada variável isolada, para comparar
for (v in c("lesoes", "umidade", "clorofila"))
  cat(v, ":", auc(roc(plantas$doente, plantas[[v]], direction = "auto")), "\n")

# ---- 6. Curva de precisão-revocação: mais informativa com classe rara
library(PRROC)
pr <- pr.curve(scores.class0 = p[plantas$doente == 1],
               scores.class1 = p[plantas$doente == 0], curve = TRUE)
pr$auc.davis.goadrich                     # precisão média
plot(pr)
# A linha de base da PR é a PREVALÊNCIA (0,125), não 0,5.

# ---- 7. CALIBRAÇÃO e escore de Brier
mean((p - plantas$doente)^2)              # Brier do modelo
mean((mean(plantas$doente) - plantas$doente)^2)   # Brier do trivial
1 - mean((p - plantas$doente)^2)/mean((mean(plantas$doente) - plantas$doente)^2)

fx <- cut(p, c(0, .05, .10, .20, .40, .70, 1.01), right = FALSE)
aggregate(cbind(escore = p, real = plantas$doente) ~ fx, FUN = mean)
# As duas colunas deveriam coincidir: é isso que calibração significa.

# ---- 8. Erro-padrão da AUC por bootstrap (escore FIXO)
library(boot)
set.seed(1)
b <- boot(data.frame(y = plantas$doente, s = p), R = 2000,
          statistic = function(d, i) as.numeric(auc(roc(d$y[i], d$s[i],
                                                        quiet = TRUE))))
sd(b$t); boot.ci(b, type = "perc")
