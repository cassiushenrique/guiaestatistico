# Máquinas de vetores de suporte lineares: exemplo completo ----
# Pacote: e1071 (interface para a biblioteca LIBSVM)
# Dados fictícios: 600 lotes de café (arquivo cafe.csv, pasta scripts/ do site).
# Baixe o arquivo para a pasta de trabalho, ou leia direto do endereço:
url <- "https://cassiushenrique.github.io/ModelosEstatisticos/scripts/cafe.csv"
cafe <- if (file.exists("cafe.csv")) read.csv("cafe.csv") else read.csv(url)
str(cafe)        # nota, fino (0/1), alt, temp, chuva, somb, nfol, ph
cafe$fino_f <- factor(cafe$fino, levels = c(0, 1), labels = c("nao", "fino"))
table(cafe$fino_f)

library(e1071)
vars <- c("alt", "temp", "chuva", "somb", "nfol", "ph")

# ---- 1. Em duas dimensões, para ver a margem
m2 <- svm(fino_f ~ alt + ph, data = cafe, kernel = "linear", cost = 1, scale = TRUE)
m2$tot.nSV                                  # número de vetores de suporte
plot(m2, data = cafe, formula = ph ~ alt)   # região de decisão e vetores de suporte
# coeficientes na escala padronizada: w = soma(alfa_i y_i x_i)
w <- t(m2$coefs) %*% m2$SV; b <- -m2$rho
w; b; 2 / sqrt(sum(w^2))                     # margem

# ---- 2. Seis variáveis: escolher C (cost) por validação cruzada
set.seed(1)
aj <- tune(svm, fino_f ~ alt + temp + chuva + somb + nfol + ph, data = cafe,
           kernel = "linear", ranges = list(cost = c(0.001, 0.01, 0.1, 1, 10, 100)),
           tunecontrol = tune.control(cross = 10))
summary(aj)                                 # erro de cada C
plot(aj)
melhor <- aj$best.model

# ---- 3. Comparação com a regressão logística
g <- glm(fino ~ alt + temp + chuva + somb + nfol + ph,
         data = transform(cafe, alt = scale(alt), temp = scale(temp), chuva = scale(chuva),
                          somb = scale(somb), nfol = scale(nfol), ph = scale(ph)),
         family = binomial)
cbind(svm = as.vector(t(melhor$coefs) %*% melhor$SV), logistica = coef(g)[-1])

# ---- 4. Probabilidades pelo método de Platt
mp <- svm(fino_f ~ alt + temp + chuva + somb + nfol + ph, data = cafe,
          kernel = "linear", cost = aj$best.parameters$cost, probability = TRUE)
p <- attr(predict(mp, cafe, probability = TRUE), "probabilities")[, "fino"]

# ---- 5. Uma variável nova faz o que a fronteira linear não faz: o degrau
cafe$acima <- as.numeric(cafe$alt >= 1050)
set.seed(1)
summary(tune(svm, fino_f ~ alt + temp + chuva + somb + nfol + ph + acima, data = cafe,
             kernel = "linear", ranges = list(cost = aj$best.parameters$cost),
             tunecontrol = tune.control(cross = 10)))$best.performance
