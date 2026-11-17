# SVM com núcleo radial (RBF): exemplo completo ----
# Pacote: e1071
# Dados fictícios: 600 lotes de café (arquivo cafe.csv, pasta scripts/ do site).
# Baixe o arquivo para a pasta de trabalho, ou leia direto do endereço:
url <- "https://cassiushenrique.github.io/ModelosEstatisticos/scripts/cafe.csv"
cafe <- if (file.exists("cafe.csv")) read.csv("cafe.csv") else read.csv(url)
str(cafe)        # nota, fino (0/1), alt, temp, chuva, somb, nfol, ph
cafe$fino_f <- factor(cafe$fino, levels = c(0, 1), labels = c("nao", "fino"))
table(cafe$fino_f)

library(e1071)
f <- fino_f ~ alt + temp + chuva + somb + nfol + ph

# ---- 1. O núcleo: semelhança entre dois lotes padronizados
Z <- scale(cafe[, c("alt","temp","chuva","somb","nfol","ph")])
d2 <- sum((Z[1, ] - Z[2, ])^2)
sapply(c(0.05, 0.5, 5), function(g) exp(-g * d2))

# ---- 2. A grade de C (cost) e gamma, por validação cruzada
set.seed(1)
aj <- tune(svm, f, data = cafe, kernel = "radial",
           ranges = list(cost = c(0.1, 1, 10, 100), gamma = c(0.01, 0.05, 0.2, 1, 5)),
           tunecontrol = tune.control(cross = 10))
summary(aj)
plot(aj, transform.x = log10, transform.y = log10)   # mapa do erro na grade
aj$best.parameters

# ---- 3. Sobreajuste: gamma grande decora o treino
for (g in c(0.05, 1, 5)) {
  m <- svm(f, data = cafe, kernel = "radial", cost = 10, gamma = g)
  cat("gamma", g, ": erro de treino", mean(predict(m) != cafe$fino_f),
      "| vetores de suporte", m$tot.nSV, "\n") }

# ---- 4. A interação, com probabilidades de Platt
m <- svm(f, data = cafe, kernel = "radial", cost = aj$best.parameters$cost,
         gamma = aj$best.parameters$gamma, probability = TRUE)
grade <- expand.grid(somb = seq(0, 80, 20), temp = c(18.5, 22.5))
grade <- transform(grade, alt = 1150, chuva = median(cafe$chuva),
                   nfol = median(cafe$nfol), ph = median(cafe$ph))
grade$p <- attr(predict(m, grade, probability = TRUE), "probabilities")[, "fino"]
grade

# ---- 5. Regressão por vetores de suporte (SVR) na nota
set.seed(1)
ajr <- tune(svm, nota ~ alt + temp + chuva + somb + nfol + ph, data = cafe,
            ranges = list(cost = c(10, 100), gamma = c(0.05, 0.2), epsilon = c(0.1, 1, 2)))
summary(ajr)
