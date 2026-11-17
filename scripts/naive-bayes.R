# Naive Bayes: exemplo completo ----
# Pacotes: e1071 (naiveBayes) e naivebayes (opcional, com núcleo)
# Dados fictícios: 600 lotes de café (arquivo cafe.csv, pasta scripts/ do site).
# Baixe o arquivo para a pasta de trabalho, ou leia direto do endereço:
url <- "https://cassiushenrique.github.io/ModelosEstatisticos/scripts/cafe.csv"
cafe <- if (file.exists("cafe.csv")) read.csv("cafe.csv") else read.csv(url)
str(cafe)        # nota, fino (0/1), alt, temp, chuva, somb, nfol, ph
cafe$fino_f <- factor(cafe$fino, levels = c(0, 1), labels = c("nao", "fino"))
table(cafe$fino_f)

library(e1071)
f <- fino_f ~ alt + temp + chuva + somb + nfol + ph

# ---- 1. O ajuste: uma média e um desvio-padrão por variável e por classe
nb <- naiveBayes(f, data = cafe)
nb                       # prioris e as tabelas de média e desvio por classe

# ---- 2. O cálculo para um lote novo
novo <- data.frame(alt = 1080, temp = 20.0, chuva = 1500, somb = 30, nfol = 2.9, ph = 5.4)
predict(nb, novo, type = "raw")
# À mão: produto das densidades normais vezes a priori
dens <- function(cl) prod(mapply(function(v, m, s) dnorm(novo[[v]], m, s),
  names(novo), sapply(names(novo), function(v) nb$tables[[v]][cl, 1]),
  sapply(names(novo), function(v) nb$tables[[v]][cl, 2]))) * nb$apriori[cl] / sum(nb$apriori)
dens("fino") / (dens("fino") + dens("nao"))
# (o e1071 usa o desvio-padrão com n-1; o scikit-learn, com n: diferença pequena)

# ---- 3. A hipótese "ingênua": correlação DENTRO de cada classe
by(cafe[, c("alt", "temp")], cafe$fino_f, cor)

# ---- 4. Calibração por validação cruzada
set.seed(1)
d <- sample(rep(1:10, length.out = nrow(cafe)))
p <- numeric(nrow(cafe))
for (j in 1:10) p[d == j] <- predict(naiveBayes(f, data = cafe[d != j, ]),
                                     cafe[d == j, ], type = "raw")[, "fino"]
faixa <- cut(p, c(0, .05, .2, .4, .6, .8, .95, 1), include.lowest = TRUE)
cbind(previsto = tapply(p, faixa, mean), observado = tapply(cafe$fino, faixa, mean),
      n = table(faixa))

# ---- 5. Sem a temperatura (redundante com a altitude)
nb2 <- naiveBayes(fino_f ~ alt + chuva + somb + nfol + ph, data = cafe)

# ---- 6. Densidades por núcleo em vez de normais
# install.packages("naivebayes")
library(naivebayes)
nbk <- naive_bayes(f, data = cafe, usekernel = TRUE)
plot(nbk, which = "alt")
