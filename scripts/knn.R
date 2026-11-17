# k vizinhos mais próximos (kNN): exemplo completo ----
# Pacotes: class (incluído no R) e FNN (opcional, para regressão)
# Dados fictícios: 600 lotes de café (arquivo cafe.csv, pasta scripts/ do site).
# Baixe o arquivo para a pasta de trabalho, ou leia direto do endereço:
url <- "https://cassiushenrique.github.io/ModelosEstatisticos/scripts/cafe.csv"
cafe <- if (file.exists("cafe.csv")) read.csv("cafe.csv") else read.csv(url)
str(cafe)        # nota, fino (0/1), alt, temp, chuva, somb, nfol, ph
cafe$fino_f <- factor(cafe$fino, levels = c(0, 1), labels = c("nao", "fino"))
table(cafe$fino_f)

library(class)
vars <- c("alt", "temp", "chuva", "somb", "nfol", "ph")
X <- scale(cafe[, vars])            # PADRONIZAR: média 0 e desvio 1 em cada variável
y <- cafe$fino_f

# ---- 1. Um lote novo e seus vizinhos
x0 <- c(alt = 1080, temp = 20.0, chuva = 1500, somb = 30, nfol = 2.9, ph = 5.4)
z0 <- (x0 - attr(X, "scaled:center")) / attr(X, "scaled:scale")
d <- sqrt(colSums((t(X) - z0)^2))
viz <- order(d)[1:7]
cbind(cafe[viz, vars], dist = round(d[viz], 3), fino = cafe$fino[viz])
mean(cafe$fino[viz])                 # P(fino) estimada com k = 7

# ---- 2. Escolher k por validação cruzada
# knn.cv() faz a validação deixando um de fora (LOOCV), de graça
ks <- c(1, 3, 5, 9, 15, 25, 41, 61, 101, 201)
erro <- sapply(ks, function(k) { set.seed(1); mean(knn.cv(X, y, k = k) != y) })
plot(ks, erro, log = "x", type = "b", xlab = "k", ylab = "erro (LOOCV)")
k_best <- ks[which.min(erro)]

# ---- 3. Sem padronizar: a distância é dominada pelas variáveis de maior escala
erro_bruto <- sapply(ks, function(k) { set.seed(1); mean(knn.cv(as.matrix(cafe[, vars]), y, k = k) != y) })
cbind(k = ks, padronizado = erro, bruto = erro_bruto)

# ---- 4. Probabilidades: proporção de votos
set.seed(1)
pr <- knn.cv(X, y, k = k_best, prob = TRUE)
p_venc <- attr(pr, "prob")           # proporção da classe VENCEDORA
p_fino <- ifelse(pr == "fino", p_venc, 1 - p_venc)

# ---- 5. A maldição da dimensionalidade: colunas de ruído
set.seed(4)
for (q in c(0, 6, 24, 48)) {
  Xq <- if (q > 0) cbind(X, matrix(rnorm(nrow(X) * q), ncol = q)) else X
  cat(q, "colunas de ruído: erro", mean(knn.cv(Xq, y, k = k_best) != y), "\n")
}

# ---- 6. Regressão (pacote FNN)
# install.packages("FNN")
library(FNN)
r <- knn.reg(train = X, y = cafe$nota, k = 15)   # sem 'test': LOOCV
mean((cafe$nota - r$pred)^2)
