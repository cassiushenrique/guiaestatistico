# Rede sem camada oculta (um neurônio = classificador logístico): exemplo completo ----
# Pacote: nnet (incluído na instalação do R)
# Dados fictícios: 600 lotes de café (arquivo cafe.csv, pasta scripts/ do site).
# Baixe o arquivo para a pasta de trabalho, ou leia direto do endereço:
url <- "https://cassiushenrique.github.io/ModelosEstatisticos/scripts/cafe.csv"
cafe <- if (file.exists("cafe.csv")) read.csv("cafe.csv") else read.csv(url)
str(cafe)        # nota, fino (0/1), alt, temp, chuva, somb, nfol, ph
cafe$fino_f <- factor(cafe$fino, levels = c(0, 1), labels = c("nao", "fino"))
table(cafe$fino_f)

vars <- c("alt", "temp", "chuva", "somb", "nfol", "ph")
Z <- scale(cafe[, vars]); y <- cafe$fino
sig <- function(a) 1 / (1 + exp(-a))

# ---- 1. Descida do gradiente à mão
w <- rep(0, 6); b <- 0; eta <- 1
for (it in 1:20000) {
  q <- sig(b + Z %*% w)
  w <- w - eta * t(Z) %*% (q - y) / nrow(Z)
  b <- b - eta * mean(q - y)
}
# ---- 2. ... é a regressão logística
g <- glm(y ~ Z, family = binomial)
cbind(gradiente = c(b, w), glm = coef(g))

# ---- 3. A mesma rede no nnet: size = 0 e skip = TRUE = sem camada oculta
library(nnet)
n0 <- nnet(fino_f ~ alt + temp + chuva + somb + nfol + ph,
           data = data.frame(Z, fino_f = cafe$fino_f), size = 0, skip = TRUE,
           decay = 0, maxit = 1000)
coef(n0)

# ---- 4. Decaimento de pesos (weight decay) = penalidade ridge
for (lam in c(0, 0.01, 0.1, 1)) {
  n1 <- nnet(fino_f ~ ., data = data.frame(Z, fino_f = cafe$fino_f), size = 0, skip = TRUE,
             decay = lam * nrow(Z), maxit = 1000, trace = FALSE)
  cat("lambda", lam, ": |w| =", sqrt(sum(coef(n1)[-1]^2)), "\n") }
# (o nnet soma a perda sobre as observações; por isso decay = lambda * n)

# ---- 5. O perceptron de Rosenblatt
ys <- 2 * y - 1; w <- rep(0, 6); b <- 0; set.seed(0)
for (ep in 1:50) {
  for (i in sample(nrow(Z))) if (ys[i] * (sum(Z[i, ] * w) + b) <= 0) {
    w <- w + ys[i] * Z[i, ]; b <- b + ys[i] }
  if (ep %% 10 == 0) cat("época", ep, ": erros", sum(ys * (Z %*% w + b) <= 0), "\n") }

# ---- 6. Três classes: camada de saída softmax = regressão multinomial
cafe$classe <- cut(cafe$nota, c(-Inf, 76, 80, Inf), right = FALSE,
                   labels = c("comercial", "bom", "fino"))
mm <- multinom(classe ~ ., data = data.frame(Z, classe = cafe$classe))
table(real = cafe$classe, previsto = predict(mm))
