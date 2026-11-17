# BART (Bayesian additive regression trees): exemplo completo ----
# Pacotes: BART (Sparapani, Spanbauer e McCulloch) ou dbarts
# Dados fictícios: 600 lotes de café (arquivo cafe.csv, pasta scripts/ do site).
# Baixe o arquivo para a pasta de trabalho, ou leia direto do endereço:
url <- "https://cassiushenrique.github.io/ModelosEstatisticos/scripts/cafe.csv"
cafe <- if (file.exists("cafe.csv")) read.csv("cafe.csv") else read.csv(url)
str(cafe)        # nota, fino (0/1), alt, temp, chuva, somb, nfol, ph
cafe$fino_f <- factor(cafe$fino, levels = c(0, 1), labels = c("nao", "fino"))
table(cafe$fino_f)

# install.packages("BART")
library(BART)
X <- as.matrix(cafe[, c("alt","temp","chuva","somb","nfol","ph")])
set.seed(5)
i_te <- sample(nrow(cafe), 200)
Xtr <- X[-i_te, ]; ytr <- cafe$nota[-i_te]; Xte <- X[i_te, ]; yte <- cafe$nota[i_te]

# ---- 1. O ajuste: 50 árvores (o padrão do pacote é 200), 1.000 aquecimento, 1.000 sorteios
set.seed(1)
fit <- wbart(x.train = Xtr, y.train = ytr, x.test = Xte,
             ntree = 50, nskip = 1000, ndpost = 1000,
             base = 0.95, power = 2, k = 2,            # priors de Chipman et al. (2010)
             sigdf = 3, sigquant = 0.90)

# ---- 2. Convergência: a cadeia de sigma
plot(fit$sigma, type = "l", xlab = "iteração", ylab = "sigma")
abline(v = 1000, lty = 2)

# ---- 3. Predição e incerteza no teste
pred <- fit$yhat.test.mean
mean((yte - pred)^2)                                    # EQM no teste
li <- apply(fit$yhat.test, 2, quantile, 0.05)            # intervalo de 90% para f(x)
ls <- apply(fit$yhat.test, 2, quantile, 0.95)
mean(ls - li)                                           # largura média
summary(fit$sigma[-(1:1000)])                           # sigma a posteriori

# Intervalo PREDITIVO para uma nova nota: soma o erro sigma a cada sorteio
s <- fit$sigma[-(1:1000)]
ynovo <- fit$yhat.test + matrix(rnorm(length(fit$yhat.test)), nrow(fit$yhat.test)) * s
pli <- apply(ynovo, 2, quantile, 0.05); pls <- apply(ynovo, 2, quantile, 0.95)
mean(yte >= pli & yte <= pls)                           # cobertura no teste

# ---- 4. Uso das variáveis (mais informativo com poucas árvores, ex.: ntree = 10)
colMeans(fit$varprob)

# ---- 5. Classificação: probit BART
set.seed(1)
fitc <- pbart(x.train = Xtr, y.train = cafe$fino[-i_te], x.test = Xte, ntree = 50)
p <- fitc$prob.test.mean
mean((p >= 0.5) != cafe$fino[i_te])
