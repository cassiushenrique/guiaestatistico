# Bootstrap: exemplo completo ----
# Pacotes: boot (Canty & Ripley)
# Dados fictícios: 120 talhões de milho. Produtividade (t/ha) em função da
# dose de nitrogênio, densidade de plantio, matéria orgânica e precipitação.
# O arquivo scripts/milho.csv acompanha o repositório.
milho <- read.csv("milho.csv")
str(milho); summary(milho$y)
colMeans(milho)

# O modelo usado em todas as fichas deste grupo
f_completo <- y ~ N + I(N^2) + densidade + I(densidade^2) + mo + chuva

library(boot)
set.seed(1)

# ---- 1. O bootstrap "à mão", para entender o mecanismo
n <- nrow(milho); B <- 10000
med_obs <- median(milho$y); med_obs
reamostras <- replicate(B, median(sample(milho$y, n, replace = TRUE)))
mean(reamostras)                      # média das reamostras
mean(reamostras) - med_obs            # viés estimado
sd(reamostras)                        # ERRO-PADRÃO estimado
quantile(reamostras, c(0.025, 0.975)) # IC 95% percentil

# Para comparar: o erro-padrão da MÉDIA tem fórmula fechada
sd(milho$y)/sqrt(n)

# ---- 2. O mesmo com o pacote boot
f_med <- function(dados, i) median(dados[i])
b <- boot(milho$y, f_med, R = 10000)
b                                     # estimativa, viés e erro-padrão
boot.ci(b, type = c("perc", "basic", "norm", "bca"))

# ---- 3. Estatísticas sem fórmula de erro-padrão
f_cv  <- function(d, i) sd(d[i])/mean(d[i])
f_cor <- function(d, i) cor(d[i, "y"], d[i, "mo"])
f_raz <- function(d, i) {
  dd <- d[i, ]
  mean(dd$y[dd$N >= 160])/mean(dd$y[dd$N <= 120])
}
boot.ci(boot(milho$y, f_cv,  R = 10000), type = "bca")
boot.ci(boot(milho,   f_cor, R = 10000), type = "bca")
boot.ci(boot(milho,   f_raz, R = 10000), type = "bca")

# ---- 4. Quantas reamostras? Duas fontes de erro
for (B in c(50, 200, 1000, 5000, 20000)) {
  eps <- replicate(30, sd(replicate(B, median(sample(milho$y, n, TRUE)))))
  cat(sprintf("B = %6d: EP %.4f, variação entre execuções %.4f\n",
              B, mean(eps), sd(eps)))
}

# ---- 5. Bootstrap para o ERRO DE PREDIÇÃO
eqm <- function(obs, prev) mean((obs - prev)^2)
m   <- lm(f_completo, data = milho)
resub <- eqm(milho$y, fitted(m)); resub

ingenuo <- oob <- numeric(0); frac <- numeric(0)
for (b in 1:200) {
  i  <- sample(n, n, replace = TRUE)
  mm <- lm(f_completo, data = milho[i, ])
  ingenuo[b] <- eqm(milho$y, predict(mm, newdata = milho))
  fora <- setdiff(1:n, unique(i)); frac[b] <- length(fora)/n
  oob[b] <- eqm(milho$y[fora], predict(mm, newdata = milho[fora, ]))
}
mean(ingenuo)                       # OTIMISTA
mean(oob)                           # PESSIMISTA
0.368*resub + 0.632*mean(oob)       # estimador 0,632 de Efron
mean(frac)                          # ~ 1/e = 0,3679

# ---- 6. Bootstrap de BLOCOS, para dados dependentes
set.seed(7)
serie <- as.numeric(arima.sim(list(ar = 0.8), 200))
# Reamostrar observações destrói a autocorrelação:
sd(replicate(2000, mean(sample(serie, 200, TRUE))))
# Reamostrar BLOCOS de 20 a preserva:
bl <- function(x, L = 20) {
  k <- ceiling(length(x)/L)
  ini <- sample(1:(length(x) - L + 1), k, replace = TRUE)
  unlist(lapply(ini, function(s) x[s:(s + L - 1)]))[1:length(x)]
}
sd(replicate(2000, mean(bl(serie))))
# O pacote boot faz isso com sim = "fixed":
boot(serie, function(d, i) mean(d[i]), R = 2000,
     sim = "fixed", l = 20)

# ---- 7. Bootstrap de RESÍDUOS em regressão (alternativa)
r <- residuals(m); fit <- fitted(m)
coefs <- replicate(2000, {
  yb <- fit + sample(r, n, replace = TRUE)
  coef(lm(yb ~ N + I(N^2) + densidade + I(densidade^2) + mo + chuva,
          data = milho)) })
apply(coefs, 1, sd)                 # EP bootstrap dos coeficientes
summary(m)$coef[, 2]                # EP clássicos, para comparar
