# Árvore de regressão (CART): exemplo completo ----
# Pacotes: rpart (incluído na instalação do R) e rpart.plot (opcional)
# Dados fictícios: 600 lotes de café (arquivo cafe.csv, pasta scripts/ do site).
# Baixe o arquivo para a pasta de trabalho, ou leia direto do endereço:
url <- "https://cassiushenrique.github.io/ModelosEstatisticos/scripts/cafe.csv"
cafe <- if (file.exists("cafe.csv")) read.csv("cafe.csv") else read.csv(url)
str(cafe)        # nota, fino (0/1), alt, temp, chuva, somb, nfol, ph
cafe$fino_f <- factor(cafe$fino, levels = c(0, 1), labels = c("nao", "fino"))
table(cafe$fino_f)

library(rpart)

# ---- 1. A referência: modelos suaves
f <- nota ~ alt + temp + chuva + somb + nfol + ph
summary(lm(f, data = cafe))$r.squared

# ---- 2. Crescer a árvore máxima (cp = 0) com validação cruzada interna
set.seed(1)
arv_max <- rpart(f, data = cafe, method = "anova",
                 control = rpart.control(cp = 0, minsplit = 2, minbucket = 1, xval = 10))
printcp(arv_max)     # tabela: cp, número de cortes, erro relativo, xerror, xstd
plotcp(arv_max)      # erro validado em função de cp (com a linha do EP único)

# ---- 3. Podar: o menor xerror, ou a regra do erro-padrão único
tab <- arv_max$cptable
i_min <- which.min(tab[, "xerror"])
cp_min <- tab[i_min, "CP"]
lim <- tab[i_min, "xerror"] + tab[i_min, "xstd"]
cp_1ep <- tab[min(which(tab[, "xerror"] <= lim)), "CP"]
arv <- prune(arv_max, cp = cp_1ep)
print(arv)
# install.packages("rpart.plot")
library(rpart.plot)
rpart.plot(arv, type = 2, extra = 101, digits = 3)

# ---- 4. A predição é uma escada: variar só a altitude
grade <- data.frame(alt = seq(600, 1500, by = 10),
                    temp = median(cafe$temp), chuva = median(cafe$chuva),
                    somb = median(cafe$somb), nfol = median(cafe$nfol), ph = median(cafe$ph))
plot(grade$alt, predict(arv, grade), type = "s", lwd = 2,
     xlab = "Altitude (m)", ylab = "Nota prevista")
abline(v = 1050, lty = 3)

# ---- 5. Instabilidade: a mesma análise em reamostras bootstrap
set.seed(3)
info <- t(replicate(200, {
  i <- sample(nrow(cafe), replace = TRUE)
  a <- prune(rpart(f, data = cafe[i, ], control = rpart.control(cp = 0, xval = 0)), cp = cp_1ep)
  c(var1 = as.character(a$frame$var[1]), folhas = sum(a$frame$var == "<leaf>"))
}))
table(info[, "var1"])                    # variável do primeiro corte
summary(as.numeric(info[, "folhas"]))    # número de folhas

# ---- 6. Comparação por validação cruzada de 10 dobras
set.seed(1)
d <- sample(rep(1:10, length.out = nrow(cafe)))
e_lin <- e_arv <- numeric(nrow(cafe))
for (j in 1:10) {
  tr <- cafe[d != j, ]; te <- cafe[d == j, ]
  e_lin[d == j] <- te$nota - predict(lm(f, data = tr), te)
  a <- rpart(f, data = tr, control = rpart.control(cp = 0, xval = 10))
  t2 <- a$cptable; k <- which.min(t2[, "xerror"])
  e_arv[d == j] <- te$nota - predict(prune(a, cp = t2[k, "CP"]), te)
}
c(linear = mean(e_lin^2), arvore = mean(e_arv^2))
# A árvore de regressão também serve para resposta de contagem ou de
# sobrevivência: method = "poisson" e method = "exp" no rpart.
