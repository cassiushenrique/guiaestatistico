# Árvore de classificação (CART): exemplo completo ----
# Pacotes: rpart (incluído no R) e rpart.plot (opcional)
# Dados fictícios: 600 lotes de café (arquivo cafe.csv, pasta scripts/ do site).
# Baixe o arquivo para a pasta de trabalho, ou leia direto do endereço:
url <- "https://cassiushenrique.github.io/ModelosEstatisticos/scripts/cafe.csv"
cafe <- if (file.exists("cafe.csv")) read.csv("cafe.csv") else read.csv(url)
str(cafe)        # nota, fino (0/1), alt, temp, chuva, somb, nfol, ph
cafe$fino_f <- factor(cafe$fino, levels = c(0, 1), labels = c("nao", "fino"))
table(cafe$fino_f)

library(rpart)
f <- fino_f ~ alt + temp + chuva + somb + nfol + ph

# ---- 1. Árvore máxima, critério de Gini (padrão), validação cruzada interna
set.seed(1)
arv_max <- rpart(f, data = cafe, method = "class",
                 parms = list(split = "gini"),
                 control = rpart.control(cp = 0, minsplit = 2, xval = 10))
printcp(arv_max); plotcp(arv_max)
tab <- arv_max$cptable
arv <- prune(arv_max, cp = tab[which.min(tab[, "xerror"]), "CP"])
print(arv)
library(rpart.plot); rpart.plot(arv, extra = 104)   # proporção de cada classe na folha

# ---- 2. Entropia em vez de Gini
arv_ent <- rpart(f, data = cafe, method = "class", parms = list(split = "information"))

# ---- 3. Probabilidades: a proporção de finos em cada folha
p <- predict(arv, type = "prob")[, "fino"]
table(round(p, 3))                 # poucas probabilidades distintas

# ---- 4. Custos desiguais: perder um lote fino custa 3 vezes mais
# Matriz de perda: linhas = classe real, colunas = classe prevista
perda <- matrix(c(0, 1,
                  3, 0), nrow = 2, byrow = TRUE)
arv_c <- rpart(f, data = cafe, method = "class", parms = list(loss = perda))
# Alternativa equivalente na decisão: limiar 1/(1 + 3) = 0,25 na probabilidade
prev <- ifelse(predict(arv, type = "prob")[, "fino"] >= 0.25, "fino", "nao")
table(real = cafe$fino_f, previsto = prev)

# ---- 5. Comparação com a regressão logística (validação cruzada)
set.seed(1)
d <- sample(rep(1:10, length.out = nrow(cafe)))
err <- matrix(NA, 10, 2, dimnames = list(NULL, c("arvore", "logistica")))
for (j in 1:10) {
  tr <- cafe[d != j, ]; te <- cafe[d == j, ]
  a <- rpart(f, data = tr, method = "class", control = rpart.control(cp = 0, xval = 10))
  a <- prune(a, cp = a$cptable[which.min(a$cptable[, "xerror"]), "CP"])
  g <- glm(fino ~ alt + temp + chuva + somb + nfol + ph, data = tr, family = binomial)
  err[j, ] <- c(mean(predict(a, te, type = "class") != te$fino_f),
                mean((predict(g, te, type = "response") >= 0.5) != te$fino))
}
colMeans(err)
