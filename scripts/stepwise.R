# Seleção stepwise por AIC: exemplo completo ----
# Pacotes: nenhum (step e add1/drop1 são do R base); MASS::stepAIC é equivalente
# Dados fictícios: 90 amostras de solo, 12 candidatas.
# O arquivo solo.csv acompanha o site, na pasta scripts/.
solo <- read.csv("solo.csv")
str(solo)          # y (t/ha) e as 12 candidatas
dim(solo)
# O modelo VERDADEIRO (conhecido por construção) usa só quatro:
#   y = 4,00 + 0,80*mo + 0,0250*pres + 0,0300*v + 0,0200*argila + erro
cand <- c("ph","mo","pres","k","ca","mg","al","ctc","v","argila","silte","areia")
verdadeiras <- c("mo","pres","v","argila")

# A colinearidade, que é o problema central do grupo
library(car)
vif(lm(y ~ ., data = solo))
round(cor(solo[, cand]), 2)

# ---- 1. As três direções
completo <- lm(y ~ ., data = solo)
nulo     <- lm(y ~ 1, data = solo)

frente <- step(nulo, scope = list(lower = nulo, upper = completo),
               direction = "forward", trace = 1)
tras   <- step(completo, direction = "backward", trace = 1)
ambos  <- step(completo, scope = list(lower = nulo, upper = completo),
               direction = "both", trace = 1)

formula(frente); formula(tras); formula(ambos)
# As três NÃO coincidem: a para frente pega a AREIA, a para trás pega
# ARGILA e SILTE. São representantes opostos do mesmo grupo colinear.

# Com o BIC, que penaliza mais (k = log(n) em vez de 2):
step(nulo, scope = list(lower = nulo, upper = completo),
     direction = "forward", k = log(nrow(solo)))

# ---- 2. O que o step examina em cada passo
add1(nulo, scope = completo)        # o ganho de AIC de cada candidata
drop1(completo)                     # a perda de AIC ao remover cada uma

# ---- 3. OS VALORES-p DO MODELO SELECIONADO NÃO SÃO VALORES-p
summary(frente)
# Todos "significativos". Mas esses p supõem que o modelo foi fixado ANTES
# de ver os dados, e ele foi escolhido OLHANDO os dados.

# A demonstração: y PERMUTADO, sem nenhuma relação com X
set.seed(13)
sim <- t(replicate(2000, {
  solo2 <- solo; solo2$y <- sample(solo$y)
  n0 <- lm(y ~ 1, data = solo2); nc <- lm(y ~ ., data = solo2)
  m  <- step(n0, scope = list(lower = n0, upper = nc),
             direction = "forward", trace = 0)
  k  <- length(coef(m)) - 1
  pm <- if (k == 0) 1 else min(summary(m)$coef[-1, 4])
  c(k = k, p = pm, r2 = summary(m)$r.squared) }))
mean(sim[, "k"])                      # variáveis selecionadas, em média
mean(sim[, "p"] < 0.05)               # deveria ser ~0,05
mean(sim[, "p"] < 0.01)               # deveria ser ~0,01
mean(sim[, "r2"])                     # R² médio em dados SEM SINAL

# ---- 4. INSTABILIDADE
set.seed(17)
cont <- numeric(12); names(cont) <- cand; tam <- numeric(1000)
for (b in 1:1000) {
  i <- sample(nrow(solo), replace = TRUE); sb <- solo[i, ]
  n0 <- lm(y ~ 1, data = sb); nc <- lm(y ~ ., data = sb)
  m  <- step(n0, scope = list(lower = n0, upper = nc),
             direction = "forward", trace = 0)
  vs <- setdiff(names(coef(m)), "(Intercept)")
  cont[vs] <- cont[vs] + 1; tam[b] <- length(vs)
}
sort(round(100*cont/1000, 1), decreasing = TRUE)
mean(tam); range(tam)

# ---- 5. O ERRO DE PREDIÇÃO: a seleção tem de entrar na validação
set.seed(3)
k <- 10; d <- sample(rep(1:k, length.out = nrow(solo)))
er_an <- er_fix <- numeric(nrow(solo))
fixo <- setdiff(names(coef(frente)), "(Intercept)")
for (j in 1:k) {
  tr <- solo[d != j, ]; te <- solo[d == j, ]
  n0 <- lm(y ~ 1, data = tr); nc <- lm(y ~ ., data = tr)
  m  <- step(n0, scope = list(lower = n0, upper = nc),
             direction = "forward", trace = 0)
  er_an[d == j]  <- te$y - predict(m, newdata = te)
  er_fix[d == j] <- te$y - predict(lm(reformulate(fixo, "y"), data = tr), newdata = te)
}
mean(er_fix^2)     # OTIMISTA
mean(er_an^2)      # honesto

# ---- 6. O REGIME EM QUE FRACASSA POR COMPLETO: p próximo de n
set.seed(23)
res <- t(replicate(200, {
  Z <- as.data.frame(matrix(rnorm(50*45), 50, 45))
  Z$y <- rnorm(50)
  n0 <- lm(y ~ 1, data = Z); nc <- lm(y ~ ., data = Z)
  m  <- step(n0, scope = list(lower = n0, upper = nc),
             direction = "forward", trace = 0)
  c(k = length(coef(m)) - 1, r2 = summary(m)$r.squared) }))
colMeans(res); max(res[, "r2"])
# NENHUMA das 45 candidatas tem relação com y.
