# Modelos aditivos generalizados (GAM): exemplo completo ----
# Pacotes: mgcv (Simon Wood) e gam (Trevor Hastie)
# Dados fictícios: 260 medições de trocas gasosas em folhas de cafeeiro.
# O arquivo fotossintese.csv acompanha o site, na pasta scripts/.
fot <- read.csv("fotossintese.csv")
str(fot)            # A (umol CO2 m-2 s-1), par, temp, vpd
summary(fot)
plot(A ~ par, data = fot, pch = 16, col = "#8aa59680",
     xlab = "PAR (umol m-2 s-1)", ylab = "A (umol CO2 m-2 s-1)")

# O ponto de partida: quanto o modelo LINEAR deixa na mesa
summary(lm(A ~ par + temp + vpd, data = fot))$r.squared   # 0,6747
# O processo gerador verdadeiro daria R² = 0,9514. É esse buraco que o
# grupo inteiro de fichas tenta preencher.

# ---- 1. O ponto de partida: quanto o modelo linear deixa na mesa
m_lin <- lm(A ~ par + temp + vpd, data = fot)
summary(m_lin)$r.squared              # 0,6747

# ---- 2. O GAM
library(mgcv)
m <- gam(A ~ s(par) + s(temp) + s(vpd), data = fot, method = "REML")
summary(m)
# A tabela de termos suaves traz, para cada parcela, os graus de
# liberdade efetivos (edf) e um teste F aproximado.
plot(m, pages = 1, shade = TRUE, seWithMean = TRUE, residuals = TRUE)
gam.check(m)        # diagnostico: k suficiente? residuos bem comportados?

# O mgcv escolhe a suavidade SOZINHO, por REML ou GCV. O argumento k de
# s() e o numero MAXIMO de graus de liberdade da base, nao o usado:
m2 <- gam(A ~ s(par, k = 20) + s(temp, k = 20) + s(vpd, k = 20),
          data = fot, method = "REML")
cbind(k10 = summary(m)$edf, k20 = summary(m2)$edf)
# Se o edf encostar em k-1, aumente k. E o que gam.check verifica.

# ---- 3. O retroajuste, no pacote gam de Hastie
library(gam)
mh <- gam::gam(A ~ s(par, df = 12) + s(temp, df = 8) + s(vpd, df = 6),
               data = fot)
summary(mh)
# Aqui os g.l. sao DADOS pelo usuario; no mgcv sao estimados.

# ---- 4. COMPARAÇÃO com os modelos mais simples
AIC(m_lin, m)
anova(m_lin, m, test = "F")
# E por validacao cruzada, que e o criterio honesto:
set.seed(1)
d <- sample(rep(1:10, length.out = nrow(fot)))
er_l <- er_g <- numeric(nrow(fot))
for (j in 1:10) {
  tr <- fot[d != j, ]; te <- fot[d == j, ]
  er_l[d == j] <- te$A - predict(lm(A ~ par + temp + vpd, data = tr), newdata = te)
  er_g[d == j] <- te$A - predict(gam(A ~ s(par) + s(temp) + s(vpd),
                                     data = tr, method = "REML"), newdata = te) }
c(linear = mean(er_l^2), gam = mean(er_g^2))

# ---- 5. CONCURVIDADE: a colinearidade dos modelos aditivos
concurvity(m, full = TRUE)
concurvity(m, full = FALSE)       # par a par
# O VIF so enxerga relacoes LINEARES. A concurvidade mede o quanto uma
# parcela pode ser aproximada pelas outras, por qualquer funcao suave.
cor(fot$temp, fot$vpd)                                   # 0,5794
summary(gam(vpd ~ s(temp), data = fot))$r.sq             # 0,4482

# ---- 6. O QUE A HIPÓTESE ADITIVA EXCLUI: interações
# Um termo de interacao suave em duas dimensoes:
mi <- gam(A ~ te(par, temp) + s(vpd), data = fot, method = "REML")
AIC(m, mi)
anova(m, mi, test = "F")
# Diagnostico barato: o residuo do GAM aditivo guarda estrutura?
res <- residuals(m)
cor(res, (fot$par - mean(fot$par))*(fot$temp - mean(fot$temp)))

# ---- 7. Para outras famílias
# O "G" de GAM e de generalizado: a mesma estrutura vale para resposta
# binaria, de contagem, etc.
#   gam(doente ~ s(idade) + s(dose), family = binomial, data = d)
#   gam(n_insetos ~ s(temp) + s(umidade), family = poisson, data = d)
