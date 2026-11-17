# Hochberg GT2: exemplo completo ----
# Pacotes: stats (base, para a implementação direta). O SPSS traz o GT2 pronto;
# no R, o caminho mais curto é calcular o módulo máximo estudentizado.
# Alternativas: mutoss, SMMT (CRAN), ou integração numérica, como abaixo.

# 1. A distribuição do módulo máximo estudentizado, por integração numérica
smm_cdf <- function(h, m, nu) {
  integrate(function(s) (2 * pnorm(h * s) - 1)^m * dchisq(s^2 * nu, nu) * 2 * nu * s,
            0, Inf)$value
}
smm_ppf <- function(p, m, nu) uniroot(function(h) smm_cdf(h, m, nu) - p, c(0.5, 12))$root

# 2. Dados: chickwts, peso de pintinhos por tipo de ração
data(chickwts)
modelo <- aov(weight ~ feed, data = chickwts)
summary(modelo)
k  <- nlevels(chickwts$feed); gl <- df.residual(modelo)
QM <- sum(residuals(modelo)^2) / gl
m  <- k * (k - 1) / 2
table(chickwts$feed)            # n desiguais: 10 a 14

# 3. O valor crítico
alfa <- 0.05
(crit <- smm_ppf(1 - alfa, m, gl))
# verificação: com m = 1 ele deve coincidir com o t bilateral
c(smm = smm_ppf(1 - alfa, 1, gl), t = qt(1 - alfa/2, gl))

# 4. Lado a lado com os demais multiplicadores
c(sem_ajuste = qt(1 - alfa/2, gl),
  tukey      = qtukey(1 - alfa, k, gl) / sqrt(2),
  gt2        = crit,
  sidak      = qt(1 - (1 - (1 - alfa)^(1/m))/2, gl),
  bonferroni = qt(1 - alfa/(2*m), gl),
  scheffe    = sqrt((k - 1) * qf(1 - alfa, k - 1, gl)))

# 5. As comparações
medias <- tapply(chickwts$weight, chickwts$feed, mean)
ns <- table(chickwts$feed)
pares <- combn(levels(chickwts$feed), 2)
resultado <- t(apply(pares, 2, function(p_) {
  d  <- medias[p_[2]] - medias[p_[1]]
  ep <- sqrt(QM * (1/ns[p_[1]] + 1/ns[p_[2]]))
  t_ <- abs(d) / ep
  c(diferenca = d, ep = ep, t = t_,
    p_ajustado = 1 - smm_cdf(t_, m, gl),
    inferior = d - crit * ep, superior = d + crit * ep)
}))
rownames(resultado) <- apply(pares, 2, paste, collapse = " - ")
round(resultado[order(resultado[, "p_ajustado"]), ], 5)

# 6. Para comparar, Tukey-Kramer
TukeyHSD(modelo)

# 7. O procedimento de Gabriel, uma variante do GT2
#    usa o mesmo valor crítico, mas o erro-padrão de cada MÉDIA, e não do par:
#    os intervalos de cada média são desenhados e a sobreposição decide.
eps <- sqrt(QM / ns)
ic_gabriel <- cbind(inferior = medias - crit * eps / sqrt(2),
                    superior = medias + crit * eps / sqrt(2))
round(ic_gabriel, 2)

# 8. Gráfico
par(mar = c(5, 11, 4, 2))
plot(TukeyHSD(modelo), las = 1, cex.axis = 0.7)
par(mar = c(5, 4, 4, 2))
