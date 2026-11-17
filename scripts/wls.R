# Mínimos quadrados ponderados: exemplo completo ----
# Pacote: apenas o stats (já vem com o R).

# 1. Dados: 20 lotes de produção (dados fictícios)
dados <- data.frame(
  teor = c(0.5, 0.7, 1.0, 1.2, 1.4, 1.7, 1.9, 2.2, 2.4, 2.6,
           2.9, 3.1, 3.3, 3.6, 3.8, 4.1, 4.3, 4.5, 4.8, 5.0),
  n    = c(2, 20, 12, 18, 18, 2, 12, 4, 15, 20, 5, 3, 10, 4, 10, 3, 5, 8, 15, 8),
  resistencia = c(38.24, 32.48, 33.92, 34.87, 34.61, 40.46, 37.01, 34.39, 40.48, 41.85,
                  40.08, 44.83, 42.93, 45.74, 43.56, 44.85, 47.84, 45.71, 49.73, 50.30))

# 2. Mínimos quadrados comuns: todos os lotes com o mesmo peso
m_ols <- lm(resistencia ~ teor, data = dados)
summary(m_ols); confint(m_ols)

# 3. Mínimos quadrados ponderados: peso proporcional ao número de medições
m_wls <- lm(resistencia ~ teor, data = dados, weights = n)
summary(m_wls); confint(m_wls)

# 4. Ganho de precisão na inclinação
c(ep_comum = summary(m_ols)$coefficients[2, 2],
  ep_ponderado = summary(m_wls)$coefficients[2, 2])

# 5. Resíduos padronizados: só os ponderados são comparáveis entre lotes
par(mfrow = c(1, 2))
plot(dados$n, rstandard(m_ols), pch = 19, xlab = "n do lote", ylab = "Resíduo padronizado (comum)")
abline(h = 0, col = "red")
plot(dados$n, rstandard(m_wls), pch = 19, xlab = "n do lote", ylab = "Resíduo padronizado (ponderado)")
abline(h = 0, col = "red")
par(mfrow = c(1, 1))

# 6. Pesos vindos de uma variância conhecida: w = 1 / s²
m_var <- lm(resistencia ~ teor, data = dados, weights = 1 / (18 / n))
coef(m_var)          # idêntico ao item 3: multiplicar os pesos por uma constante não muda nada

# 7. Quando a variância não é conhecida, estima-se em duas etapas (FGLS)
res2 <- residuals(m_ols)^2
aux  <- lm(log(res2) ~ teor, data = dados)          # modelo para a variância
m_fgls <- lm(resistencia ~ teor, data = dados, weights = 1 / exp(fitted(aux)))
coef(m_fgls)

# 8. Alternativa sem pesos: erros-padrão robustos à heterocedasticidade
# install.packages(c("sandwich", "lmtest"))
# lmtest::coeftest(m_ols, vcov = sandwich::vcovHC(m_ols, type = "HC3"))

# 9. Gráfico: as duas retas, com o tamanho do ponto proporcional a n
plot(resistencia ~ teor, data = dados, pch = 19, cex = sqrt(dados$n) / 2,
     xlab = "Teor de aditivo (%)", ylab = "Resistência média (MPa)")
abline(m_ols, lty = 2); abline(m_wls, col = "red", lwd = 2)
legend("topleft", c("comum", "ponderado"), lty = c(2, 1), col = c("black", "red"), bty = "n")
