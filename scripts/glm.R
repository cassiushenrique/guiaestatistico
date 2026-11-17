# Modelos lineares generalizados: exemplo completo ----
# Pacote: apenas o stats (já vem com o R).

# 1. Dados: besouros expostos a dissulfeto de carbono (Bliss, 1935)
dados <- data.frame(
  ldose    = c(1.6907, 1.7242, 1.7552, 1.7842, 1.8113, 1.8369, 1.8610, 1.8839),
  expostos = c(59, 60, 62, 56, 63, 59, 62, 60),
  mortos   = c(6, 13, 18, 28, 52, 53, 61, 60))
dados$proporcao <- dados$mortos / dados$expostos
dados

# 2. Resposta binomial agrupada: cbind(sucessos, fracassos)
resposta <- cbind(dados$mortos, dados$expostos - dados$mortos)

# 3. A mesma família com três ligações diferentes
m_logit   <- glm(resposta ~ ldose, data = dados, family = binomial(link = "logit"))
m_probit  <- glm(resposta ~ ldose, data = dados, family = binomial(link = "probit"))
m_cloglog <- glm(resposta ~ ldose, data = dados, family = binomial(link = "cloglog"))
summary(m_logit)
summary(m_cloglog)

# 4. Comparação dos ajustes pela deviance e pelo AIC
sapply(list(logit = m_logit, probit = m_probit, cloglog = m_cloglog),
       function(m) c(deviance = deviance(m), gl = df.residual(m), AIC = AIC(m)))

# 5. Teste de ajuste: a deviance segue aproximadamente uma qui-quadrado
pchisq(deviance(m_cloglog), df.residual(m_cloglog), lower.tail = FALSE)

# 6. Dose letal 50% (log10 e na escala original)
b <- coef(m_cloglog)
ld50 <- (log(-log(0.5)) - b[1]) / b[2]
c(log10_LD50 = ld50, LD50 = 10^ld50)

# 7. Outras famílias usam exatamente a mesma sintaxe
# glm(y ~ x, family = gaussian)                   # regressão linear
# glm(y ~ x, family = poisson)                    # contagens
# glm(y ~ x, family = Gamma(link = "log"))        # resposta contínua positiva e assimétrica
# glm(cbind(s, f) ~ x, family = binomial)         # proporções

# 8. Diagnóstico
par(mfrow = c(2, 2)); plot(m_cloglog); par(mfrow = c(1, 1))

# 9. Gráfico: as três curvas sobre os dados
plot(proporcao ~ ldose, data = dados, pch = 19, ylim = c(0, 1),
     xlab = "log10 da dose", ylab = "Proporção de mortos")
grade <- data.frame(ldose = seq(1.65, 1.92, length.out = 200))
lines(grade$ldose, predict(m_logit, grade, type = "response"), col = "darkgreen", lwd = 2)
lines(grade$ldose, predict(m_probit, grade, type = "response"), col = "orange", lwd = 2)
lines(grade$ldose, predict(m_cloglog, grade, type = "response"), col = "red", lwd = 2)
legend("topleft", c("logit", "probit", "cloglog"),
       col = c("darkgreen", "orange", "red"), lwd = 2, bty = "n")
