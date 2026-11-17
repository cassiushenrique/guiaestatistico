# Teste t pareado: exemplo completo ----
# Pacote: apenas o stats (já vem com o R). Dados: sleep.

# 1. Dados: horas extras de sono de 10 pacientes com dois medicamentos
#    Fonte original: Cushny e Peebles (1905); usados por Student (1908).
medicamento1 <- sleep$extra[sleep$group == 1]
medicamento2 <- sleep$extra[sleep$group == 2]
d <- medicamento2 - medicamento1
data.frame(paciente = 1:10, medicamento1, medicamento2, diferenca = d)
c(media_das_diferencas = mean(d), dp_das_diferencas = sd(d))

# 2. Diagnóstico gráfico: perfis individuais e distribuição das diferenças
par(mfrow = c(1, 2))
matplot(rbind(medicamento1, medicamento2), type = "b", pch = 19, lty = 1, col = "gray50",
        xaxt = "n", xlab = "", ylab = "Horas extras de sono")
axis(1, at = 1:2, labels = c("medicamento 1", "medicamento 2"))
lines(c(mean(medicamento1), mean(medicamento2)), col = "red", lwd = 3, type = "b", pch = 19)
hist(d, col = "gray85", xlab = "Diferença (h)", main = ""); abline(v = 0, col = "red", lwd = 2)
par(mfrow = c(1, 1))

# 3. Pressuposto: normalidade das DIFERENÇAS (não das medidas originais)
shapiro.test(d)
qqnorm(d, pch = 19); qqline(d, col = "red", lwd = 2)

# 4. Teste t pareado
t.test(medicamento2, medicamento1, paired = TRUE)

# 5. É exatamente um teste t de uma amostra sobre as diferenças
t.test(d, mu = 0)

# 6. O que se perderia ignorando o pareamento
t.test(medicamento2, medicamento1, var.equal = TRUE)
cor(medicamento1, medicamento2)        # é esta correlação que o pareamento aproveita

# 7. Tamanho do efeito
mean(d) / sd(d)

# 8. Alternativas não paramétricas
wilcox.test(d, mu = 0)
binom.test(sum(d > 0), sum(d != 0))    # teste do sinal
