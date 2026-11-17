# Teste t de Welch: exemplo completo ----
# Pacote: apenas o stats (já vem com o R). Dados: mtcars.

# 1. Dados: consumo (mpg) por tipo de câmbio (am: 0 = automático, 1 = manual)
dados <- mtcars
dados$cambio <- factor(dados$am, levels = c(0, 1), labels = c("automatico", "manual"))
aggregate(mpg ~ cambio, data = dados, FUN = function(v) c(n = length(v), media = mean(v), dp = sd(v), variancia = var(v)))

# 2. O teste de Welch é o PADRÃO do R: não é preciso argumento nenhum
t.test(mpg ~ cambio, data = dados)

# 3. Cálculo manual do erro-padrão e dos graus de liberdade
g  <- split(dados$mpg, dados$cambio)
n1 <- length(g$automatico); n2 <- length(g$manual)
a  <- var(g$automatico) / n1; b <- var(g$manual) / n2
ep <- sqrt(a + b)
gl <- (a + b)^2 / (a^2 / (n1 - 1) + b^2 / (n2 - 1))
c(erro_padrao = ep, gl_Welch = gl,
  t = (mean(g$automatico) - mean(g$manual)) / ep,
  gl_minimo = min(n1, n2) - 1, gl_Student = n1 + n2 - 2)

# 4. Comparação com o teste de Student
t.test(mpg ~ cambio, data = dados, var.equal = TRUE)

# 5. Pressupostos
tapply(dados$mpg, dados$cambio, function(v) shapiro.test(v)$p.value)
# install.packages("car"); car::leveneTest(mpg ~ cambio, data = dados)

# 6. Tamanho do efeito
s2p <- ((n1 - 1) * var(g$automatico) + (n2 - 1) * var(g$manual)) / (n1 + n2 - 2)
c(d_de_Cohen = (mean(g$automatico) - mean(g$manual)) / sqrt(s2p),
  delta_de_Glass = (mean(g$automatico) - mean(g$manual)) / sd(g$automatico))

# 7. Simulação: taxa de erro tipo I com médias IGUAIS e variâncias diferentes
set.seed(2026)
sim <- replicate(5000, {
  g1 <- rnorm(20, 0, 1); g2 <- rnorm(8, 0, 3)
  c(student = t.test(g1, g2, var.equal = TRUE)$p.value < 0.05,
    welch   = t.test(g1, g2)$p.value < 0.05) })
rowMeans(sim)      # o nominal é 0,05

# 8. Gráfico
boxplot(mpg ~ cambio, data = dados, col = "gray85", ylab = "Consumo (mpg)")
