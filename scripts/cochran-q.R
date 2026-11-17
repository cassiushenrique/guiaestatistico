# Teste Q de Cochran: exemplo completo ----
# Pacotes: DescTools (teste e comparações), rcompanion (alternativa)
# install.packages(c("DescTools", "rcompanion"))

# Dados fictícios: 18 analistas executaram a MESMA tarefa em quatro versões de
# uma interface, e registrou-se se concluíram (1) ou não (0) dentro do tempo.
dados <- matrix(c(
  1,1,1,0, 1,1,1,1, 0,1,1,0, 1,1,1,1, 0,0,1,0, 1,1,1,1,
  0,1,1,0, 1,1,1,1, 0,0,1,0, 1,1,1,1, 0,1,1,1, 1,1,1,1,
  0,0,1,0, 1,1,1,0, 0,1,1,1, 1,1,1,1, 0,0,0,0, 0,1,1,1),
  ncol = 4, byrow = TRUE,
  dimnames = list(paste0("analista", 1:18),
                  c("versão A", "versão B", "versão C", "versão D")))
dados
colSums(dados)                  # sucessos por versão
colMeans(dados)                 # taxas
rowSums(dados)                  # sucessos por analista
table(rowSums(dados))           # quantos blocos são "mistos"

# O teste
library(DescTools)
CochranQTest(dados)

# A conta, à mão
k  <- ncol(dados); Gj <- colSums(dados); Li <- rowSums(dados); N <- sum(dados)
Q  <- (k - 1) * (k * sum(Gj^2) - N^2) / (k * N - sum(Li^2))
Q
pchisq(Q, df = k - 1, lower.tail = FALSE)

# Blocos uniformes não contribuem: refaça só com os mistos
mistos <- dados[rowSums(dados) > 0 & rowSums(dados) < k, ]
Gm <- colSums(mistos); Lm <- rowSums(mistos); Nm <- sum(mistos)
(k - 1) * (k * sum(Gm^2) - Nm^2) / (k * Nm - sum(Lm^2))   # idêntico

# Verificação: com duas colunas, Q é o McNemar sem correção
duas <- dados[, c("versão A", "versão B")]
CochranQTest(duas)
mcnemar.test(table(duas[,1], duas[,2]), correct = FALSE)

# Comparações pareadas pós-teste (McNemar exato + ajuste)
pares <- combn(colnames(dados), 2)
pv <- apply(pares, 2, function(p) {
  b <- sum(dados[, p[1]] == 1 & dados[, p[2]] == 0)
  c_ <- sum(dados[, p[1]] == 0 & dados[, p[2]] == 1)
  binom.test(b, b + c_, p = 0.5)$p.value
})
data.frame(par = apply(pares, 2, paste, collapse = " - "),
           bruto = round(pv, 5),
           bonferroni = round(p.adjust(pv, "bonferroni"), 5),
           holm = round(p.adjust(pv, "holm"), 5))

# Concordância entre os analistas (W de Kendall)
DescTools::KendallW(dados, correct = TRUE, test = TRUE)

# O mesmo teste pela via do Friedman: com resposta binária, coincide
friedman.test(dados)

# Formato longo, se precisar
longo <- data.frame(
  analista = factor(rep(rownames(dados), times = 4)),
  versao   = factor(rep(colnames(dados), each = 18)),
  sucesso  = as.vector(dados))
library(rcompanion)
# pairwiseMcnemar(sucesso ~ versao | analista, data = longo, method = "holm")

# Gráficos
par(mfrow = c(1, 2))
barplot(colMeans(dados) * 100, col = "steelblue", ylim = c(0, 100),
        ylab = "Taxa de conclusão (%)", main = "Por versão")
matplot(t(dados), type = "l", lty = 1, col = rgb(0, 0, 0, 0.3),
        xaxt = "n", ylab = "Sucesso", main = "Perfis dos analistas")
axis(1, at = 1:4, labels = colnames(dados))
par(mfrow = c(1, 1))
