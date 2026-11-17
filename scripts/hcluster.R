# Agrupamento hierárquico: exemplo completo ----
# Pacotes: stats (base), cluster, factoextra, dendextend
# install.packages(c("cluster", "factoextra", "dendextend"))

# PARTE 1 - seis amostras, duas medidas: a mecânica do algoritmo
X <- matrix(c(1.0,1.0, 1.5,1.2, 5.0,5.0, 5.5,5.2, 3.0,4.5, 8.0,1.0),
            ncol = 2, byrow = TRUE,
            dimnames = list(letters[1:6], c("x1", "x2")))
X
d <- dist(X)                        # distâncias euclidianas
round(as.matrix(d), 3)

# Os quatro critérios de ligação
hs <- hclust(d, method = "single")
hc <- hclust(d, method = "complete")
ha <- hclust(d, method = "average")
hw <- hclust(d, method = "ward.D2")

# A sequência de fusões: merge e height contam a história
cbind(hs$merge, altura = round(hs$height, 4))
cbind(hw$merge, altura = round(hw$height, 4))

# Fidelidade do dendrograma: correlação cofenética
sapply(list(single = hs, complete = hc, average = ha, ward = hw),
       function(h) cor(d, cophenetic(h)))

par(mfrow = c(2, 2))
plot(hs, main = "ligação simples", xlab = "", sub = "")
plot(hc, main = "ligação completa", xlab = "", sub = "")
plot(ha, main = "ligação média", xlab = "", sub = "")
plot(hw, main = "Ward", xlab = "", sub = "")
par(mfrow = c(1, 1))

# PARTE 2 - iris: 150 flores, quatro medidas. A espécie NÃO entra no
# agrupamento; ela só serve para avaliar o resultado no final.
data(iris)
Z  <- scale(iris[, 1:4])            # padronizado
dZ <- dist(Z);  dO <- dist(iris[, 1:4])

apply(iris[, 1:4], 2, function(v) c(media = mean(v), dp = sd(v), ampl = diff(range(v))))

hw_o <- hclust(dO, method = "ward.D2")
hw_z <- hclust(dZ, method = "ward.D2")
g_o  <- cutree(hw_o, k = 3)
g_z  <- cutree(hw_z, k = 3)
table(iris$Species, g_o)
table(iris$Species, g_z)

# Índice de Rand ajustado: concordância com a espécie, corrigida pelo acaso
library(mclust)                      # ou fossil::adj.rand.index
adjustedRandIndex(iris$Species, g_o)
adjustedRandIndex(iris$Species, g_z)

# Onde cortar? As últimas alturas de fusão e o maior salto
tail(hw_z$height, 10)
plot(rev(tail(hw_z$height, 10)), type = "b", pch = 19,
     xlab = "Número de grupos", ylab = "Altura da fusão")

library(factoextra)
fviz_dend(hw_z, k = 3, rect = TRUE)
fviz_nbclust(Z, FUN = hcut, method = "silhouette")

# Silhueta do agrupamento escolhido
library(cluster)
sil <- silhouette(g_z, dZ)
summary(sil)
plot(sil)

# PARTE 3 - por que a ligação simples costuma falhar: o encadeamento
# Dois blocos separados com uma ponte de pontos entre eles (dados fictícios).
set.seed(42)
A <- cbind(rnorm(30, 0, 0.6), rnorm(30, 0, 0.6))
B <- cbind(rnorm(30, 10, 0.6), rnorm(30, 0, 0.6))
P <- cbind(seq(2.2, 7.8, length.out = 22), rnorm(22, 0, 0.12))
Y <- rbind(A, B, P)
dy <- dist(Y)
sapply(c("single", "complete", "average", "ward.D2"), function(m) {
  h <- hclust(dy, method = m)
  c(table(cutree(h, 2)), razao = h$height[length(h$height)] /
                                 h$height[length(h$height) - 1])
})

# Comparar dois dendrogramas lado a lado
library(dendextend)
tanglegram(as.dendrogram(hclust(dy, "single")),
           as.dendrogram(hclust(dy, "ward.D2")))
