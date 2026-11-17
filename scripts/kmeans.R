# k-médias: exemplo completo ----
# Pacotes: stats (base), cluster, factoextra, mclust (índice de Rand)
# install.packages(c("cluster", "factoextra", "mclust"))

# PARTE 1 - o algoritmo, passo a passo, em doze pontos
P <- matrix(c(1.0,1.0, 1.4,0.8, 0.8,1.5, 1.2,1.3,
              5.0,5.2, 5.4,4.8, 4.8,5.5, 5.2,5.0,
              1.0,5.0, 1.3,4.6, 0.7,5.3, 1.1,4.9),
            ncol = 2, byrow = TRUE, dimnames = list(NULL, c("x1", "x2")))
P
plot(P, pch = 19, asp = 1, main = "Doze amostras")

# Centros iniciais propositalmente ruins: os três no mesmo canto
ini <- matrix(c(1.0,1.0, 1.4,0.8, 0.8,1.5), ncol = 2, byrow = TRUE)
km_ruim <- kmeans(P, centers = ini, iter.max = 20, algorithm = "Lloyd")
km_ruim$cluster
km_ruim$centers
km_ruim$tot.withinss          # a inércia final
km_ruim$iter

# Uma partida só pode cair num mínimo local; nstart resolve
set.seed(1)
km <- kmeans(P, centers = 3, nstart = 25)
km$tot.withinss
km$centers
km$size

# Quantas partidas caem no mínimo global? (experimento)
inercias <- replicate(2000, kmeans(P, centers = 3, nstart = 1,
                                   algorithm = "Lloyd")$tot.withinss)
round(sort(table(round(inercias, 4)), decreasing = TRUE) / 2000, 4)

# PARTE 2 - iris: 150 flores, quatro medidas, sem usar a espécie
data(iris)
X <- iris[, 1:4]
set.seed(7)

# Escolha de k: inércia, silhueta e Calinski-Harabasz
library(cluster); library(factoextra)
res <- t(sapply(2:7, function(k) {
  km <- kmeans(X, centers = k, nstart = 50)
  sil <- mean(silhouette(km$cluster, dist(X))[, 3])
  c(k = k, inercia = km$tot.withinss, silhueta = sil)
}))
round(res, 4)
fviz_nbclust(X, kmeans, method = "wss")          # cotovelo
fviz_nbclust(X, kmeans, method = "silhouette")   # silhueta

# A solução com k = 3
km3 <- kmeans(X, centers = 3, nstart = 50)
km3$centers
km3$size
table(iris$Species, km3$cluster)

library(mclust)
adjustedRandIndex(iris$Species, km3$cluster)              # dados originais
adjustedRandIndex(iris$Species, kmeans(scale(X), 3, nstart = 50)$cluster)

fviz_cluster(km3, data = X, ellipse.type = "norm")

# PARTE 3 - o que o k-médias não consegue fazer (dados fictícios)
set.seed(3)
# (a) dois anéis concêntricos
ang <- runif(300, 0, 2*pi); r <- c(rnorm(150, 1, 0.08), rnorm(150, 3, 0.08))
An <- cbind(r*cos(ang), r*sin(ang))
plot(An, col = kmeans(An, 2, nstart = 25)$cluster, pch = 19, asp = 1,
     main = "Anéis concêntricos: o k-médias corta ao meio")
# (b) grupos alongados, deslocados perpendicularmente ao eixo longo
library(MASS)
S  <- matrix(c(9, 8.5, 8.5, 9), 2)
Al <- rbind(mvrnorm(150, c(0, 0), S), mvrnorm(150, c(1.6, -1.6), S))
plot(Al, col = kmeans(Al, 2, nstart = 25)$cluster, pch = 19, asp = 1)
# (c) tamanhos e dispersões muito desiguais
Bd <- rbind(cbind(rnorm(285, 0, 1.2), rnorm(285, 0, 1.2)),
            cbind(rnorm(15, 2.5, 0.3), rnorm(15, 0, 0.3)))
plot(Bd, col = kmeans(Bd, 2, nstart = 25)$cluster, pch = 19, asp = 1)

# Alternativas para esses casos
# kernlab::specc()   - agrupamento espectral (resolve os anéis)
# dbscan::dbscan()   - densidade (resolve os anéis e identifica ruído)
# mclust::Mclust()   - misturas de gaussianas (resolve os alongados)
# cluster::pam()     - k-medoides, mais robusto a atípicos
