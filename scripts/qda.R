# Análise discriminante quadrática: exemplo completo ----
# Pacotes: MASS (qda), biotools (M de Box), klaR (fronteiras), caret
# install.packages(c("MASS", "biotools", "klaR", "caret"))

data(iris)

# O pressuposto que distingue a QDA da LDA: covariâncias iguais ou não?
by(iris[, 1:4], iris$Species, function(x) round(det(cov(x)), 8))
library(biotools)
boxM(iris[, 1:4], iris$Species)

# Os dois ajustes, lado a lado
library(MASS)
ml <- lda(Species ~ ., data = iris)
mq <- qda(Species ~ ., data = iris)
mq                       # priores e médias; as covariâncias ficam em mq$scaling

# Reclassificação (otimista) e validação cruzada (a que se reporta)
table(Real = iris$Species, Predita = predict(ml)$class)
table(Real = iris$Species, Predita = predict(mq)$class)
lcv <- lda(Species ~ ., data = iris, CV = TRUE)
qcv <- qda(Species ~ ., data = iris, CV = TRUE)
mean(lcv$class == iris$Species)
mean(qcv$class == iris$Species)
table(LDA = lcv$class, QDA = qcv$class)        # onde as duas discordam
which(lcv$class != qcv$class)

# Um caso em que a QDA é claramente superior: mesma média, dispersões
# muito diferentes (dados fictícios)
set.seed(7)
A <- MASS::mvrnorm(200, c(0, 0), diag(2))        # compacto
B <- MASS::mvrnorm(200, c(0, 0), 9 * diag(2))    # disperso
dd <- data.frame(x1 = c(A[,1], B[,1]), x2 = c(A[,2], B[,2]),
                 g  = factor(rep(c("A", "B"), each = 200)))
boxM(dd[, 1:2], dd$g)
mean(lda(g ~ x1 + x2, data = dd, CV = TRUE)$class == dd$g)
mean(qda(g ~ x1 + x2, data = dd, CV = TRUE)$class == dd$g)
library(klaR)
partimat(g ~ x1 + x2, data = dd, method = "lda")
partimat(g ~ x1 + x2, data = dd, method = "qda")

# Quando n é pequeno, a QDA pode PERDER mesmo com o modelo a favor dela
# (a simulação completa está no código Python desta ficha)
set.seed(11)
sim <- function(n, c_, rep = 200, p = 5) {
  res <- replicate(rep, {
    Z <- rbind(MASS::mvrnorm(n, rep(0, p), diag(p)),
               MASS::mvrnorm(n, rep(0.7, p), c_ * diag(p)))
    g <- factor(rep(c("A", "B"), each = n))
    Tt <- rbind(MASS::mvrnorm(1500, rep(0, p), diag(p)),
                MASS::mvrnorm(1500, rep(0.7, p), c_ * diag(p)))
    gt <- factor(rep(c("A", "B"), each = 1500))
    d1 <- data.frame(Z, g); d2 <- data.frame(Tt)
    names(d2) <- names(d1)[1:p]
    c(mean(predict(lda(g ~ ., data = d1), d2)$class != gt),
      mean(predict(qda(g ~ ., data = d1), d2)$class != gt))
  })
  rowMeans(res)
}
# sim(20, 1.4); sim(100, 1.4); sim(20, 3.0)

# Análise discriminante REGULARIZADA: interpola entre LDA e QDA
library(klaR)
rda(Species ~ ., data = iris)         # estima lambda e gamma por validação

# Validação honesta com treino e teste
library(caret)
set.seed(1)
idx <- createDataPartition(iris$Species, p = 0.7, list = FALSE)
confusionMatrix(predict(qda(Species ~ ., data = iris[idx, ]),
                        iris[-idx, ])$class, iris[-idx, ]$Species)
