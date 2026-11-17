# Redes neurais convolucionais (CNN): exemplo completo ----
# Pacote: keras3 (exige instalar o Python e o TensorFlow uma única vez)
# install.packages("keras3"); keras3::install_keras()
library(keras3)
url <- "https://cassiushenrique.github.io/ModelosEstatisticos/scripts/folhas.csv"
fol <- if (file.exists("folhas.csv")) read.csv("folhas.csv") else read.csv(url)
table(fol$conjunto, fol$classe)
img <- function(d) array_reshape(as.matrix(d[, paste0("p", 1:256)]), c(nrow(d), 16, 16, 1))
tr <- fol[fol$conjunto == "treino", ]; te <- fol[fol$conjunto == "teste", ]
ds <- fol[fol$conjunto == "deslocado", ]
Xtr <- img(tr); Xte <- img(te); Xds <- img(ds)
image(t(Xtr[1, 16:1, , 1]), col = gray.colors(64), axes = FALSE)   # uma folha

# ---- 1. CNN com máximo GLOBAL: 8 filtros 3x3, ReLU, máximo em toda a imagem
set.seed(1); set_random_seed(1)
cnn <- keras_model_sequential(input_shape = c(16, 16, 1)) |>
  layer_conv_2d(filters = 8, kernel_size = c(3, 3), activation = "relu") |>
  layer_global_max_pooling_2d() |>
  layer_dense(units = 1, activation = "sigmoid")
summary(cnn)                          # 89 parâmetros
cnn |> compile(optimizer = optimizer_adam(0.01), loss = "binary_crossentropy",
               metrics = "accuracy")
hist <- cnn |> fit(Xtr, tr$classe, epochs = 40, batch_size = 32, validation_split = 0.2)
plot(hist)
cnn |> evaluate(Xte, te$classe)       # teste do mesmo tipo
cnn |> evaluate(Xds, ds$classe)       # teste DESLOCADO

# ---- 2. CNN com máximo em blocos 2x2 seguido de camada densa
cnn2 <- keras_model_sequential(input_shape = c(16, 16, 1)) |>
  layer_conv_2d(8, c(3, 3), activation = "relu") |>
  layer_max_pooling_2d(c(2, 2)) |>
  layer_flatten() |>
  layer_dense(1, activation = "sigmoid")
cnn2 |> compile(optimizer = optimizer_adam(0.01), loss = "binary_crossentropy", metrics = "accuracy")
cnn2 |> fit(Xtr, tr$classe, epochs = 40, batch_size = 32, verbose = 0)
cnn2 |> evaluate(Xds, ds$classe)

# ---- 3. Referências: logística e MLP nos 256 pixels
mlp <- keras_model_sequential(input_shape = 256) |>
  layer_dense(64, activation = "relu") |> layer_dense(1, activation = "sigmoid")
mlp |> compile(optimizer = "adam", loss = "binary_crossentropy", metrics = "accuracy")
mlp |> fit(array_reshape(Xtr, c(800, 256)), tr$classe, epochs = 100, verbose = 0)
mlp |> evaluate(array_reshape(Xds, c(400, 256)), ds$classe)

# ---- 4. Os filtros aprendidos
W <- get_weights(cnn)[[1]]            # 3 x 3 x 1 x 8
par(mfrow = c(2, 4)); for (k in 1:8) image(W[, , 1, k], main = paste("filtro", k))
