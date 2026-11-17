# Redes neurais recorrentes (RNN): exemplo completo ----
# Pacote: keras3 (exige instalar o Python e o TensorFlow uma única vez)
# install.packages("keras3"); keras3::install_keras()
library(keras3)
url <- "https://cassiushenrique.github.io/ModelosEstatisticos/scripts/clima.csv"
cl <- if (file.exists("clima.csv")) read.csv("clima.csv") else read.csv(url)
tr <- cl[cl$conjunto == "treino", ]; te <- cl[cl$conjunto == "teste", ]
temp_tr <- as.matrix(tr[, paste0("temp", 1:30)]); molh_tr <- as.matrix(tr[, paste0("molh", 1:30)])
temp_te <- as.matrix(te[, paste0("temp", 1:30)]); molh_te <- as.matrix(te[, paste0("molh", 1:30)])

# ---- 1. As referências que ignoram a ordem
fav <- function(t, m) (m >= 8) & (t >= 19) & (t <= 25)
maior_seq <- function(f) apply(f, 1, function(r) { x <- rle(r); max(c(0, x$lengths[x$values])) })
d_tr <- data.frame(surto = tr$surto, n_fav = rowSums(fav(temp_tr, molh_tr)),
                   seq = maior_seq(fav(temp_tr, molh_tr)))
d_te <- data.frame(surto = te$surto, n_fav = rowSums(fav(temp_te, molh_te)),
                   seq = maior_seq(fav(temp_te, molh_te)))
g_cont <- glm(surto ~ n_fav, data = d_tr, family = binomial)
mean((predict(g_cont, d_te, type = "response") >= 0.5) != d_te$surto)   # conta os dias
g_seq <- glm(surto ~ seq, data = d_tr, family = binomial)               # o "oráculo"
mean((predict(g_seq, d_te, type = "response") >= 0.5) != d_te$surto)

# ---- 2. A RNN nos dados brutos: entradas (talhões, 30 dias, 2 variáveis)
mt <- mean(temp_tr); st <- sd(temp_tr); mm <- mean(molh_tr); sm <- sd(molh_tr)
arr <- function(t, m) { a <- array(0, c(nrow(t), 30, 2))
  a[, , 1] <- (t - mt) / st; a[, , 2] <- (m - mm) / sm; a }
Xtr <- arr(temp_tr, molh_tr); Xte <- arr(temp_te, molh_te)
set_random_seed(1)
rnn <- keras_model_sequential(input_shape = c(30, 2)) |>
  layer_simple_rnn(units = 16, activation = "tanh") |>
  layer_dense(units = 1, activation = "sigmoid")
summary(rnn)                                   # 321 parâmetros
rnn |> compile(optimizer = optimizer_adam(learning_rate = 0.003, clipnorm = 1),
               loss = "binary_crossentropy", metrics = "accuracy")
rnn |> fit(Xtr, tr$surto, epochs = 150, batch_size = 64, validation_split = 0.2, verbose = 0)
rnn |> evaluate(Xte, te$surto)

# ---- 3. A RNN no indicador de dia favorável (+1 / -1)
Ftr <- array(2 * fav(temp_tr, molh_tr) - 1, c(nrow(tr), 30, 1))
Fte <- array(2 * fav(temp_te, molh_te) - 1, c(nrow(te), 30, 1))
rnn2 <- keras_model_sequential(input_shape = c(30, 1)) |>
  layer_simple_rnn(16) |> layer_dense(1, activation = "sigmoid")
rnn2 |> compile(optimizer = optimizer_adam(0.001, clipnorm = 1), loss = "binary_crossentropy",
                metrics = "accuracy")
rnn2 |> fit(Ftr, tr$surto, epochs = 150, batch_size = 64, verbose = 0)
rnn2 |> evaluate(Fte, te$surto)

# ---- 4. A rede usa a ordem? Avalie com os dias embaralhados
set.seed(9); o <- sample(30)
rnn2 |> evaluate(Fte[, o, , drop = FALSE], te$surto)

# ---- 5. As variantes com portões: troque layer_simple_rnn por
#   layer_lstm(16)   ou   layer_gru(16)
