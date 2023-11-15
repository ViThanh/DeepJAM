# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# Author:       Vi Thanh Pham
# GitHub:       https://github.com/ViThanh
# Description:  Example how to run the DeepJAM algorithm
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# Path --------------------------------------------------------------------
path <- paste0(paste(head(strsplit(rstudioapi::getSourceEditorContext()$path, split = "/")[[1]], -1), collapse = "/"), "/")

# Source functions --------------------------------------------------------
source(paste0(path, "Code/Functions.R"))
source(paste0(path, "Code/Plot_results.R"))
tf$config$experimental$enable_op_determinism() # to ensure reproducibility
tf$config$experimental$enable_tensor_float_32_execution(F) # to ensure reproducibility

# -------------------------------------------------------------------------
# Univariate data ---------------------------------------------------------
# -------------------------------------------------------------------------
data_type <- "univariate"
# Load data
load(paste0(path, "Data/", data_type, ".RData"))
K_true <- K
# K is the number of period
# mu is the common template

# Calculate number of functions per observation
dims <- dim(f_train)
L <- dims[2]
n <- ifelse(length(dims) > 2, dims[3], 1)

# Calculate SRSF
for(i in c("train", "val", "test")){
  assign(paste("q", i, sep = "_"), SRSF(eval(parse(text = paste("f", i, sep = "_")))))
}

# Define model (example architecture)
warp_kernel <- ceiling((L + K_true - 1) / K_true)
layers <- ceiling(2 * L / (warp_kernel - 1))
filters <- rep((n + 3) * K_true, layers)
kernel <- rep(warp_kernel, layers)

# First we show the case when we do not know how many periods are in the data by setting K <- 1
K <- 1
set_random_seed(211918092023 %% .Machine$integer.max, disable_gpu = F)
# The training of the model takes time, for convenience, it is possible to load a saved model
model <- DeepJAM(L = L, n = n, filters = filters, kernel = kernel, warp_kernel = warp_kernel, activation = "tanh") # initialize model
model %>% compile(loss = LossWarping, optimizer = optimizer_adam(learning_rate = 1 / model$count_params())) # compile model
history <- model %>% fit(K = K, x = q_train, validation_data = q_val,
                         epochs = 2500, early_stopping = T, patience = 25, early_stopping_criterion = 0.0001) # fit model
SaveDeepJAM(model, file = paste0(path, "Models/", data_type, "_", K, if(K == 1) "_period" else "_periods")) # save model
saveRDS(history, file = paste0(path, "Models/", data_type, "_", K, if(K == 1) "_period" else "_periods", "/history.RDS")) # save history
model <- LoadDeepJAM(file = paste0(path, "Models/", data_type, "_", K, if(K == 1) "_period" else "_periods")) # load model

# From the plot it is clear, that there are K = 5 periods, so let's set K <- 5
K <- K_true
set_random_seed(133420092023 %% .Machine$integer.max, disable_gpu = F)
# The training of the model takes time, for convenience, it is possible to load a saved model
model <- DeepJAM(L = L, n = n, filters = filters, kernel = kernel, warp_kernel = warp_kernel, activation = "tanh") # initialize model
model %>% compile(loss = LossWarping, optimizer = optimizer_adam(learning_rate = 1 / model$count_params())) # compile model
history <- model %>% fit(K = K, x = q_train, validation_data = q_val,
                         epochs = 2500, early_stopping = T, patience = 25, early_stopping_criterion = 0.0001) # fit model
SaveDeepJAM(model, file = paste0(path, "Models/", data_type, "_", K, if(K == 1) "_period" else "_periods")) # save model
saveRDS(history, file = paste0(path, "Models/", data_type, "_", K, if(K == 1) "_period" else "_periods", "/history.RDS")) # save history
model <- LoadDeepJAM(file = paste0(path, "Models/", data_type, "_", K, if(K == 1) "_period" else "_periods")) # load model

# Finally, let's see what happens when the number of periods is wrong, for example K <- 2
K <- 2
set_random_seed(134220092023 %% .Machine$integer.max, disable_gpu = F)
# The training of the model takes time, for convenience, it is possible to load a saved model
model <- DeepJAM(L = L, n = n, filters = filters, kernel = kernel, warp_kernel = warp_kernel, activation = "tanh") # initialize model
model %>% compile(loss = LossWarping, optimizer = optimizer_adam(learning_rate = 1 / model$count_params())) # compile model
history <- model %>% fit(K = K, x = q_train, validation_data = q_val,
                         epochs = 2500, early_stopping = T, patience = 25, early_stopping_criterion = 0.0001) # fit model
SaveDeepJAM(model, file = paste0(path, "Models/", data_type, "_", K, if(K == 1) "_period" else "_periods")) # save model
saveRDS(history, file = paste0(path, "Models/", data_type, "_", K, if(K == 1) "_period" else "_periods", "/history.RDS")) # save history
model <- LoadDeepJAM(file = paste0(path, "Models/", data_type, "_", K, if(K == 1) "_period" else "_periods")) # load model

# Plot results ------------------------------------------------------------
svg(paste0(path, "Results/", data_type, ".svg"), width = 11.75, height = 8.25)
par(mfrow = c(4, 1), mar = c(2, 0, 2, 0))
K <- K_true
plot(model, f_test, K, type = "observed", mu = mu, s = 100, seed = 114520092023 %% .Machine$integer.max)
plot(LoadDeepJAM(file = paste0(path, "Models/", data_type, "_", K, if(K == 1) "_period" else "_periods")), 
     f_test, K, type = "aligned", mu = mu, s = 100, seed = 114520092023 %% .Machine$integer.max)
K <- 1
plot(LoadDeepJAM(file = paste0(path, "Models/", data_type, "_", K, if(K == 1) "_period" else "_periods")), 
     f_test, K, type = "aligned", mu = mu, s = 100, seed = 114520092023 %% .Machine$integer.max)
K <- 2
plot(LoadDeepJAM(file = paste0(path, "Models/", data_type, "_", K, if(K == 1) "_period" else "_periods")), 
     f_test, K, type = "aligned", mu = mu, s = 100, seed = 114520092023 %% .Machine$integer.max)
dev.off()

# -------------------------------------------------------------------------
# Multivariate data -------------------------------------------------------
# -------------------------------------------------------------------------
data_type <- "multivariate"
# Load data
load(paste0(path, "Data/", data_type, ".RData"))
K_true <- K
# K is the number of period
# mu are the common templates

# Calculate number of functions per observation
dims <- dim(f_train)
L <- dims[2]
n <- ifelse(length(dims) > 2, dims[3], 1)

# Calculate SRSF
for(i in c("train", "val", "test")){
  assign(paste("q", i, sep = "_"), SRSF(eval(parse(text = paste("f", i, sep = "_")))))
}

# Define model (example architecture)
warp_kernel <- ceiling((L + K_true - 1) / K_true)
layers <- ceiling(2 * L / (warp_kernel - 1))
filters <- rep((n + 3) * K_true, layers)
kernel <- rep(warp_kernel, layers)

# Fit model
set_random_seed(141420092023 %% .Machine$integer.max, disable_gpu = F)
# The training of the model takes time, for convenience, it is possible to load a saved model
model <- DeepJAM(L = L, n = n, filters = filters, kernel = kernel, warp_kernel = warp_kernel, activation = "tanh") # initialize model
model %>% compile(loss = LossWarping, optimizer = optimizer_adam(learning_rate = 1 / model$count_params())) # compile model
history <- model %>% fit(K = K, x = q_train, validation_data = q_val,
                         epochs = 2500, early_stopping = T, patience = 25, early_stopping_criterion = 0.0001) # fit model
SaveDeepJAM(model, file = paste0(path, "Models/", data_type, "_", K, if(K == 1) "_period" else "_periods")) # save model
saveRDS(history, file = paste0(path, "Models/", data_type, "_", K, if(K == 1) "_period" else "_periods", "/history.RDS")) # save history
model <- LoadDeepJAM(file = paste0(path, "Models/", data_type, "_", K, if(K == 1) "_period" else "_periods")) # load model

# Plot results ------------------------------------------------------------
svg(paste0(path, "Results/", data_type, ".svg"), width = 11.75, height = 8.25)
par(mfrow = c(n, 2), mar = c(2, 0, 0, 0))
for(j in seq_len(n)){
  plot(model, f_test, K, type = "observed", j = j, mu = mu, s = 100, seed = 114520092023 %% .Machine$integer.max)
  plot(model, f_test, K, type = "aligned", j = j, mu = mu, s = 100, seed = 114520092023 %% .Machine$integer.max)
}
dev.off()

# -------------------------------------------------------------------------
# ECG data ----------------------------------------------------------------
# -------------------------------------------------------------------------
data_type <- "ECG"
# Load data
load(paste0(path, "Data/", data_type, ".RData"))
K_true <- K
# K is the number of period

# Calculate number of functions per observation
dims <- dim(f_train)
L <- dims[2]
n <- ifelse(length(dims) > 2, dims[3], 1)

# Calculate SRSF
for(i in c("train", "val", "test")){
  assign(paste("q", i, sep = "_"), SRSF(eval(parse(text = paste("f", i, sep = "_")))))
}

# Define model (example architecture)
warp_kernel <- ceiling((L + K_true - 1) / K_true)
layers <- ceiling(2 * L / (warp_kernel - 1))
filters <- rep((n + 3) * K_true, layers)
kernel <- rep(warp_kernel, layers)

# Fit model
set_random_seed(141420092023 %% .Machine$integer.max, disable_gpu = F)
# The training of the model takes time, for convenience, it is possible to load a saved model
model <- DeepJAM(L = L, n = n, filters = filters, kernel = kernel, warp_kernel = warp_kernel, activation = "tanh") # initialize model
model %>% compile(loss = LossWarping, optimizer = optimizer_adam(learning_rate = 1 / model$count_params())) # compile model
history <- model %>% fit(K = K, x = q_train, validation_data = q_val,
                         epochs = 2500, early_stopping = T, patience = 25, early_stopping_criterion = 0.0001) # fit model
SaveDeepJAM(model, file = paste0(path, "Models/", data_type, "_", K, if(K == 1) "_period" else "_periods")) # save model
saveRDS(history, file = paste0(path, "Models/", data_type, "_", K, if(K == 1) "_period" else "_periods", "/history.RDS")) # save history
model <- LoadDeepJAM(file = paste0(path, "Models/", data_type, "_", K, if(K == 1) "_period" else "_periods")) # load model

# Plot results ------------------------------------------------------------
svg(paste0(path, "Results/", data_type, ".svg"), width = 11.75, height = 8.25 * 2)
par(mfrow = c(n, 2), mar = c(2, 0, 0, 0))
for(j in seq_len(n)){
  plot(model, f_test, K, type = "observed", j = j, mu = NULL, s = 25, seed = 114520092023 %% .Machine$integer.max)
  plot(model, f_test, K, type = "aligned", j = j, mu = NULL, s = 25, seed = 114520092023 %% .Machine$integer.max)
}
dev.off()