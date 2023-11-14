# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# Author:       Vi Thanh Pham
# GitHub:       https://github.com/ViThanh
# Description:  Simulate data for DeepJAM
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# Path --------------------------------------------------------------------
path <- sub("Code/", "", paste0(paste(head(strsplit(rstudioapi::getSourceEditorContext()$path, split = "/")[[1]], -1), collapse = "/"), "/"))

# Source functions --------------------------------------------------------
source(paste0(path, "Code/Functions.R"))
tf$config$experimental$enable_op_determinism()
tf$config$experimental$enable_tensor_float_32_execution(F)
nk <- import("neurokit2")

# Set seed ----------------------------------------------------------------
set_random_seed(135314112022 %% .Machine$integer.max, disable_gpu = F)

# -------------------------------------------------------------------------
# Univariate data ---------------------------------------------------------
# -------------------------------------------------------------------------
data_type <- "univariate"
# Configuration -----------------------------------------------------------
rate <- 0.5 # rate for Karcher mean
criterion <- 10 ^ (-16) # stopping criterion for Karcher mean
max_iter <- 10000L # maximum number of iterations for Karcher mean
K <- 5L # number of periods
P <- 65L # number of sampling points per period
L <- as.integer(K * P - K + 1) # total length of the time-series
N <- 14000L # number of observations
gamma_local <- fdasrvf::rgam(P, 0.1, N)[sample(N),] # random local warping functions
gamma_local_ext <- ExtendWarp(gamma_local, K) # extend local warp to K periods
gamma_global <- fdasrvf::rgam(L, 0.1, N)[sample(N),] # random warping functions
gamma_total <- Warp(gamma_local_ext, gamma_global) # get total warp
gamma_total_inv <- Inverse(gamma_total) # calculate the inverse of the total warps
gamma_total_inv_overall_KM <- ExtractOverallKM(gamma_total_inv, K, drop_axis = F, as.tensor = F, criterion = criterion, max_iter = max_iter, rate = rate)
gamma_total_inv_overall_KM_ext <- ExtendWarp(gamma_total_inv_overall_KM, K)
gamma_total <- Warp(gamma_total_inv_overall_KM_ext, gamma_total)
gamma_total_inv <- Inverse(gamma_total)
gamma_total_inv_individual_KM <- ExtractIndividualKM(gamma_total_inv, K, drop_axis = F, as.tensor = F, criterion = criterion, max_iter = max_iter, rate = rate)
gamma_total_inv_individual_KM_ext <- ExtendWarp(gamma_total_inv_individual_KM, K)
gamma_global <- Warp(gamma_total_inv_individual_KM_ext, gamma_total)
gamma_local <- Inverse(gamma_total_inv_individual_KM)
gamma_local_ext <- ExtendWarp(gamma_local, K)

# Sample functions --------------------------------------------------------
t <- seq(0, 1, length.out = L)
mu <- sin(2 * pi * t * K)

# Warp
f_local <- Warp(matrix(mu, nrow = 1, ncol = L), gamma_local_ext)
f <- as.array(Warp(f_local, gamma_global))

# Split train and validation data -----------------------------------------
id_train <- seq_len(N * 5 / 7)
id_val <- seq_len((N - length(id_train)) / 2) + length(id_train)
id_test <- setdiff(seq_len(N), c(id_train, id_val))
for(i in c("train", "val", "test")){
  for(j in paste("gamma", c("local", "global"), sep = "_")){
    assign(paste(j, i, sep = "_"), as.array(eval(parse(text = j)))[eval(parse(text = paste("id", i, sep = "_"))),])
  }
  assign(paste("f", i, sep = "_"), as.array(f)[eval(parse(text = paste("id", i, sep = "_"))),])
}

# Plot data ---------------------------------------------------------------
s <- 200
pdf(file = paste0(path, "Data/", data_type, ".pdf"), width = 11.75, height = 8.25)
par(mfrow = c(2, 1), mar = c(0, 0, 2, 0))
for(i in c("_local", "")){
  matplot(t(as.matrix(eval(parse(text = paste0("f", i)))[seq_len(s),])), 
          type = "l", lty = 1, col = scales::alpha(1:6, 0.25), ylab = "", yaxt = "n", xlab = "", xaxt = "n", bty = "n", 
          main = if(i == "") "Observed data" else "Local warp")
  lines(mu, lw = 2)
}
dev.off()

# Save data ---------------------------------------------------------------
mu <- as.matrix(mu, ncol = 1)
save(K, mu,
     gamma_local_train, gamma_global_train, f_train,
     gamma_local_val, gamma_global_val, f_val,
     gamma_local_test, gamma_global_test, f_test,
     file = paste0(path, "Data/", data_type, ".RData"))

# -------------------------------------------------------------------------
# Multivariate data -------------------------------------------------------
# -------------------------------------------------------------------------
data_type <- "multivariate"
# Configuration -----------------------------------------------------------
rate <- 0.5 # rate for Karcher mean
criterion <- 10 ^ (-16) # stopping criterion for Karcher mean
max_iter <- 10000L # maximum number of iterations for Karcher mean
K <- 3L # number of periods
P <- 65L # number of sampling points per period
L <- as.integer(K * P - K + 1) # total length of the time-series
N <- 14000L # number of observations
gamma_local <- fdasrvf::rgam(P, 0.1, N)[sample(N),] # random local warping functions
gamma_local_ext <- ExtendWarp(gamma_local, K) # extend local warp to K periods
gamma_global <- fdasrvf::rgam(L, 0.1, N)[sample(N),] # random warping functions
gamma_total <- Warp(gamma_local_ext, gamma_global) # get total warp
gamma_total_inv <- Inverse(gamma_total) # calculate the inverse of the total warps
gamma_total_inv_overall_KM <- ExtractOverallKM(gamma_total_inv, K, drop_axis = F, as.tensor = F, criterion = criterion, max_iter = max_iter, rate = rate)
gamma_total_inv_overall_KM_ext <- ExtendWarp(gamma_total_inv_overall_KM, K)
gamma_total <- Warp(gamma_total_inv_overall_KM_ext, gamma_total)
gamma_total_inv <- Inverse(gamma_total)
gamma_total_inv_individual_KM <- ExtractIndividualKM(gamma_total_inv, K, drop_axis = F, as.tensor = F, criterion = criterion, max_iter = max_iter, rate = rate)
gamma_total_inv_individual_KM_ext <- ExtendWarp(gamma_total_inv_individual_KM, K)
gamma_global <- Warp(gamma_total_inv_individual_KM_ext, gamma_total)
gamma_local <- Inverse(gamma_total_inv_individual_KM)
gamma_local_ext <- ExtendWarp(gamma_local, K)

# Sample functions --------------------------------------------------------
t <- seq(0, 1, length.out = P) # sampling times should contain 0.25, 0.5 and 0.75
# Sample random numbers
for(i in seq_len(K)){
  assign(paste0("z1_", i), abs(rnorm(N, 1, 0.25)))
}
for(i in seq_len(K)){
  for(j in 1:2){
    assign(paste0("z2_", i, "_", j), abs(rnorm(N, 1, 0.25)))
  }
}
for(i in seq_len(K)){
  for(j in 1:3){
    assign(paste0("z3_", i, "_", j), abs(rnorm(N, 1, 0.25)))
  }
}

# Function 1
t_mat <- matrix(t, nrow = N, ncol = P, byrow = T)
mu_1 <- ExtendFunction(sin(2 * pi * t), K)
f_true_1 <- CombinePeriods(lapply(seq_len(K), \(x) eval(parse(text = paste0("z1_", x))) * sin(t_mat * pi * 2)))

# Function 2
t_mat <- matrix(t * 6 - 3, nrow = N, ncol = P, byrow = T)
mu_2_1 <- exp(-(t_mat[1,] + 1.5) ^ 2 / 2)
mu_2_2 <- exp(-(t_mat[1,] - 1.5) ^ 2 / 2)
mu_2 <- mu_2_1 + mu_2_2
mu_2_range <- range(mu_2)
mu_2 <- ((mu_2 - mu_2_range[1]) / diff(mu_2_range) - 0.5) * 2
mu_2 <- ExtendFunction(mu_2, K)
offset_2_1 <- seq(mu_2_1[1], mu_2_1[P], length.out = P)
offset_2_2 <- seq(mu_2_2[1], mu_2_2[P], length.out = P)
f_true_2 <- lapply(seq_len(K), \(x){
  do.call("+", lapply(1:2, \(y){
    mu <- eval(parse(text = paste0("mu_2_", y)))
    offset <- eval(parse(text = paste0("offset_2_", y)))
    z <- eval(parse(text = paste0("z2_", x, "_", y)))
    z * matrix(mu - offset, nrow = N, ncol = P, byrow = T) + matrix(offset, nrow = N, ncol = P, byrow = T)
  })
  )
})
f_true_2 <- CombinePeriods(lapply(f_true_2, \(x) ((x - mu_2_range[1]) / diff(mu_2_range) - 0.5) * 2))

# Function 3
t_mat_1 <- t_mat_2 <- t_mat_3 <- matrix(t, nrow = N, ncol = P, byrow = T)
t_mat_1[t_mat_1 > 0.5] <- 0.5
t_mat_3[t_mat_3 < 0.5] <- 0.5
mu_3_1 <- dnorm(t_mat_1[1,], 0.25, 0.1)
mu_3_1 <- mu_3_1 - min(mu_3_1)
mu_3_2 <- dnorm(t_mat_2[1,], 0.5, 0.15)
mu_3_2_range <- range(mu_3_2)
mu_3_2 <- mu_3_2 - min(mu_3_2)
mu_3_3 <- dnorm(t_mat_3[1,], 0.75, 0.1)
mu_3_3 <- mu_3_3 - min(mu_3_2)
mu_3 <- - mu_3_1 + mu_3_2 + mu_3_3
mu_3_range <- range(mu_3)
mu_3 <- ((mu_3 - mu_3_range[1]) / diff(mu_3_range) - 0.5) * 2
mu_3 <- ExtendFunction(mu_3, K)
f_true_3 <- lapply(seq_len(K), \(x){
  lapply(1:3, \(y){
    mu <- eval(parse(text = paste0("mu_3_", y)))
    z <- eval(parse(text = paste0("z3_", x, "_", y)))
    z * matrix(mu, nrow = N, ncol = P, byrow = T)
  })
})
f_true_3 <- CombinePeriods(lapply(f_true_3, \(x) (((-x[[1]] + x[[2]] + x[[3]]) - mu_3_range[1]) / diff(mu_3_range) - 0.5) * 2))

# Combine into one dataset ------------------------------------------------
mu <- cbind(as.numeric(mu_1), as.numeric(mu_2), as.numeric(mu_3))
f_true <- abind::abind(list(as.array(f_true_1), as.array(f_true_2), as.array(f_true_3)), along = 3)
f_local <- as.array(Warp(f_true, gamma_local_ext))
f <- as.array(Warp(f_local, gamma_global))

# Split train and validation data -----------------------------------------
id_train <- seq_len(N * 5 / 7)
id_val <- seq_len((N - length(id_train)) / 2) + length(id_train)
id_test <- setdiff(seq_len(N), c(id_train, id_val))
for(i in c("train", "val", "test")){
  for(j in paste("gamma", c("local", "global"), sep = "_")){
    assign(paste(j, i, sep = "_"), as.array(eval(parse(text = j)))[eval(parse(text = paste("id", i, sep = "_"))),])
  }
  assign(paste("f", i, sep = "_"), f[eval(parse(text = paste("id", i, sep = "_"))),,])
}

# Plot data ---------------------------------------------------------------
s <- 200
pdf(file = paste0(path, "Data/", data_type, ".pdf"), width = 11.75, height = 8.25)
par(mfrow = c(3, 1), mar = c(0, 0, 2, 0))
for(i in c("_true", "_local", "")){
  dt <- eval(parse(text = paste0("f", i)))[seq_len(s),,]
  for(j in 1:3){
    matplot(t(dt[,,j]), type = "l", lty = 1, col = scales::alpha(1:6, 0.25), ylab = "", yaxt = "n", xlab = "", xaxt = "n", bty = "n", 
            main = if(j == 1){if(i == "_true") "True data" else if(i == "_local") "Local warp" else "Observed data"} else "")
    lines(mu[,j], lw = 2)
  }
}
dev.off()

# Save data ---------------------------------------------------------------
save(K, mu,
     gamma_global_train, gamma_local_train, f_train,
     gamma_global_val, gamma_local_val, f_val,
     gamma_global_test, gamma_local_test, f_test,
     file = paste0(path, "Data/", data_type, ".RData"))

# -------------------------------------------------------------------------
# ECG data ----------------------------------------------------------------
# -------------------------------------------------------------------------
data_type <- "ECG"
# Sample functions --------------------------------------------------------
K <- 4L # number of periods
P <- 201L # number of sampling points per period
L <- as.integer(K * P - K + 1) # total length of the time-series
N <- 700L # number of observations
start <- 3L # starting heartbeat

# Simulate function -------------------------------------------------------
SimulateECG <- \(N = 10000L, start = 3L, periods = 4L,
                 sampling_rate_period = 250L, length = as.integer(periods * (sampling_rate_period - 1) + 1),
                 noise = 0.075, std = 7.5, R_position = 0.385){
  # Simulate 12-lead ECG
  # N: number of observations
  # start: starting R-peak
  # periods: number of heartbeats
  # sampling_rate_period: sampling points per heartbeat
  # length: length of the final heartbeat
  # noise: noise for the ecg_simulate function from neurokit package
  # std: standard deviation for the ecg_simulate function from neurokit package
  # R_position: relative position of the R-peak in a heartbeat
  
  end <- start + periods - 1L
  ECG <- array(dim = c(N, length, 12))
  for(i in seq_len(N)){
    ecg <- nk$ecg_simulate(duration = as.integer(end + 2 * std), noise = noise, heart_rate_std = std, method = "multileads")
    for(j in seq_len(ncol(ecg))){
      ecg[,j] <- nk$ecg_clean(ecg[,j])
    }
    RMS_lead <- sqrt(rowMeans(ecg ^ 2))
    peaks <- nk$ecg_peaks(RMS_lead)
    Rs <- which(peaks[[1]]$ECG_R_Peaks == 1)
    s <- floor(Rs[start - 1] * R_position + Rs[start] * (1 - R_position))
    e <- ceiling(Rs[end] * R_position + Rs[end + 1] * (1 - R_position))
    ecg <- as.matrix(ecg[s:e,])
    ECG[i,,] <- as.array(Warp(array(ecg, dim = c(1, dim(ecg))), array(seq(0, 1, length.out = length), dim = c(1, length))))
  }
  return(ECG)
}
f <- SimulateECG(N = N, start = start, periods = K, sampling_rate_period = P, length = L)

# Split train and validation data -----------------------------------------
id_train <- seq_len(N * 5 / 7)
id_val <- seq_len((N - length(id_train)) / 2) + length(id_train)
id_test <- setdiff(seq_len(N), c(id_train, id_val))
for(i in c("train", "val", "test")){
  assign(paste("f", i, sep = "_"), f[eval(parse(text = paste("id", i, sep = "_"))),,])
}

# Plot data ---------------------------------------------------------------
s <- 200
pdf(file = paste0(path, "Data/", data_type, ".pdf"), height = 11.75, width = 8.25)
par(mfrow = c(dim(f)[3], 1), mar = rep(0, 4))
for(i in c("")){
  dt <- eval(parse(text = paste0("f", i)))[seq_len(s),,]
  for(j in seq_len(dim(f)[3])){
    matplot(t(dt[,,j]), type = "l", lty = 1, col = scales::alpha(1:6, 0.25), ylab = "", yaxt = "n", xlab = "", xaxt = "n", bty = "n")
  }
}
dev.off()

# Save data ---------------------------------------------------------------
save(K, f_train, f_val, f_test,
     file = paste0(path, "Data/", data_type, ".RData"))