# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# Author:       Vi Thanh Pham
# GitHub:       https://github.com/ViThanh
# Description:  Plot results of DeepJAM
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# Plot results ------------------------------------------------------------
plot.DeepJAM <- \(object, data, K, type, j, mu = NULL, s, seed = NULL,...){
  # Plot result of DeepJAM
  # object: DeepJAM neural network
  # data: input data
  # K: number of periods
  # type: choose between observed and aligned
  # j: j-th function of the multivariate data
  # s: subset of data
  # seed: seed for random sample
  
  # To array
  if("numeric" %in% class(data)) data <- array(data, dim = c(1, length(data)))
  
  # Extract dimensions
  dims <- dim(data)
  
  # Extract number of functions per observation
  n <- if(length(dims) == 2) 1 else dims[3]
  
  # Extract means
  mean_observed <- as.array(tf$reduce_mean(data, axis = 0L))
  
  if(type == "aligned"){
    # Extract SRSF
    srsf <- SRSF(data)
    
    # Extract warps
    total_warps <- ExtractTotalWarp(object, srsf, K, ...)
    
    # Extract means
    warped <- Warp(tf$cast(data, dtype = total_warps$dtype), total_warps)
    mean_aligned <- as.array(tf$reduce_mean(warped, axis = 0L))
    mean_orbit <- as.array(tf$squeeze(ExtendFunction(MeanOrbitPeriod(warped, K), K)), axis = 0L)
  }
  
  # Subset data
  if(!is.null(seed)) set_random_seed(seed, disable_gpu = F)
  N <- dim(data)[1]
  idx <- sample(seq_len(N))[seq_len(s)]
  
  # Plot
  if(length(dims) == 2){
    dt <- if(type == "observed") t(data) else t(as.array(Warp(tf$cast(data, dtype = total_warps$dtype), total_warps)))
    matplot(seq(0, 1, length.out = L) * K, dt[,idx], col = 0, type = "l", xlab = "", ylab = "", xaxt = "n", yaxt = "n", bty = "n", main = if(type == "observed") "Observed data" else paste("Aligned data -", K, if(K == 1) "period" else "periods"))
    abline(v = c(0, seq_len(K)), lty = "dashed", col = "grey")
    matlines(seq(0, 1, length.out = L) * K, dt[,idx], col = scales::alpha(1:6, 0.2), lty = 1)
    if(!is.null(mu)) matlines(seq(0, 1, length.out = L) * K, mu, lwd = 2, col = 2)
    if(type == "observed"){
      lines(seq(0, 1, length.out = L) * K, mean_observed, lwd = 2, col = 1)
    }else if(type == "aligned"){
      lines(seq(0, 1, length.out = L) * K, mean_aligned, lwd = 2, col = 1, lty = "dashed")
      lines(seq(0, 1, length.out = L) * K, mean_orbit, lwd = 2, col = 1, lty = "dotted")
    }
    axis(side = 1, at = c(0, seq_len(K)), labels = c(0, expression(tau), sapply(seq_len(K - 1) + 1, \(x) as.expression(bquote(.(x) * tau)))))
  }else{
    dt <- if(type == "observed") t(data[,,j]) else t(as.array(Warp(tf$cast(data[,,j], dtype = total_warps$dtype), total_warps)))
    matplot(seq(0, 1, length.out = L) * K, dt[,idx], col = 0, type = "l", xlab = "", ylab = "", xaxt = "n", yaxt = "n", bty = "n")
    abline(v = c(0, seq_len(K)), lty = "dashed", col = "grey")
    matlines(seq(0, 1, length.out = L) * K, dt[,idx], col = scales::alpha(1:6, 0.2), lty = 1)
    if(!is.null(mu)) matlines(seq(0, 1, length.out = L) * K, mu[,j], lwd = 2, col = 2)
    if(type == "observed"){
      lines(seq(0, 1, length.out = L) * K, mean_observed[,j], lwd = 2, col = 1)
    }else if(type == "aligned"){
      lines(seq(0, 1, length.out = L) * K, mean_aligned[,j], lwd = 2, col = 1, lty = "dashed")
      lines(seq(0, 1, length.out = L) * K, mean_orbit[,j], lwd = 2, col = 1, lty = "dotted")
    }
    axis(side = 1, at = c(0, seq_len(K)), labels = c(0, expression(tau), sapply(seq_len(K - 1) + 1, \(x) as.expression(bquote(.(x) * tau)))))
    if(j == 1) title(if(type == "observed") "Observed data" else "Aligned data", line = -1)
  }
}