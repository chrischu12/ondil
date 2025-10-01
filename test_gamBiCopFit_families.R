# Deep dive into bicoppd1d2 behavior with different Clayton families
# Understanding why family 301 vs 302 gives different coefficients

library(gamCopula)

# Simulate the same data structure as your merged_data
set.seed(123)
n_obs <- 100
x1 <- rnorm(n_obs)
x2 <- rnorm(n_obs)
x3 <- rnorm(n_obs)

# Create copula data (you can replace this with your actual merged_data)
u1 <- runif(n_obs, 0.1, 0.9)
u2 <- runif(n_obs, 0.1, 0.9)

merged_data <- data.frame(u1 = u1, u2 = u2, x1 = x1, x2 = x2, x3 = x3)

cat("=== Testing gamBiCopFit with different Clayton families ===\n")
cat("Data dimensions:", dim(merged_data), "\n")
cat("First few rows:\n")
print(head(merged_data, 3))
cat("\n")

# Test different Clayton families
clayton_families <- c(300, 301, 302, 303)
family_names <- c("Clayton 0°", "Clayton 90°", "Clayton 180°", "Clayton 270°")

fits <- list()
coefficients_comparison <- data.frame()

for(i in 1:length(clayton_families)) {
    fam <- clayton_families[i]
    cat(sprintf("=== Fitting with family %d (%s) ===\n", fam, family_names[i]))
    
    tryCatch({
        # Fit the model
        fit <- gamBiCopFit(merged_data,
                          formula = ~ (x1 + x2 + x3),
                          family = fam,
                          verbose = FALSE,  # Set to TRUE if you want detailed output
                          method = "NR",
                          n.iters = 50,  # Reduced for faster testing
                          tau = TRUE)
        
        fits[[as.character(fam)]] <- fit
        
        # Extract coefficients
        coef_vec <- fit$res@model$coefficients
        cat("Coefficients:", sprintf("%.6f ", coef_vec), "\n")
        
        # Store for comparison
        coef_df <- data.frame(
            family = fam,
            family_name = family_names[i],
            intercept = coef_vec[1],
            x1_coef = coef_vec[2],
            x2_coef = coef_vec[3],
            x3_coef = coef_vec[4]
        )
        coefficients_comparison <- rbind(coefficients_comparison, coef_df)
        
        # Get log-likelihood
        loglik <- fit$res@logLik
        cat("Log-likelihood:", loglik, "\n")
        
        # Test bicoppd1d2 with a few sample points to understand internal behavior
        sample_indices <- 1:3
        test_data <- merged_data[sample_indices, ]
        
        # Create the prediction for these points
        linear_pred <- as.matrix(cbind(1, test_data[,c("x1", "x2", "x3")])) %*% coef_vec
        
        # Create bicoppd1d2 input matrix
        bicop_input <- cbind(test_data$u1, test_data$u2, linear_pred, rep(0, nrow(test_data)))
        
        # Call bicoppd1d2
        bicop_result <- gamCopula:::bicoppd1d2(bicop_input, fam, p=TRUE, d1=TRUE, d2=TRUE)
        
        cat("bicoppd1d2 results (first 3 obs):\n")
        cat("  PDF:      ", sprintf("%.8f ", bicop_result[1,]), "\n")
        cat("  1st deriv:", sprintf("%.8f ", bicop_result[2,]), "\n") 
        cat("  2nd deriv:", sprintf("%.8f ", bicop_result[3,]), "\n")
        
        cat("\n")
        
    }, error = function(e) {
        cat("ERROR fitting family", fam, ":", e$message, "\n\n")
    })
}

# Compare coefficients across families
cat("=== COEFFICIENT COMPARISON ===\n")
print(coefficients_comparison)

if(nrow(coefficients_comparison) > 1) {
    cat("\nDifferences from family 300 (standard Clayton):\n")
    base_coefs <- coefficients_comparison[coefficients_comparison$family == 300, ]
    
    for(i in 2:nrow(coefficients_comparison)) {
        curr_coefs <- coefficients_comparison[i, ]
        cat(sprintf("Family %d vs 300:\n", curr_coefs$family))
        cat(sprintf("  Intercept: %+.6f\n", curr_coefs$intercept - base_coefs$intercept))
        cat(sprintf("  x1 coeff:  %+.6f\n", curr_coefs$x1_coef - base_coefs$x1_coef))
        cat(sprintf("  x2 coeff:  %+.6f\n", curr_coefs$x2_coef - base_coefs$x2_coef))
        cat(sprintf("  x3 coeff:  %+.6f\n", curr_coefs$x3_coef - base_coefs$x3_coef))
        cat("\n")
    }
}

# Analyze why coefficients differ by examining the rotation transformations
cat("=== ROTATION TRANSFORMATION ANALYSIS ===\n")
cat("Understanding why different families give different coefficients:\n\n")

for(i in 1:length(clayton_families)) {
    fam <- clayton_families[i]
    fams <- gamCopula:::getFams(fam)
    
    cat(sprintf("Family %d (%s):\n", fam, family_names[i]))
    cat(sprintf("  Maps to VineCopula families: [%d, %d]\n", fams[1], fams[2]))
    
    # Show the implied data transformation
    if(fam == 300) {
        cat("  Data transformation: u1=u1, u2=u2, theta=theta (no rotation)\n")
    } else if(fam == 301) {
        cat("  Data transformation: u1=1-u1, u2=u2, theta=-theta (90° rotation)\n")
    } else if(fam == 302) {
        cat("  Data transformation: u1=u1, u2=1-u2, theta=-theta (270° rotation)\n") 
    } else if(fam == 303) {
        cat("  Data transformation: u1=1-u1, u2=1-u2, theta=theta (180° rotation)\n")
    }
    cat("\n")
}

cat("=== KEY INSIGHT ===\n")
cat("Different rotation families transform the copula data differently.\n")
cat("Even if bicoppd1d2 gives similar derivative values at individual points,\n") 
cat("the overall likelihood surface is different due to the data transformations.\n")
cat("This creates different optimal coefficient values in the regression.\n")
cat("\nThe derivatives might look similar at specific points, but the entire\n")
cat("optimization landscape is rotated, leading to different solutions.\n")