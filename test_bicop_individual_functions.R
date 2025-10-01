# Test individual bivariate copula functions instead of bicoppd1d2
# This calls biCopPdf, BiCopDeriv, and BiCopDeriv2 separately with Clayton rotation 3 (family 33)

library(VineCopula)

# Exact values computed from the first 10 rows and fitted coefficients
u1 <- c(0.930144, 0.853130, 0.431601, 0.701803, 0.736621, 0.585327, 0.595364, 0.168512, 0.946548, 0.893125)
u2 <- c(0.533249, 0.699036, 0.813250, 0.453937, 0.653818, 0.571856, 0.855921, 0.153280, 0.862996, 0.977878)
theta <- rep(1.47, 10)

cat("Testing individual bivariate copula functions with exact computed parameters:\n")
cat("u1[1:5]:    ", sprintf("%.6f ", u1[1:5]), "\n")
cat("u2[1:5]:    ", sprintf("%.6f ", u2[1:5]), "\n")
cat("theta[1:5]: ", sprintf("%.6f ", theta[1:5]), "\n")
cat("Family: 33 (Clayton 270° rotation)\n\n")

# Family 33 corresponds to Clayton copula with 270° rotation
family <- 33

# Call individual functions
cat("=== PDF (biCopPdf) ===\n")
pdf_values <- numeric(length(u1))
for(i in 1:length(u1)) {
    pdf_values[i] <- BiCopPDF(u1[i], u2[i], family, theta[i])
}
cat("PDF values: ", sprintf("%.8f ", pdf_values), "\n\n")

cat("=== Log PDF ===\n")
logpdf_values <- log(pdf_values)
cat("Log PDF values: ", sprintf("%.8f ", logpdf_values), "\n\n")

cat("=== First Derivative (BiCopDeriv) ===\n")
deriv1_values <- numeric(length(u1))
for(i in 1:length(u1)) {
    deriv1_values[i] <- BiCopDeriv(u1[i], u2[i], family, theta[i], deriv="par")
}
cat("1st derivative: ", sprintf("%.8f ", deriv1_values), "\n\n")

cat("=== Second Derivative (BiCopDeriv2) ===\n")
deriv2_values <- numeric(length(u1))
for(i in 1:length(u1)) {
    deriv2_values[i] <- BiCopDeriv2(u1[i], u2[i], family, theta[i], deriv="par")
}
cat("2nd derivative: ", sprintf("%.8f ", deriv2_values), "\n\n")

# Create results summary
cat("=== SUMMARY RESULTS ===\n")
cat("Index | u1       | u2       | theta   | PDF      | LogPDF   | 1st Deriv | 2nd Deriv\n")
cat("------|----------|----------|---------|----------|----------|-----------|----------\n")
for(i in 1:length(u1)) {
    cat(sprintf("%5d | %8.6f | %8.6f | %7.2f | %8.6f | %8.6f | %9.6f | %9.6f\n", 
                i, u1[i], u2[i], theta[i], pdf_values[i], logpdf_values[i], 
                deriv1_values[i], deriv2_values[i]))
}

# Also test with vectorized calls if available
cat("\n=== VECTORIZED CALLS (if supported) ===\n")
tryCatch({
    pdf_vec <- BiCopPDF(u1, u2, family, theta)
    cat("Vectorized PDF: ", sprintf("%.8f ", pdf_vec), "\n")
    
    deriv1_vec <- BiCopDeriv(u1, u2, family, theta, deriv="par")
    cat("Vectorized 1st deriv: ", sprintf("%.8f ", deriv1_vec), "\n")
    
    deriv2_vec <- BiCopDeriv2(u1, u2, family, theta, deriv="par")
    cat("Vectorized 2nd deriv: ", sprintf("%.8f ", deriv2_vec), "\n")
}, error = function(e) {
    cat("Vectorized calls not supported or error occurred: ", e$message, "\n")
})

# Compare with the bicoppd1d2 function if gamCopula is available
cat("\n=== COMPARISON WITH bicoppd1d2 (if gamCopula available) ===\n")
tryCatch({
    library(gamCopula)
    
    # Create data matrix for bicoppd1d2 (family 302 = Clayton rotation 3)
    data_matrix <- cbind(u1, u2, theta, rep(0, 10))
    
    # Run bicoppd1d2 with family 302 
    bicoppd1d2_result <- gamCopula:::bicoppd1d2(data_matrix, 302, p=TRUE, d1=TRUE, d2=TRUE)
    
    cat("bicoppd1d2 PDF: ", sprintf("%.8f ", bicoppd1d2_result$pdf), "\n")
    cat("bicoppd1d2 1st deriv: ", sprintf("%.8f ", bicoppd1d2_result$d1), "\n")
    cat("bicoppd1d2 2nd deriv: ", sprintf("%.8f ", bicoppd1d2_result$d2), "\n")
    
    # Compare differences
    cat("\nDifferences (VineCopula - gamCopula):\n")
    cat("PDF diff: ", sprintf("%.10f ", pdf_values - bicoppd1d2_result$pdf), "\n")
    cat("1st deriv diff: ", sprintf("%.10f ", deriv1_values - bicoppd1d2_result$d1), "\n")
    cat("2nd deriv diff: ", sprintf("%.10f ", deriv2_values - bicoppd1d2_result$d2), "\n")
    
}, error = function(e) {
    cat("gamCopula not available or error occurred: ", e$message, "\n")
})

cat("\n=== FUNCTION CALL DETAILS ===\n")
cat("Functions used:\n")
cat("- BiCopPDF(u1, u2, family=33, par)\n")
cat("- BiCopDeriv(u1, u2, family=33, par, deriv='par')\n")
cat("- BiCopDeriv2(u1, u2, family=33, par, deriv='par')\n")
cat("\nFamily 33 = Clayton copula with 270° rotation\n")
cat("Parameter = 1.47 (constant for all observations)\n")