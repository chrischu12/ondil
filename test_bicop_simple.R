# Direct replacement for bicoppd1d2 using individual VineCopula functions
# This replicates the exact same computation but with separate function calls

library(VineCopula)

# Exact values computed from the first 10 rows and fitted coefficients
u1 <- c(0.930144, 0.853130, 0.431601, 0.701803, 0.736621, 0.585327, 0.595364, 0.168512, 0.946548, 0.893125)
u2 <- c(0.533249, 0.699036, 0.813250, 0.453937, 0.653818, 0.571856, 0.855921, 0.153280, 0.862996, 0.977878)
theta <- rep(1.47, 10)

cat("Testing individual VineCopula functions with exact computed parameters:\n")
cat("u1[1:5]:    ", sprintf("%.6f ", u1[1:5]), "\n")
cat("u2[1:5]:    ", sprintf("%.6f ", u2[1:5]), "\n")
cat("theta[1:5]: ", sprintf("%.6f ", theta[1:5]), "\n\n")

# Family 33 = Clayton copula with 270° rotation (equivalent to gamCopula family 302)
family <- 33

# Initialize result vectors
n <- length(u1)
pdf_result <- numeric(n)
d1_result <- numeric(n)
d2_result <- numeric(n)

# Calculate PDF, first derivative, and second derivative for each observation
for(i in 1:n) {
    # PDF calculation
    pdf_result[i] <- BiCopPDF(u1[i], u2[i], family, theta[i])
    
    # First derivative w.r.t. parameter
    d1_result[i] <- BiCopDeriv(u1[i], u2[i], family, theta[i], deriv="par")
    
    # Second derivative w.r.t. parameter  
    d2_result[i] <- BiCopDeriv2(u1[i], u2[i], family, theta[i], deriv="par")
}

# Create result object similar to bicoppd1d2 output
result <- list(
    pdf = pdf_result,
    d1 = d1_result,
    d2 = d2_result
)

# Print results in the same format as bicoppd1d2
cat("PDF values:\n")
print(result$pdf)
cat("\nFirst derivatives:\n") 
print(result$d1)
cat("\nSecond derivatives:\n")
print(result$d2)

# Also print the result object
cat("\nResult object:\n")
print(result)