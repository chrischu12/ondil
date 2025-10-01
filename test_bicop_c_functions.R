# Direct C function calls using .C() method instead of R wrapper functions
# This calls the underlying C functions directly for PDF and derivatives

library(VineCopula)


# Exact values computed from the first 10 rows and fitted coefficients
u1 <- c(0.930144, 0.853130, 0.431601, 0.701803, 0.736621, 0.585327, 0.595364, 0.168512, 0.946548, 0.893125)
u2 <- c(0.533249, 0.699036, 0.813250, 0.453937, 0.653818, 0.571856, 0.855921, 0.153280, 0.862996, 0.977878)
theta <- rep(1.47, 10)

cat("Testing C functions directly with .C() method:\n")
cat("u1[1:5]:    ", sprintf("%.6f ", u1[1:5]), "\n")
cat("u2[1:5]:    ", sprintf("%.6f ", u2[1:5]), "\n")
cat("theta[1:5]: ", sprintf("%.6f ", theta[1:5]), "\n\n")

# Test both the gamCopula approach and direct VineCopula approach
cat("=== Understanding family mapping ===\n")

# Get the family mapping for gamCopula 302
tryCatch({
    library(gamCopula)
    fams <- gamCopula:::getFams(302)  # gamCopula family 302 = Clayton rotation 3
    cat("getFams(302) returns VineCopula families:", fams, "\n")
    cat("Family for positive params:", fams[1], "\n")
    cat("Family for negative params:", fams[2], "\n")
    
    # Since our theta = 1.47 > 0, we use fams[1]
    family <- fams[1]
    cat("Using VineCopula family:", family, "(for positive theta)\n\n")
}, error = function(e) {
    cat("gamCopula not available, using family 3\n")
    family <- 3
})
n <- length(u1)

# Initialize output vectors
pdf_out <- numeric(n)
d1_out <- numeric(n)
d2_out <- numeric(n)

cat("=== Direct C function calls ===\n")

# Call C function for PDF calculation
# C function: void PDF(double* u, double* v, int* n, double* param, int* copula, double* out)

# Call C function for first derivative  
# C function: void diffPDF_mod(double* u, double* v, int* n, double* param, int* copula, double* out)
d1_result <- .C("difflPDF_mod",
                u = as.double(u1),
                v = as.double(u2),
                n = as.integer(n),
                param = as.double(theta),
                copula = as.integer(rep(family, n)),
                out = as.double(d1_out),
                PACKAGE = "VineCopula")

cat("\nFirst derivative (C function):\n")
print(d1_result$out)
