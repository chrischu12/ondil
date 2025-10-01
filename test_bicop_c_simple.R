# Compact version using direct C function calls via .C() method
# Equivalent to bicoppd1d2 but using direct C calls

library(VineCopula)

# Exact values computed from the first 10 rows and fitted coefficients
u1 <- c(0.930144, 0.853130, 0.431601, 0.701803, 0.736621, 0.585327, 0.595364, 0.168512, 0.946548, 0.893125)
u2 <- c(0.533249, 0.699036, 0.813250, 0.453937, 0.653818, 0.571856, 0.855921, 0.153280, 0.862996, 0.977878)
theta <- rep(1.47, 10)

cat("Testing direct C functions with exact computed parameters:\n")
cat("u1[1:5]:    ", sprintf("%.6f ", u1[1:5]), "\n")
cat("u2[1:5]:    ", sprintf("%.6f ", u2[1:5]), "\n")
cat("theta[1:5]: ", sprintf("%.6f ", theta[1:5]), "\n\n")

# Parameters
family <- 33  # Clayton 270° rotation
n <- length(u1)

# Direct C function calls using .C()
pdf_result <- .C("PDF",
                 u = as.double(u1),
                 v = as.double(u2), 
                 n = as.integer(n),
                 param = as.double(theta),
                 copula = as.integer(rep(family, n)),
                 out = numeric(n),
                 PACKAGE = "VineCopula")$out

d1_result <- .C("difflPDF_mod",  # Using log derivative version
                u = as.double(u1),
                v = as.double(u2),
                n = as.integer(n),
                param = as.double(theta),
                copula = as.integer(rep(family, n)),
                out = numeric(n),
                PACKAGE = "VineCopula")$out

d2_result <- .C("diff2lPDF_mod",  # Using log second derivative version
                u = as.double(u1),
                v = as.double(u2),
                n = as.integer(n),
                param = as.double(theta),
                copula = as.integer(rep(family, n)),
                out = numeric(n),
                PACKAGE = "VineCopula")$out

# Create result object matching bicoppd1d2 output structure
result <- list(
    pdf = pdf_result,
    d1 = d1_result,
    d2 = d2_result
)

# Print results
cat("PDF values:\n")
print(result$pdf)
cat("\nFirst derivatives (log):\n") 
print(result$d1)
cat("\nSecond derivatives (log):\n")
print(result$d2)

cat("\nResult object:\n")
print(result)