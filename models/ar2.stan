

// Posterior Predictive Check 

data {
    int<lower=0> T; // Time Steps

    vector[T] y; 
}

parameters {
    real alpha;
    array[2] real beta;
    real<lower=0> sigma;
}

model {
    alpha ~ normal(10, 0.1);
    beta ~ normal(0.2, 0.1);
    sigma ~ normal (0, 8);

    for (t in 3:T) {
        y[t] ~ normal(alpha + beta[1] * y[t-1] + beta[2] * y[t-2], sigma);
      }
}

generated quantities {
    vector[T] y_rep;  

    for (t in 1:2) {
        y_rep[t] = normal_rng(9, 0.1);
    }

    for (t in 3:T) {
        y_rep[t] = normal_rng(alpha + beta[1] * y[t-1] + beta[2] * y[t-2], sigma);
    }
}
