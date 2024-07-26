// Prior Predictive Check

data {
    int<lower=0> T; // Time Steps
}


generated quantities {
    real alpha = normal_rng(10, 0.1);
    real<lower=0> sigma = abs(normal_rng(0, 8));
    array[2] real beta;
    array[T] real y_sim;

    for (i in 1:2) {
        beta[i] = normal_rng(0.2, 0.1);
    }

    for (t in 1:2) {
        y_sim[t] = normal_rng(9, 0.1);
    }

    for (t in 3:T) {
        y_sim[t] = normal_rng(alpha + beta[1] * y_sim[t-1] + beta[2] * y_sim[t-2], sigma);
    }
}