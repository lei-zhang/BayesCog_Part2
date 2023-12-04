data {
    int<lower=1> nSubjects;
    int<lower=1> nTrials;
    int<lower=1,upper=2> choice[nSubjects, nTrials];
    int<lower=-1,upper=1> reward[nSubjects, nTrials];
}

transformed data{
    vector[2] initV;
    initV = rep_vector(0, 2);
}

parameters {
    // group level parameter (hyper-params), raw space
    real mu_alpha_raw;
    real<lower=0> sigma_alpha;
    real mu_tau_raw;
    real<lower=0> sigma_tau;

    // individual params, raw space <-- to be sampled from std normal
    vector[nSubjects] alpha_raw;
    vector[nSubjects] tau_raw;
}

transformed parameters {
    real<lower=0,upper=1> mu_alpha;
    real<lower=0,upper=20> mu_tau;
    vector<lower=0,upper=1>[nSubjects] alpha; // learning rate
    vector<lower=0,upper=20>[nSubjects] tau;  // softmax inv.temp.

    mu_alpha = Phi_approx(mu_alpha_raw);
    mu_tau = Phi_approx(mu_tau_raw) * 20;

    alpha = Phi_approx(alpha_raw * sigma_alpha + mu_alpha_raw);
    tau = Phi_approx(tau_raw * sigma_tau + mu_tau_raw) * 20;
}

model {
    // priors for group-level params
    mu_alpha_raw ~ normal(0,1); //std_normal()
    sigma_alpha ~ cauchy(0,1);
    mu_tau_raw ~ normal(0,5); 
    sigma_tau ~ cauchy(0,5);

    // priors for indv params
    alpha_raw ~ normal(0,1);
    tau_raw ~ normal(0,1);
    
    for (s in 1:nSubjects) {
        real pe;
        vector[2] v;

        v = initV;

        for (t in 1:nTrials) {            
            choice[s,t] ~ categorical_logit(tau[s] * v);
            
            pe = reward[s,t] - v[choice[s,t]]; // prediction error
            v[choice[s,t]] = v[choice[s,t]] + alpha[s] * pe; // value update, chosen V only
        }
    }
}

generated quantities { // this is run only after the sampling is done, ie, working with posteriors
    int y_pred [nSubjects, nTrials];
    real log_lik[nSubjects];

    for (s in 1:nSubjects) {
        real pe;
        vector[2] v;

        v = initV;
        log_lik[s] = 0;

        for (t in 1:nTrials) {            
            y_pred[s,t] = categorical_logit_rng(tau[s] * v); // simulate "fake" choices from the posterior
            log_lik[s] = log_lik[s] + categorical_logit_lpmf( choice[s,t] | tau[s] * v);
            
            pe = reward[s,t] - v[choice[s,t]]; // prediction error
            v[choice[s,t]] = v[choice[s,t]] + alpha[s] * pe; // value update, chosen V only
        }
    }
}

