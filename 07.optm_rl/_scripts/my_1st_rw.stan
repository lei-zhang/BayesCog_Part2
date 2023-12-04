data {
    int<lower=1> nTrials;
    int<lower=1,upper=2> choice[nTrials];
    int<lower=-1,upper=1> reward[nTrials];
}

transformed data{
    vector[2] initV;
    initV = rep_vector(0, 2);
}

parameters {
    real<lower=0,upper=1> alpha; // learning rate
    real<lower=0,upper=20> tau;  // softmax inv.temp.
}

model {
    real pe;
    vector[2] v;
    vector[2] p;
    
    v = initV;

    for (t in 1:nTrials) {
        //p = softmax( tau * v); // compute action prob
        //choice[t] ~ categorical(p); // likelihood to model choice

        //choice[t] ~ categorical(softmax(tau * v));
        
        choice[t] ~ categorical_logit(tau * v);
        
        pe = reward[t] - v[choice[t]]; // prediction error
        v[choice[t]] = v[choice[t]] + alpha * pe; // value update, chosen V only
    }
}
