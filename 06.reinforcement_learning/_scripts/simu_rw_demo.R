# simulate data using Rescorla-Wagner Rl model

#### =========================================================
# solution for simulating multiple participants
#### =========================================================
alpha_mu = 0.3 
tau_mu = 2 

alpha_sd = 0.3
tau_sd = 1.2

ns = 10

tmp = 1

while ( tmp ) {
    alpha = rnorm(ns, alpha_mu, alpha_sd)
    if (sum(alpha < 0) == 0 & sum(alpha > 1)==0 ) { tmp = 0 }
}

tmp1 = 1
while ( tmp1 ) {
    tau = rnorm(ns, tau_mu, tau_sd)
    if (sum(tau < 0) == 0 & sum(tau > 5)==0 ) { tmp1 = 0 }
}


#### =========================================================
# simulation for one person
#### =========================================================

alpha = 0.3 # learning rate
tau = 2     # inverse temperature
nt = 40    # number of trials
p = 0.7     # reward probality of option 1

# w - winner, 1 or 2, who actually returns probablistic reward
# c - choice, 1 or 2
# r - reward, 1 or -1, according to choice

w = rbinom(nt, 1, p)
w[w==0] = 2
c = c() # init an empty c
r = c() # init an empty r

v = c(0,0) # initial value

for (t in 1:nt) {
    # ap - action prob of option 2, from value with softmax
    ap = 1 / (1 + exp(-tau*(v[1] - v[2]))) 
    c[t] = 2 - rbinom(1, 1, ap) # choices are 1 and 2
    r[t] = ifelse(c[t]==w[t], 1, -1)
    pe = r[t] - v[c[t]]
    v[c[t]] = v[c[t]] + alpha * pe
}

rl_ss = matrix(, nrow = nt, ncol = 2)
rl_ss[,1] = c
rl_ss[,2] = r

save(rl_ss, file = '_data/rl_sp_ss_simulation_demo.RData')















