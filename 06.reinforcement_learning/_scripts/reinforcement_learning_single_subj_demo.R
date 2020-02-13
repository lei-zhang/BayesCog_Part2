# =============================================================================
#### Info #### 
# =============================================================================
# simple reinforcement learning model
# single true parameters, true lr = 0.6, tau = 1.5, pRew = 0.7
#
# Lei Zhang
# lei.zhang@univie.ac.at


# =============================================================================
#### Construct Data #### 
# =============================================================================
# clear workspace
library(rstan)
library(ggplot2)

load('_data/rl_sp_ss.RData')
sz <- dim(rl_ss)
nTrials <- sz[1]

dataList <- list(nTrials=nTrials, 
                 choice=rl_ss[,1], 
                 reward=rl_ss[,2])

# get variable names
str(dataList)

# =============================================================================
#### Running Stan #### 
# =============================================================================
rstan_options(auto_write = TRUE)
options(mc.cores = 4)

modelFile <- '_scripts/rw.stan'

nIter     <- 2000
nChains   <- 4 
nWarmup   <- floor(nIter/2)
nThin     <- 1
pars      <- c('alpha','tau','v','pe','y_pred','acc', 'v_chn', 'log_lik')

cat("Estimating", modelFile, "model... \n")
startTime = Sys.time(); print(startTime)
cat("Calling", nChains, "simulations in Stan... \n")

fit_rl <- stan(modelFile, 
               data    = dataList, 
               chains  = nChains,
               pars    = pars,
               iter    = nIter,
               warmup  = nWarmup,
               thin    = nThin,
               init    = "random",
               seed    = 1450154626)

cat("Finishing", modelFile, "model simulation ... \n")
endTime = Sys.time(); print(endTime)  
cat("It took",as.character.Date(endTime - startTime), "\n")

# =============================================================================
#### Model Summary and Diagnostics #### 
# =============================================================================
print(fit_rl)

stan_trace(fit_rl, pars = c('alpha','tau'), inc_warmup = F)
stan_plot(fit_rl, pars=c('alpha','tau'), show_density=T, fill_color = 'skyblue')

# =============================================================================
#### plot internal model variables
# =============================================================================

# get trial-by-trial chosen value and pe
v_chn = get_posterior_mean(fit_rl, pars=c('v_chn'))[,5]
pe = get_posterior_mean(fit_rl, pars=c('pe'))[,5]

# save in a dataframe for ggplot
df = data.frame(trial  = 1:dataList$nTrials,
                value  = v_chn,
                pe     = pe)

#### make plots of choice, reward, v(chn), and pe
library(ggplot2)
myconfig <- theme_bw(base_size = 20) +
    theme(panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.background = element_blank() )

g1 <- ggplot(df, aes(x=trial, y=value)) 
g1 <- g1 + geom_line(size = 2, color = 'black') + geom_point(size = 3, shape = 21, fill='black')
g1 <- g1 + myconfig + labs(x = 'Trial', y = 'Chosen Value')
g1

g2 <- ggplot(df, aes(x=trial, y=pe)) 
g2 <- g2 + geom_line(size = 2, color = 'black') + geom_point(size = 3, shape = 21, fill='black')
g2 <- g2 + myconfig + labs(x = 'Trial', y = 'Prediction Error')
g2



