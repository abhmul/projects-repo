library("nloptr")

# Set up parameters and constants
birthyear <- 1997
lifespan <- 100
startage.i <- 22
startage.r <- 60

startyear.i <-  birthyear + startage.i # start investing
startyear.r <- birthyear + startage.r # start retiring (end investing)
endyear.r <- birthyear + lifespan
n.i <- startyear.r - startyear.i # number of years of investing
n.r <- endyear.r - startyear.r # number of years of retirement
n.total <- n.i + n.r

year2ind <- function(year) { year - startyear.i + 1 }
age2ind <- function(age) { age - startage.i + 1 }

PARAM.NAMES <- c('d.roth', 'd.trad', 'w.roth', 'w.trad')

unpack_params_simple <- function(packed_params) {
  params <- as.list(packed_params)
  names(params) <- PARAM.NAMES
  params
}

pack_params_simple <- function(params) {
  unlist(params)
}

init_params_simple <- function(init_val=0) {
  unpack_params_simple(rep(init_val, times=length(PARAM.NAMES)))
}

params.simple <- init_params_simple()

income.estimated <- 5e5
interest.rate <- 1.08

# Tax function - using 2020 single filers tax brackets
BRACKETS <- list(rate=c(0.1, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37),
                 cutoffs=c(9875, 40125, 85525, 163300, 207350, 518400, Inf))

clip <- function(vec, LB=0, UB=10) pmax( LB, pmin( vec, UB))
shift <- function(x, by) c(rep(0,times=by), x[1:length(x) - by])

tax_fn_helper <- function(income) {
  lb <- shift(BRACKETS$cutoffs, 1)
  ub <- BRACKETS$cutoffs
  if (income == 0) {
    BRACKETS$rate[1]
  } else {
    (BRACKETS$rate %*% (clip(income, LB=lb, UB=ub) - lb)) / income 
  }
}
tax_fn <- Vectorize(tax_fn_helper)

# Don't count traditional deposits as income
adjust.income <- function(income, d.trad.vals) {
  income - d.trad.vals
}
# value.roth <- function(d.roth, w.roth, adjusted.income, interest.cum, n.i, n.r) {
#   d.roth.vals <- c(rep(d.roth, times=n.i), rep(0, times=n.r))
#   w.roth.vals <- c(rep(0, times=n.i), rep(w.roth, times=n.r))
#   sum(interest.cum * (d.roth.vals * (1 - tax_fn(adjusted.income))) - interest.cum * w.roth.vals)
# }
deposit_mult <- function(interest, n.i, n.r) {
  (1 - interest^n.i) / (1 - interest) * interest * interest^n.r 
}
withdrawal_mult <- function(interest, n.r) {
  (1 - interest^n.r) / (1 - interest) * interest
}
value.roth <- function(d.roth, w.roth, adjusted.income, interest, n.i, n.r) {
  d.mult <- (1 - tax_fn(adjusted.income)) * deposit_mult(interest, n.i, n.r)
  w.mult <- withdrawal_mult(interest, n.r)
  d.roth * d.mult - w.roth * w.mult
}
# value.trad <- function(d.trad, w.trad, interest.cum, n.i, n.r) {
#   d.trad.vals <- c(rep(d.trad, times=n.i), rep(0, times=n.r))
#   w.trad.vals <- c(rep(0, times=n.i), rep(w.trad, times=n.r))
#   sum(interest.cum * d.trad.vals - interest.cum * w.trad.vals)
# }
value.trad <- function(d.trad, w.trad, interest, n.i, n.r) {
  d.mult <- deposit_mult(interest, n.i, n.r)
  w.mult <- withdrawal_mult(interest, n.r)
  d.trad * d.mult - w.trad * w.mult
}

obj_simple_grad <- function(params, tax_mult) {
  params$d.roth <- 0
  params$d.trad <- 0
  params$w.roth <- -1
  params$w.trad <- -tax_mult
  pack_params_simple(params)
}
eval_obj_simple <- function(x, income, interest.rate, n.i, n.r) {
  params <- unpack_params_simple(x)
  tax_mult <- (1 - tax_fn(params$w.trad))
  list(objective=-(params$w.roth + params$w.trad * (1 - tax_fn(params$w.trad))),
       gradient=obj_simple_grad(params, tax_mult)
  )
}

# Inequality Constraints - should be in form g(x) <= 0
IRA_CONTRIB_LIM <- 6e3

ira.limit.ineq.grad <- function(params, tax_mult) {
  params$d.roth <- tax_mult
  params$d.trad <- 1
  params$w.roth <- 0
  params$w.trad <- 0
  pack_params_simple(params)
}
ira.limit.ineq <- function(params, adjusted.incomes) {
  tax_mult <- (1 - tax_fn(adjusted.incomes))
  list(
    inequality=(params$d.roth * tax_mult) + params$d.trad - IRA_CONTRIB_LIM,
    gradient=ira.limit.ineq.grad(params, tax_mult)
  )
}

posdef.account.roth.grad <- function(params, adjusted.income, interest.rate, n.i, n.r) {
  params$d.roth <--1 * (1 - tax_fn(adjusted.income)) * deposit_mult(interest.rate, n.i, n.r)
  params$d.trad <- 0
  params$w.roth <- 1 * withdrawal_mult(interest.rate, n.r)
  params$w.trad <- 0
  pack_params_simple(params)
}
posdef.account.roth <- function(params, adjusted.income, interest.rate, n.i, n.r) {
  list(
    inequality=-value.roth(params$d.roth, params$w.roth, adjusted.income, interest.rate, n.i, n.r),
    gradient=posdef.account.roth.grad(params, adjusted.income, interest.rate, n.i, n.r)
  )
}
posdef.account.trad.grad <- function(params, interest.rate, n.i, n.r) {
  params$d.trad <--1 * deposit_mult(interest.rate, n.i, n.r)
  params$d.roth <- 0
  params$w.trad <- 1 * withdrawal_mult(interest.rate, n.r)
  params$w.roth <- 0
  pack_params_simple(params)
}
posdef.account.trad <- function(params, interest.rate, n.i, n.r) {
  list(
    inequality=-value.trad(params$d.trad, params$w.trad, interest.rate, n.i, n.r),
    gradient=posdef.account.trad.grad(params, interest.rate, n.i, n.r)
  )
}

eval_ineq_simple <- function(x, income, interest.rate, n.i, n.r) {
  params <- unpack_params_simple(x)
  adjusted.income <- adjust.income(income, params$d.trad)
  ira_limit <- ira.limit.ineq(params, adjusted.income)
  roth_posdef <- posdef.account.roth(params, adjusted.income, interest.rate, n.i, n.r)
  trad_posdef <- posdef.account.trad(params, interest.rate, n.i, n.r)
  
  list(
    constraints=c(ira_limit$inequality, roth_posdef$inequality, trad_posdef$inequality),
    jacobian=rbind(c(ira_limit$gradient, roth_posdef$gradient, trad_posdef$gradient))
  )
}

# Simple optimize
local_opts <- list(
  algorithm="NLOPT_LD_TNEWTON",
  xtol_rel=1e-04,
  maxeval=10000
)
opts <- list(
  algorithm="NLOPT_LD_AUGLAG",
  xtol_rel=1e-08,
  maxeval=100000,
  local_opts=local_opts
)
params.packed <- pack_params_simple(params.simple)
res <- nloptr(
  x0=params.packed,
  eval_f=eval_obj_simple,
  lb= rep(0, times=length(params.packed)),
  ub= rep(IRA_CONTRIB_LIM, times=length(params.packed)),
  eval_g_ineq=eval_ineq_simple,
  opts=opts,
  income=income.estimated,
  interest.rate=interest.rate, 
  n.i=n.i, 
  n.r=n.r
)

cat("Failed?: ",  res$status)
cat("Status Message: ", res$message)
cat("Total Withdrawal achieved: ", -res$objective)
params.opt <- unpack_params_simple(res$solution)

cat("Traditional Deposits: ", params.opt$d.trad)
cat("Roth Deposits: ", params.opt$d.roth)
cat("Traditional Withdrawals: ", params.opt$w.trad)
cat("Roth Withdrawals: ", params.opt$w.roth)

