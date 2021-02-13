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

year2ind <- function(year) { year - startyear.i + 1 }
age2ind <- function(age) { age - startage.i + 1 }

PARAM.NAMES <- c('d.roth', 'd.trad', 'w.roth', 'w.trad')

# This is used to pass nloptr a single vector to optimize
pack_params <- function(unpacked_params) {
  vals <- c(unpacked_params$d.roth$vals, unpacked_params$d.trad$vals, unpacked_params$w.roth$vals, unpacked_params$w.trad$vals)
  fixed <- c(unpacked_params$d.roth$fixed, unpacked_params$d.trad$fixed, unpacked_params$w.roth$fixed, unpacked_params$w.trad$fixed)
  
  list(vals=vals, fixed=fixed)
}

# This is used to unpack the single vector for nloptr into something more manageable
unpack_params <- function(packed_params, size) {
  params <- list()
  params$d.roth$vals <- packed_params$vals[(1: size)]
  params$d.roth$fixed <- packed_params$fixed[(1: size)]
  params$d.trad$vals <- packed_params$vals[(size + 1):(2 * size)]
  params$d.trad$fixed <- packed_params$fixed[(size + 1):(2 * size)]
  params$w.roth$vals <- packed_params$vals[(2 * size + 1):(3 * size)]
  params$w.roth$fixed <- packed_params$fixed[(2 * size + 1):(3 * size)]
  params$w.trad$vals <- packed_params$vals[(3 * size + 1):(4 * size)]
  params$w.trad$fixed <- packed_params$fixed[(3 * size + 1):(4 * size)]
  
  params
}

fix_param <- function(param, indexes, vals) {
  param$vals[indexes] <- vals
  param$fixed[indexes] <- TRUE
  param
}

init_params <- function(n.i, n.r){
  size <- n.i + n.r
  vals <- vector(mode="numeric", length=(size))
  fixed <- vector(mode="logical", length=(size))
  params <- rep(list(list(vals=vals, fixed=fixed)), times=length(PARAM.NAMES))
  names(params) <- PARAM.NAMES
  # Fix retirement withdrawals to 0 when not retired
  params$w.roth <- fix_param(params$w.roth, 1:n.i, 0)
  params$w.trad <- fix_param(params$w.trad, 1:n.i, 0)
  # Fix retirement deposits to 0 when retired
  params$d.roth <- fix_param(params$d.roth, (n.i + 1):(size), 0)
  params$d.trad <- fix_param(params$d.trad, (n.i + 1):(size), 0)
  
  params
}

params <- init_params(n.i = n.i, n.r = n.r)

# Fix any variables and set up known values
# TODO
incomes <- vector(mode="numeric", length=(n.i + n.r))
incomes[year2ind(2019)] <- 9e5
incomes[year2ind(2020)] <- 8.5e5
incomes[year2ind(2021):year2ind(2026)] <- 2e5

# Estimate unknown values
incomes[year2ind(2027):n.i] <- 5e5
interest.rates <- rep(1.08, (n.i + n.r))

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

# Account value functions - deposits and withdrawals are pre-tax
interest.cum <- rev(cumprod(interest.rates))

value.roth <- function(d.roth.vals, w.roth.vals, incomes, interest.cum) {
  interest.cum %*% (d.roth.vals * (1 - tax_fn(incomes))) - interest.cum %*% w.roth.vals
}
value.trad <- function(d.trad.vals, w.trad.vals, interest.cum) {
  interest.cum %*% d.trad.vals - interest.cum %*% w.trad.vals
}

# Objective - maximize the total amount I get from withdrawals after tax
eval_obj <- function(x, fixed, incomes, interest.cum, size) { #TODO
  params <- unpack_params(list(vals=x, fixed=fixed, size))
  # Minimizing the negative sum
  -sum(params$w.roth$vals + params$w.trad$vals * (1 - tax_fn(params$w.trad$vals)))
}

# Inequality Constraints - should be in form g(x) <= 0
IRA_CONTRIB_LIM <- 6e3

ira.limit.ineq <- function(d.roth.vals, d.trad.vals, incomes) {
  (d.roth.vals * (1 - tax_fn(incomes))) + d.trad.vals - IRA_CONTRIB_LIM
}
posdef.transfer <- function(d.roth.vals, d.trad.vals, w.roth.vals, w.trad.vals) {
  c(-d.roth.vals, -d.trad.vals, -w.roth.vals, -w.trad.vals)
}
posdef.account.roth <- function(d.roth.vals, w.roth.vals, incomes, interest.cum) {
  -value.roth(d.roth.vals, w.roth.vals, incomes, interest.cum)
}
posdef.account.trad <- function(d.trad.vals, w.trad.vals, interest.cum) {
  -value.trad(d.trad.vals, w.trad.vals, interest.cum)
}

# Currently only uses final value of account to determine if this was a legal deposit/withdrawal schedule
# This is okay as long as deposits and withdrawals do not overlap, since a negative balance => final value will be negative
# However, this needs to be changed if I allow withdrawals and deposits at the same time.
eval_ineq <- function(x, fixed, incomes, interest.cum, size) {
  params <- unpack_params(list(vals=x, fixed=fixed), size)
  c(ira.limit.ineq(params$d.roth$vals, params$d.trad$vals, incomes),
    posdef.transfer(params$d.roth$vals, params$d.trad$vals, params$w.roth$vals, params$w.trad$vals),
    posdef.account.roth(params$d.roth$vals, params$w.roth$vals, incomes, interest.cum),
    posdef.account.trad(params$d.trad$vals, params$w.trad$vals, interest.cum))
}

# Equality Constraints - should be in form h(x) = 0
eval_eq <- function(x, fixed, incomes, interest.cum, size) {
  x * fixed # TODO figure out how to check fixed params
}

# Optimize

