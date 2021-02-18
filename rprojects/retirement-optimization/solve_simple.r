library("ggplot2")
library("manipulate")

# Tax function - using 2020 single filers tax brackets
BRACKETS <- list(rate=c(0.1, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37),
                 cutoffs=c(9875, 40125, 85525, 163300, 207350, 518400, Inf))

clip <- function(vec, LB=0, UB=10) pmax( LB, pmin( vec, UB))
shift <- function(x, by) c(rep(0,times=by), x[1:length(x) - by])

tax_bracket_fn_helper <- function(income) {
  lb <- shift(BRACKETS$cutoffs, 1)
  ub <- BRACKETS$cutoffs
  if (income == 0) {
    BRACKETS$rate[1]
  } else {
    bracket_mask <- (income >  lb) & (income <= ub)
    stopifnot(sum(bracket_mask) == 1)
    BRACKETS$rate[[which(bracket_mask == TRUE)]]
  }
}
tax_bracket_fn <- Vectorize(tax_bracket_fn_helper)

avg_tax_fn_helper <- function(income) {
  lb <- shift(BRACKETS$cutoffs, 1)
  ub <- BRACKETS$cutoffs
  if (income == 0) {
    BRACKETS$rate[1]
  } else {
    (BRACKETS$rate %*% (clip(income, LB=lb, UB=ub) - lb)) / income 
  }
}
avg_tax_fn <- Vectorize(avg_tax_fn_helper)

objective <- function(d_t, fund_limit, contribution_limit, interest_rate, income, n_invest_years, n_retirement_years) {
  r <- interest_rate
  n.i <- n_invest_years
  n.r <- n_retirement_years
  Rd <- r * (1 - r^n.i) / (1 - r) * r^n.r
  Rw <- r * (1 - r^n.r) / (1 - r)
  
  I <- income
  L <- fund_limit
  C <- contribution_limit
  
  mu <- Rd / Rw
  beta <- 1 - avg_tax_fn(d_t * mu)
  alpha <- 1 - avg_tax_fn(I - d_t)
  
  d_r <- pmin(L- d_t, (C - d_t) / alpha)
  beta * d_t * mu + d_r * alpha * mu
}

CONTRIBUTION_LIMIT <- 6e3
FUND_LIMIT <- 7e3
INTEREST_RATE <- 1.08
INCOME <- 1e5
START_INVEST_AGE <- 22
START_RETIREMENT_AGE <- 60
DEATH_AGE <- 100
manipulate(plot(1:5, cex=size), size = slider(0.5,10,step=0.5))
generate_plot <- function(fund_limit, interest_rate, income, start_invest_age, start_retirement_age, death_age) {
  traditional_deposit <- 0:min(CONTRIBUTION_LIMIT, fund_limit)
  n.i <- start_retirement_age - start_invest_age
  n.r <- death_age - start_retirement_age
  withdrawals <- objective(traditional_deposit, fund_limit, CONTRIBUTION_LIMIT, interest_rate, income, n.i, n.r)
  
  data <- data.frame(traditional_deposit = traditional_deposit, retirement_value = withdrawals)
  ggplot(data=data, mapping=aes(x=data$traditional_deposit, y=data$retirement_value)) + geom_line() +
    xlab("Traditional Deposit ($/yr)") + ylab("Total Withdrawals - Roth and Trad ($/yr)")
}

manipulate(
  generate_plot(fund_limit=fund_limit, interest_rate=interest_rate, income=income, start_invest_age=start_invest_age, start_retirement_age = start_retirement_age, death_age = death_age),
  fund_limit=slider(0, 2e4, initial = FUND_LIMIT), 
  interest_rate=slider(0.5, 2.0, initial = INTEREST_RATE), 
  income=slider(0, 1e6, initial = INCOME), 
  start_invest_age=slider(18, 59, initial = START_INVEST_AGE), 
  start_retirement_age=slider(60, 99, initial = START_RETIREMENT_AGE), 
  death_age=slider(70, 120, initial=DEATH_AGE))
