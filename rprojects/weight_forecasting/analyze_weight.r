require(forecast)

# Load data
weight_file <- "~/gdrive/sync-documents/data/weight.csv"
weightdf <- read.csv(weight_file)
names(weightdf) <- c('time', 'weight')
weightdf <- weightdf[, c('time', 'weight')]
weightdf$time <- as.Date(weightdf$time, format="%b %d, %Y %I:%M:%S %p")
# Remove an outlier (I measured that day with backpack and clothes on)
weightdf <- weightdf[-which(weightdf$time == "2020-01-30" & weightdf$weight == 153.0),]
# Aggregate with min measurements (assumed to be without any additional factors like food affecting measurement)
weightdf <- aggregate(list(weight = weightdf$weight), by=list(time = weightdf$time), FUN=min)

# Fit a line and plot data
ggplot(weightdf, aes(x=time, y=weight)) + geom_point() + geom_smooth(method="lm")
fit <-lm(weight ~ time, data=weightdf)
# Notice that there is significant autocorrelation for lags up to 10
# Breusch-Godfrey test shows p-value of < 2.2e-16 (very significant)
# There is additional patterns to mine from data
checkresiduals(fit)
# Residuals shows clear pattern
cbind(Fitted = fitted(fit),
      Residuals=residuals(fit)) %>%
  as.data.frame() %>%
  ggplot(aes(x=Fitted, y=Residuals)) + geom_point()


# Convert to timeseries
full_dates <- seq(min(weightdf$time), max(weightdf$time), by=1)
full_dates <- data.frame(time=full_dates)
weight_mins <- aggregate(weightdf, by=list("time"), FUN=min)
weightdf <-merge(full_dates, weightdf, by="time", all.x = TRUE)
weightts <- ts(weightdf$weight, start=min(weightdf$time), end=max(weightdf$time))
# Interp where I have data
weightts2 = c(na.interp(weightts[1:235]), weightts[236:288], na.interp(weightts[289:366]))

