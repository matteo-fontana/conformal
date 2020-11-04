## Elastic net: use a fixed sequence of 100 lambda values

# Generate some example training data
set.seed(1991)
n = 200; p = 500; s = 10; q=2
x = matrix(rnorm(n*p),n,p)
beta = matrix(rep(c(rnorm(s),rep(0,p-s)),q),ncol=q) 
y = x %*% beta + rnorm(n)

#simpler data
n = 200; p = 500; s = 10; q=2




# Generate some example test data
n0 = 1
x0 = matrix(rnorm(n0*p),n0,p)
y0 = x0 %*% beta + rnorm(n0)


lambda = 10000
funs = elastic.funs(gamma=0,lambda=lambda,family='mgauss')
train.fun=funs$train
predict.fun=funs$predict.fun


out = conformal.multi.pred(x, y, x0,num.grid.pts.dim = 10,
  train.fun=funs$train, predict.fun=funs$predict,ncm.method = "l2",verbose=T)

