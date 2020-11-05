## Elastic net: use a fixed sequence of 100 lambda values

# Generate some example training data
set.seed(1991)
n = 100; p = 100; q=2; s=10


x = matrix(rnorm(n*p),n,p)

beta = cbind(c(rnorm(p-s),rep(0,s)),c(rnorm(p-s),rep(0,s)))

y = x %*% beta +mvtnorm::rmvnorm(100,sigma=matrix(c(1,.4,.4,1),nrow=q))


# Generate some example test data
n0 = 1
x0 = matrix(rnorm(n0*p),n0,p)
y0 = x0 %*% beta + mvtnorm::rmvnorm(n0,sigma=matrix(c(1,.4,.4,1),nrow=q))

out.gnet = cv.glmnet(x,y,alpha=0.5,nlambda=100,lambda.min.ratio=1e-4,family='mgauss')


lambda = out.gnet$lambda.1se


funs = elastic.funs(gamma=0.5,lambda=lambda,family='mgauss',penalty.factor = rep(1,p))
train.fun=funs$train.fun
predict.fun=funs$predict.fun

out = conformal.multi.pred(x, y, x0,num.grid.pts.dim = 20,
train.fun=funs$train, predict.fun=funs$predict,ncm.method = "l2",verbose=T)

#library(ggplot2)

ggplot(out, aes(x=Var1,y=Var2,fill=p_value,z=p_value)) +
  geom_raster() + 
  stat_contour()
