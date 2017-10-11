#include<RcppArmadillo.h>
#include<stdio.h>
using namespace arma;

// This is a simple example of exporting a C++ function to R. You can
// source this function into an R session using the Rcpp::sourceCpp
// function (or via the Source button on the editor toolbar). Learn
// more about Rcpp at:
//
//   http://www.rcpp.org/
//   http://adv-r.had.co.nz/Rcpp.html
//   http://gallery.rcpp.org/
//

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
double slamuda(double z,double lamuda){
  if(z>lamuda){
    return (z-lamuda);
  }else if(z<(-lamuda)){
    return (z+lamuda);
  }else{
    return 0;
  }
}

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::vec cacur(const arma::vec& y,const arma::mat& x,const arma::vec &belta,int k,int ro){
  arma::vec r(ro);
  int i;
  for(i=0;i<ro;i++){
    r(i)=y(i)+belta(k-1)*x(i,k-1)-dot(vectorise(x.submat(i,0,i,x.n_cols-1)),belta);
  }
  return r;
}

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
double cacub(double lamuda,const arma::vec &r,const arma::mat& x,int k,int ro){
  double v=dot(vectorise(x.submat(0,k-1,x.n_rows-1,k-1)),r)/double(ro);
  double belta=slamuda(v,lamuda);
  return belta;
}

double mean(arma::vec x){
  double me=0;
  int i,n=(int)x.n_elem;
  for(i=0;i<n;i++){
    me+=x(i);
  }
  me=me/(double)n;
  return me;
}

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::vec lasso2(arma::vec y,arma::mat x,double lamuda){
  int ro=(int)x.n_rows;
  int co=(int)x.n_cols;
  int i;
  arma::vec yy=y;
  arma::mat xx=x;
  for(i=0;i<co;i++){
    double m=mean(vectorise(x.submat(0,i,x.n_rows-1,i)));
    x.submat(0,i,x.n_rows-1,i)=x.submat(0,i,x.n_rows-1,i)-m;
    double sq=dot(vectorise(x.submat(0,i,x.n_rows-1,i)),vectorise(x.submat(0,i,x.n_rows-1,i)));
    x.submat(0,i,x.n_rows-1,i)=x.submat(0,i,x.n_rows-1,i)*(sqrt(ro/sq));
  }
  y=y-mean(y);
  double d=1;
  int k=1,j=0;
  arma::vec belta(co,fill::zeros);
  arma::vec q(co,fill::zeros);
  arma::vec yhat=x*belta;
  arma::vec newyhat=yhat;
  while(d>0.001 || j<=co){
    arma::vec r=cacur(y,x,belta,k,ro);
    belta(k-1)=cacub(lamuda,r,x,k,ro);
    newyhat=x*belta;
    if(k<co){
      k=k+1;
    }else {
      k=1;
    }
    d=dot(yhat-newyhat,yhat-newyhat)/dot(yhat,yhat);
    yhat=newyhat;
    j+=1;
  }
  arma::vec beltahat(co+1,fill::zeros);
  arma::vec xcmean(co,fill::zeros);
  for(i=0;i<co;i++){
    xcmean(i)=mean(vectorise(xx.submat(0,i,x.n_rows-1,i)));
  }
  arma::vec sq(co,fill::zeros);
  for(i=0;i<co;i++){
    double m=mean(vectorise(xx.submat(0,i,xx.n_rows-1,i)));
    xx.submat(0,i,x.n_rows-1,i)=xx.submat(0,i,x.n_rows-1,i)-m;
    sq(i)=dot(vectorise(xx.submat(0,i,xx.n_rows-1,i)),vectorise(xx.submat(0,i,xx.n_rows-1,i)));
  }
  for(i=0;i<co;i++){
    belta(i)=belta(i)*sqrt(ro/sq(i));
    beltahat(i+1)=belta(i);
  }
  beltahat(0)=mean(yy)-dot(belta,xcmean);
  return beltahat;
}
