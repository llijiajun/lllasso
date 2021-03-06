// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// slamuda
double slamuda(double z, double lamuda);
RcppExport SEXP lllasso_slamuda(SEXP zSEXP, SEXP lamudaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double >::type z(zSEXP);
    Rcpp::traits::input_parameter< double >::type lamuda(lamudaSEXP);
    rcpp_result_gen = Rcpp::wrap(slamuda(z, lamuda));
    return rcpp_result_gen;
END_RCPP
}
// cacur
arma::vec cacur(const arma::vec& y, const arma::mat& x, const arma::vec& belta, int k, int ro);
RcppExport SEXP lllasso_cacur(SEXP ySEXP, SEXP xSEXP, SEXP beltaSEXP, SEXP kSEXP, SEXP roSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type x(xSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type belta(beltaSEXP);
    Rcpp::traits::input_parameter< int >::type k(kSEXP);
    Rcpp::traits::input_parameter< int >::type ro(roSEXP);
    rcpp_result_gen = Rcpp::wrap(cacur(y, x, belta, k, ro));
    return rcpp_result_gen;
END_RCPP
}
// cacub
double cacub(double lamuda, const arma::vec& r, const arma::mat& x, int k, int ro);
RcppExport SEXP lllasso_cacub(SEXP lamudaSEXP, SEXP rSEXP, SEXP xSEXP, SEXP kSEXP, SEXP roSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double >::type lamuda(lamudaSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type r(rSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type x(xSEXP);
    Rcpp::traits::input_parameter< int >::type k(kSEXP);
    Rcpp::traits::input_parameter< int >::type ro(roSEXP);
    rcpp_result_gen = Rcpp::wrap(cacub(lamuda, r, x, k, ro));
    return rcpp_result_gen;
END_RCPP
}
// lasso2
arma::vec lasso2(arma::vec y, arma::mat x, double lamuda);
RcppExport SEXP lllasso_lasso2(SEXP ySEXP, SEXP xSEXP, SEXP lamudaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< double >::type lamuda(lamudaSEXP);
    rcpp_result_gen = Rcpp::wrap(lasso2(y, x, lamuda));
    return rcpp_result_gen;
END_RCPP
}
