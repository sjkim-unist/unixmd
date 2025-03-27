#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <string.h>
#include "derivs_xf.h"

// Complex datatype
struct _dcomplex {double real, imag;};
typedef struct _dcomplex dcomplex;

// Importing heev and gemm
extern void zheev_(char *jobz, char *uplo, int *n, dcomplex *a, int *lda, double *w, dcomplex *work, int *lwork, double *rwork, int *info);
extern void zgemm_(char *transa, char *transb, int *m, int *n, int *k, dcomplex *alpha, dcomplex *a, int *lda,
    dcomplex *b, int *ldb, dcomplex *beta, dcomplex *c, int *ldc);

// Routine for coefficient propagation scheme in exponential propagator
static void exponential_coef(int nat, int ndim, int nst, int nesteps, double dt, int *l_coh,
    double *mass, double *energy, double *energy_old, double **sigma, double **nacme,
    double **nacme_old, double **pos, double **qmom, double ***aux_pos, double ***phase, double complex *coef,
    int verbosity, double *dotpopdec);

// Interface routine for propagation scheme in exponential propagator
static void exponential(int nat, int ndim, int nst, int nesteps, double dt, char *elec_object, int *l_coh,
    double *mass, double *energy, double *energy_old, double **sigma, double **nacme, double **nacme_old,
    double **pos, double **qmom, double ***aux_pos, double ***phase, double complex *coef, int verbosity,
    double *dotpopdec){

    if(strcmp(elec_object, "coefficient") == 0){
        exponential_coef(nat, ndim, nst, nesteps, dt, l_coh, mass, energy, energy_old, sigma,
            nacme, nacme_old, pos, qmom, aux_pos, phase, coef, verbosity, dotpopdec);
    }
//    else if(strcmp(elec_object, "density") == 0){
//        exponential_rho(nat, ndim, nst, nesteps, dt, l_coh, mass, energy, energy_old, sigma,
//            nacme, nacme_old, pos, qmom, aux_pos, phase, rho, verbosity, dotpopdec);
//    }

}

static void exponential_coef(int nat, int ndim, int nst, int nesteps, double dt, int *l_coh,
    double *mass, double *energy, double *energy_old, double **sigma, double **nacme,
    double **nacme_old, double **pos, double **qmom, double ***aux_pos, double ***phase, double complex *coef,
    int verbosity, double *dotpopdec){

    double *eenergy = malloc(nst * sizeof(double));
    double **dv = malloc(nst * sizeof(double*));
    double *rwork = malloc((3 * nst - 2) * sizeof(double));
    double **dec = malloc(nst * sizeof(double*));
    double complex **rho = malloc(nst * sizeof(double complex*));
    double complex **dec_h = malloc(nst * sizeof(double complex*));
    double complex *coef_new = malloc(nst * sizeof(double complex));

    // (Hamiltonian - i * (NACME + decoherence)) * dt
    double complex **exponent = malloc(nst * sizeof(double complex*)); 
    // eigenvectors of (energy - i * (NACME + decoherence)) * dt, P
    dcomplex *eigenvectors = malloc((nst * nst) * sizeof(dcomplex));
    // eigenvalues of (energy - i * (NACME + decoherence)) * dt, D
    double *eigenvalues = malloc(nst * sizeof(double));

    // Diagonal matrix using eigenvalues, exp(- i * D)
    dcomplex *exp_idiag = malloc((nst * nst) * sizeof(dcomplex));
    // exp(- i * exponent) = P * exp(- i * D) * P^-1
    dcomplex *exp_iexponent = malloc((nst * nst) * sizeof(dcomplex));
    // product of P * exp(- i * D) * P^-1 until previous step
    dcomplex *product_old = malloc((nst * nst) * sizeof(dcomplex));
    // product of (P * exp(- i * D) * P^-1) until current step
    dcomplex *product_new = malloc((nst * nst) * sizeof(dcomplex));
    // final product of exp(- i * exponent)
    double complex **propagator = malloc(nst * sizeof(double complex*));

    dcomplex *tmp_mat = malloc((nst * nst) * sizeof(dcomplex));
    dcomplex *identity = malloc((nst * nst) * sizeof(dcomplex));
    double complex *tmp_dec = malloc((nst * nst) * sizeof(double complex));

    int ist, jst, isp, iat, iestep, lwork, info;
    double frac, edt;
    double complex tmp_coef;

    for(ist = 0; ist < nst; ist++){
        dv[ist] = malloc(nst * sizeof(double));
        dec[ist] = malloc(nst * sizeof(double));
        rho[ist] = malloc(nst * sizeof(double complex));
        dec_h[ist] = malloc(nst * sizeof(double complex));
        exponent[ist] = malloc(nst * sizeof(double complex));
        propagator[ist] = malloc(nst * sizeof(double complex));
    }

    dcomplex dcone = {1.0, 0.0};
    dcomplex dczero = {0.0, 0.0};
    dcomplex wkopt;
    dcomplex *work;

    for(ist = 0; ist < nst; ist++){
        // Diagonal element
        identity[nst * ist + ist].real = 1.0;
        identity[nst * ist + ist].imag = 0.0;
        product_old[nst * ist + ist].real = 1.0;
        product_old[nst * ist + ist].imag = 0.0;
        for(jst = ist + 1; jst < nst; jst++){
            // Off-diagonal elements
            // Upper triangle
            identity[nst * ist + jst].real = 0.0;
            identity[nst * ist + jst].imag = 0.0;
            product_old[nst * ist + jst].real = 0.0;
            product_old[nst * ist + jst].imag = 0.0;
            // Lower triangle
            identity[nst * jst + ist].real = 0.0;
            identity[nst * jst + ist].imag = 0.0;
            product_old[nst * jst + ist].real = 0.0;
            product_old[nst * jst + ist].imag = 0.0;
        }
    }

    for(ist = 0; ist < nst * nst; ist++){
        exp_idiag[ist].real = 0.0;
        exp_idiag[ist].imag = 0.0;
    }

    for(ist = 0; ist < nst; ist++){
        free(propagator[ist]);
        free(exponent[ist]);
        free(dec_h[ist]);
        free(rho[ist]);
        free(dec[ist]);
        free(dv[ist]);
    }

    free(tmp_dec);
    free(coef_new);
    free(propagator);
    free(exp_iexponent);
    free(identity);
    free(product_new);
    free(product_old);
    free(eigenvectors);
    free(exp_idiag);
    free(tmp_mat);
    free(eigenvalues);
    free(rwork);
    free(exponent);
    free(dec_h);
    free(rho);
    free(dec);
    free(eenergy);
    free(dv);

}
