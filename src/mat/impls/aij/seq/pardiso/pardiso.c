
#include <../src/mat/impls/aij/seq/aij.h>        /*I "petscmat.h" I*/
#include <../src/mat/impls/sbaij/seq/sbaij.h>    /*I "petscmat.h" I*/
#include <../src/mat/impls/dense/seq/dense.h>    /*I "petscmat.h" I*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define JOB_ANALYSIS 11
#define JOB_ANALYSIS_NUMERICAL_FACTORIZATION 12
#define JOB_ANALYSIS_NUMERICAL_FACTORIZATION_SOLVE_ITERATIVE_REFINEMENT 13
#define JOB_NUMERICAL_FACTORIZATION 22
#define JOB_NUMERICAL_FACTORIZATION_SOLVE_ITERATIVE_REFINEMENT 23
#define JOB_SOLVE_ITERATIVE_REFINEMENT 33
#define JOB_RELEASE_OF_LU_MEMORY 0
#define JOB_RELEASE_OF_ALL_MEMORY -1

#define IPARM_SIZE 64
#define DPARM_SIZE 64

/* Note: this implementation does not allow for any floating point type other than double, yet */
#define DOUBLE_TYPE double

/* Note: this implementation does not allow for 64-bit integers, yet
   (but see the MKL_pardiso interface for some logic on how to define
   these values to extend to that case) */
#define INT_TYPE int
#define PARDISO pardiso
#define PARDISO_INIT pardisoinit

/* Pardiso Function Prototypes */
extern void pardisoinit(void *pt, INT_TYPE *mtype, INT_TYPE *solver, INT_TYPE iparm[], DOUBLE_TYPE dparm[],INT_TYPE *err);
extern void pardiso(void *pt, INT_TYPE *maxfct, INT_TYPE *mnum, INT_TYPE *mtype, INT_TYPE *phase, INT_TYPE *n, DOUBLE_TYPE a[], INT_TYPE ia[], INT_TYPE ja[],INT_TYPE perm[], INT_TYPE *nrhs, INT_TYPE iparm[], INT_TYPE *msglvl, DOUBLE_TYPE b[], DOUBLE_TYPE x[], INT_TYPE *error, DOUBLE_TYPE dparm[]);
extern void pardiso_chkmatrix(INT_TYPE *mtype, INT_TYPE *n, DOUBLE_TYPE a[], INT_TYPE ia[], INT_TYPE ja[], INT_TYPE *err);

/*
 *  Internal data structure.
 *  For more information, see the Pardiso manual
 */
typedef struct {

  /* Configuration vectors*/
  INT_TYPE     iparm[IPARM_SIZE];
  DOUBLE_TYPE  dparm[DPARM_SIZE];

  /*
   * Internal pardiso memory location.
   * After the first call to pardiso, do not modify pt, as that could cause a serious memory leak.
   */
  void         *pt[IPARM_SIZE];

  /* Basic pardiso info*/
  INT_TYPE     phase, maxfct, mnum, mtype, n, nrhs, msglvl, err;
  INT_TYPE     solver;

  /* Matrix structure*/
  void         *a;
  INT_TYPE     *ia, *ja;

  /* Number of non-zero elements*/
  INT_TYPE     nz;

  /* Row permutaton vector*/
  INT_TYPE     *perm;

  /* Define if matrix preserves sparse structure.*/
  MatStructure matstruc;

  /* True if pardiso functions have been used.*/
  PetscBool CleanUp;
} Mat_PARDISO;

/* Convert 0-based indexing to 1-based indexing */
PETSC_STATIC_INLINE void zeroToOneIndexing(Mat_PARDISO *mat_pardiso)
{
   INT_TYPE *ia = mat_pardiso->ia;
   INT_TYPE *ja = mat_pardiso->ja;
   INT_TYPE n = mat_pardiso->n;
   INT_TYPE nz = mat_pardiso->nz;
   int i;

   for (i = 0; i <= n; i++)
      ++ia[i];
   for (i = 0; i < nz; i++)
      ++ja[i];
}

/* Convert 1-based indexing to 0-based indexing */
PETSC_STATIC_INLINE void oneToZeroIndexing(Mat_PARDISO *mat_pardiso)
{
   INT_TYPE *ia = mat_pardiso->ia;
   INT_TYPE *ja = mat_pardiso->ja;
   INT_TYPE n = mat_pardiso->n;
   INT_TYPE nz = mat_pardiso->nz;
   int i;

   for (i = 0; i <= n; i++)
      --ia[i];
   for (i = 0; i < nz; i++)
      --ja[i];
}

PETSC_STATIC_INLINE PetscErrorCode MatCopy_PARDISO(Mat A, MatReuse reuse, INT_TYPE *nnz, INT_TYPE **r, INT_TYPE **c, void **v)
{
  Mat_SeqAIJ *aa=(Mat_SeqAIJ*)A->data;

  PetscFunctionBegin;
  *v=aa->a;
  if (reuse == MAT_INITIAL_MATRIX) {
    *r   = (INT_TYPE*)aa->i;
    *c   = (INT_TYPE*)aa->j;
    *nnz = aa->nz;
  }
  PetscFunctionReturn(0);
}

/*
 * Free memory for Mat_PARDISO structure and pointers to objects.
 */
PetscErrorCode MatDestroy_PARDISO(Mat A)
{
  Mat_PARDISO *mat_pardiso=(Mat_PARDISO*)A->spptr;
  PetscBool       isSeqSBAIJ;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  /* Terminate instance, deallocate memory */
  if (mat_pardiso->CleanUp) {
    mat_pardiso->phase = JOB_RELEASE_OF_ALL_MEMORY;

   zeroToOneIndexing(mat_pardiso);
#if defined(PETSC_USE_DEBUG)
    pardiso_chkmatrix(
    &mat_pardiso->mtype,
    &mat_pardiso->n,
    mat_pardiso->a,
    mat_pardiso->ia,
    mat_pardiso->ja,
    &mat_pardiso->err
    );
#endif
    PARDISO (mat_pardiso->pt,
      &mat_pardiso->maxfct,
      &mat_pardiso->mnum,
      &mat_pardiso->mtype,
      &mat_pardiso->phase,
      &mat_pardiso->n,
      NULL,
      NULL,
      NULL,
      mat_pardiso->perm,
      &mat_pardiso->nrhs,
      mat_pardiso->iparm,
      &mat_pardiso->msglvl,
      NULL,
      NULL,
      &mat_pardiso->err,
      mat_pardiso->dparm);
   oneToZeroIndexing(mat_pardiso);
  }
  ierr = PetscFree(mat_pardiso->perm);CHKERRQ(ierr);
  ierr = PetscFree(A->spptr);CHKERRQ(ierr);

  /* clear composed functions */
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatFactorGetSolverType_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"Mat_PardisoSetCntl_C",NULL);CHKERRQ(ierr);

  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQSBAIJ,&isSeqSBAIJ);CHKERRQ(ierr);
  if (isSeqSBAIJ) {ierr = MatDestroy_SeqSBAIJ(A);CHKERRQ(ierr);}
  else            {ierr = MatDestroy_SeqAIJ(A);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*
 * Computes Ax = b
 */
PetscErrorCode MatSolve_PARDISO(Mat A,Vec b,Vec x)
{
  Mat_PARDISO   *mat_pardiso=(Mat_PARDISO*)(A)->spptr;
  PetscErrorCode    ierr;
  PetscScalar       *xarray;
  const PetscScalar *barray;

  PetscFunctionBegin;
  mat_pardiso->nrhs = 1;
  ierr = VecGetArray(x,&xarray);CHKERRQ(ierr);
  ierr = VecGetArrayRead(b,&barray);CHKERRQ(ierr);

  /* solve phase */
  mat_pardiso->phase = JOB_SOLVE_ITERATIVE_REFINEMENT;
   zeroToOneIndexing(mat_pardiso);
#if defined(PETSC_USE_DEBUG)
    pardiso_chkmatrix(
    &mat_pardiso->mtype,
    &mat_pardiso->n,
    mat_pardiso->a,
    mat_pardiso->ia,
    mat_pardiso->ja,
    &mat_pardiso->err
    );
#endif
  PARDISO (mat_pardiso->pt, 
    &mat_pardiso->maxfct,
    &mat_pardiso->mnum,
    &mat_pardiso->mtype,
    &mat_pardiso->phase,
    &mat_pardiso->n,
    mat_pardiso->a,
    mat_pardiso->ia,
    mat_pardiso->ja,
    mat_pardiso->perm,
    &mat_pardiso->nrhs,
    mat_pardiso->iparm,
    &mat_pardiso->msglvl,
    (void*)barray,
    (void*)xarray,
    &mat_pardiso->err,
    mat_pardiso->dparm);
   oneToZeroIndexing(mat_pardiso);

  if (mat_pardiso->err < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by PARDISO: err=%d. Please check manual\n",mat_pardiso->err);
  ierr = VecRestoreArray(x,&xarray);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(b,&barray);CHKERRQ(ierr);
  mat_pardiso->CleanUp = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolveTranspose_PARDISO(Mat A,Vec b,Vec x)
{
  Mat_PARDISO *mat_pardiso=(Mat_PARDISO*)A->spptr;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  mat_pardiso->iparm[12 - 1] = 1;
#else
  mat_pardiso->iparm[12 - 1] = 2;
#endif
  ierr = MatSolve_PARDISO(A,b,x);CHKERRQ(ierr);
  mat_pardiso->iparm[12 - 1] = 0;
  PetscFunctionReturn(0);
}


PetscErrorCode MatMatSolve_PARDISO(Mat A,Mat B,Mat X)
{
  Mat_PARDISO   *mat_pardiso=(Mat_PARDISO*)(A)->spptr;
  PetscErrorCode    ierr;
  PetscScalar       *barray, *xarray;
  PetscBool         flg;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)B,MATSEQDENSE,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Matrix B must be MATSEQDENSE matrix");
  ierr = PetscObjectTypeCompare((PetscObject)X,MATSEQDENSE,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Matrix X must be MATSEQDENSE matrix");

  ierr = MatGetSize(B,NULL,(PetscInt*)&mat_pardiso->nrhs);CHKERRQ(ierr);

  if(mat_pardiso->nrhs > 0){
    ierr = MatDenseGetArray(B,&barray);
    ierr = MatDenseGetArray(X,&xarray);

    /* solve phase */
    mat_pardiso->phase = JOB_SOLVE_ITERATIVE_REFINEMENT;
    zeroToOneIndexing(mat_pardiso);
#if defined(PETSC_USE_DEBUG)
    pardiso_chkmatrix(
    &mat_pardiso->mtype,
    &mat_pardiso->n,
    mat_pardiso->a,
    mat_pardiso->ia,
    mat_pardiso->ja,
    &mat_pardiso->err
    );
#endif
    PARDISO (mat_pardiso->pt,
      &mat_pardiso->maxfct,
      &mat_pardiso->mnum,
      &mat_pardiso->mtype,
      &mat_pardiso->phase,
      &mat_pardiso->n,
      mat_pardiso->a,
      mat_pardiso->ia,
      mat_pardiso->ja,
      mat_pardiso->perm,
      &mat_pardiso->nrhs,
      mat_pardiso->iparm,
      &mat_pardiso->msglvl,
      (void*)barray,
      (void*)xarray,
      &mat_pardiso->err,
      mat_pardiso->dparm);
   oneToZeroIndexing(mat_pardiso);
    if (mat_pardiso->err < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by PARDISO: err=%d. Please check manual\n",mat_pardiso->err);
  }
  mat_pardiso->CleanUp = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*
 * LU Decomposition
 */
PetscErrorCode MatFactorNumeric_PARDISO(Mat F,Mat A,const MatFactorInfo *info)
{
  Mat_PARDISO *mat_pardiso=(Mat_PARDISO*)(F)->spptr;
  PetscErrorCode  ierr;

  /* numerical factorization phase */
  PetscFunctionBegin;
  mat_pardiso->matstruc = SAME_NONZERO_PATTERN;
  ierr = MatCopy_PARDISO(A, MAT_REUSE_MATRIX, &mat_pardiso->nz, &mat_pardiso->ia, &mat_pardiso->ja, &mat_pardiso->a);CHKERRQ(ierr);

  /* numerical factorization phase */
  mat_pardiso->phase = JOB_NUMERICAL_FACTORIZATION;
   zeroToOneIndexing(mat_pardiso);
#if defined(PETSC_USE_DEBUG)
    pardiso_chkmatrix(
    &mat_pardiso->mtype,
    &mat_pardiso->n,
    mat_pardiso->a,
    mat_pardiso->ia,
    mat_pardiso->ja,
    &mat_pardiso->err
    );
#endif
  PARDISO (mat_pardiso->pt,
    &mat_pardiso->maxfct,
    &mat_pardiso->mnum,
    &mat_pardiso->mtype,
    &mat_pardiso->phase,
    &mat_pardiso->n,
    mat_pardiso->a,
    mat_pardiso->ia,
    mat_pardiso->ja,
    mat_pardiso->perm,
    &mat_pardiso->nrhs,
    mat_pardiso->iparm,
    &mat_pardiso->msglvl,
    NULL,
    NULL,
    &mat_pardiso->err,
    mat_pardiso->dparm);
   oneToZeroIndexing(mat_pardiso);
  if (mat_pardiso->err < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by PARDISO: err=%d. Please check manual\n",mat_pardiso->err);

  mat_pardiso->matstruc = SAME_NONZERO_PATTERN;
  mat_pardiso->CleanUp  = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/* Sets pardiso options from the options database */
PetscErrorCode PetscSetPARDISOFromOptions(Mat F, Mat A)
{
  Mat_PARDISO     *mat_pardiso = (Mat_PARDISO*)F->spptr;
  PetscErrorCode  ierr;
  PetscInt        icntl, threads = 1;
  PetscReal       dcntl;
  PetscBool       flg;

  PetscFunctionBegin;

  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)A),((PetscObject)A)->prefix,"PARDISO Options","Mat");CHKERRQ(ierr);

  ierr = PetscOptionsInt("-mat_pardiso_solver","Which solver to use","None",mat_pardiso->solver,&icntl,&flg);CHKERRQ(ierr);
  if (flg) mat_pardiso->solver = icntl;

  ierr = PetscOptionsInt("-mat_pardiso_65","Number of threads to use","None",threads,&threads,&flg);CHKERRQ(ierr);
  if (flg) omp_set_num_threads((int)threads);

  ierr = PetscOptionsInt("-mat_pardiso_66","Maximum number of factors with identical sparsity structure that must be kept in memory at the same time","None",mat_pardiso->maxfct,&icntl,&flg);CHKERRQ(ierr);
  if (flg) mat_pardiso->maxfct = icntl;

  ierr = PetscOptionsInt("-mat_pardiso_67","Indicates the actual matrix for the solution phase","None",mat_pardiso->mnum,&icntl,&flg);CHKERRQ(ierr);
  if (flg) mat_pardiso->mnum = icntl;
 
  ierr = PetscOptionsInt("-mat_pardiso_68","Message level information","None",mat_pardiso->msglvl,&icntl,&flg);CHKERRQ(ierr);
  if (flg) mat_pardiso->msglvl = icntl;

  ierr = PetscOptionsInt("-mat_pardiso_69","Defines the matrix type","None",mat_pardiso->mtype,&icntl,&flg);CHKERRQ(ierr);
  if(flg){
    mat_pardiso->mtype = icntl;
    PARDISO_INIT(mat_pardiso->pt, &mat_pardiso->mtype, &mat_pardiso->solver, mat_pardiso->iparm, mat_pardiso->dparm, &mat_pardiso->err);
  }
  ierr = PetscOptionsInt("-mat_pardiso_1","Use default values","None",mat_pardiso->iparm[0],&icntl,&flg);CHKERRQ(ierr);
  if (flg) mat_pardiso->iparm[1-1] = icntl;

  /* Skip processing if iparm(1) is zero, because PARDISO will then use default values */
  if(mat_pardiso->iparm[0]){
    ierr = PetscOptionsInt("-mat_pardiso_2","METIS fill-in reducing ordering for the input matrix","None",mat_pardiso->iparm[2-1],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_pardiso->iparm[2-1] = icntl;

    ierr = PetscOptionsInt("-mat_pardiso_4","Preconditioned CGS/CG","None",mat_pardiso->iparm[4-1],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_pardiso->iparm[4-1] = icntl;

    ierr = PetscOptionsInt("-mat_pardiso_5","User permutation","None",mat_pardiso->iparm[5-1],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_pardiso->iparm[5-1] = icntl;

    ierr = PetscOptionsInt("-mat_pardiso_6","Write solution on x","None",mat_pardiso->iparm[7-1],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_pardiso->iparm[6-1] = icntl;

    ierr = PetscOptionsInt("-mat_pardiso_8","Iterative refinement step","None",mat_pardiso->iparm[8-1],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_pardiso->iparm[8-1] = icntl;

    ierr = PetscOptionsInt("-mat_pardiso_10","Pivoting perturbation","None",mat_pardiso->iparm[10-1],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_pardiso->iparm[10-1] = icntl;

    ierr = PetscOptionsInt("-mat_pardiso_11","Scaling vectors","None",mat_pardiso->iparm[11-1],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_pardiso->iparm[11-1] = icntl;

    ierr = PetscOptionsInt("-mat_pardiso_12","Solve with transposed or conjugate transposed matrix A","None",mat_pardiso->iparm[12-1],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_pardiso->iparm[12-1] = icntl;

    ierr = PetscOptionsInt("-mat_pardiso_13","Improved accuracy using (non-) symmetric weighted matching","None",mat_pardiso->iparm[13-1],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_pardiso->iparm[13-1] = icntl;

    ierr = PetscOptionsInt("-mat_pardiso_18","Numbers of non-zero elements","None",mat_pardiso->iparm[18-1],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_pardiso->iparm[18-1] = icntl;

    ierr = PetscOptionsInt("-mat_pardiso_19","Report Gflops for LU factorization","None",mat_pardiso->iparm[19-1],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_pardiso->iparm[19-1] = icntl;

    ierr = PetscOptionsInt("-mat_pardiso_21","Pivoting for symmetric indefinite matrices","None",mat_pardiso->iparm[21-1],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_pardiso->iparm[21-1] = icntl;

    ierr = PetscOptionsInt("-mat_pardiso_24","Parallel factorization control","None",mat_pardiso->iparm[24-1],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_pardiso->iparm[24-1] = icntl;

    ierr = PetscOptionsInt("-mat_pardiso_25","Parallel forward/backward solve control","None",mat_pardiso->iparm[25-1],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_pardiso->iparm[25-1] = icntl;

    ierr = PetscOptionsInt("-mat_pardiso_30","Supernode size","None",mat_pardiso->iparm[30-1],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_pardiso->iparm[30-1] = icntl;

    ierr = PetscOptionsInt("-mat_pardiso_31","Partial solve and computing selected components of the solution vectors","None",mat_pardiso->iparm[31-1],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_pardiso->iparm[31-1] = icntl;

    ierr = PetscOptionsInt("-mat_pardiso_32","Use the multi-recursive iterative linear solver","None",mat_pardiso->iparm[32-1],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_pardiso->iparm[32-1] = icntl;

    ierr = PetscOptionsInt("-mat_pardiso_34","Optimal number of threads for conditional numerical reproducibility (CNR) mode","None",mat_pardiso->iparm[34-1],&icntl,&flg);CHKERRQ(ierr);
    if (flg) mat_pardiso->iparm[34-1] = icntl;

  }

  /* Accept DPARM arguments */
  ierr = PetscOptionsReal("-mat_pardiso_dparm_1","","None",mat_pardiso->dparm[1-1],&dcntl,&flg);CHKERRQ(ierr);
  if (flg) mat_pardiso->dparm[1-1] = dcntl;
  ierr = PetscOptionsReal("-mat_pardiso_dparm_2","","None",mat_pardiso->dparm[2-1],&dcntl,&flg);CHKERRQ(ierr);
  if (flg) mat_pardiso->dparm[2-1] = dcntl;
  ierr = PetscOptionsReal("-mat_pardiso_dparm_3","","None",mat_pardiso->dparm[3-1],&dcntl,&flg);CHKERRQ(ierr);
  if (flg) mat_pardiso->dparm[3-1] = dcntl;
  ierr = PetscOptionsReal("-mat_pardiso_dparm_4","","None",mat_pardiso->dparm[4-1],&dcntl,&flg);CHKERRQ(ierr);
  if (flg) mat_pardiso->dparm[4-1] = dcntl;
  ierr = PetscOptionsReal("-mat_pardiso_dparm_5","","None",mat_pardiso->dparm[5-1],&dcntl,&flg);CHKERRQ(ierr);
  if (flg) mat_pardiso->dparm[5-1] = dcntl;
  ierr = PetscOptionsReal("-mat_pardiso_dparm_6","","None",mat_pardiso->dparm[6-1],&dcntl,&flg);CHKERRQ(ierr);
  if (flg) mat_pardiso->dparm[6-1] = dcntl;
  ierr = PetscOptionsReal("-mat_pardiso_dparm_7","","None",mat_pardiso->dparm[7-1],&dcntl,&flg);CHKERRQ(ierr);
  if (flg) mat_pardiso->dparm[7-1] = dcntl;
  ierr = PetscOptionsReal("-mat_pardiso_dparm_8","","None",mat_pardiso->dparm[8-1],&dcntl,&flg);CHKERRQ(ierr);
  if (flg) mat_pardiso->dparm[8-1] = dcntl;
  ierr = PetscOptionsReal("-mat_pardiso_dparm_9","","None",mat_pardiso->dparm[9-1],&dcntl,&flg);CHKERRQ(ierr);
  if (flg) mat_pardiso->dparm[9-1] = dcntl;
  ierr = PetscOptionsReal("-mat_pardiso_dparm_33","","None",mat_pardiso->dparm[33-1],&dcntl,&flg);CHKERRQ(ierr);
  if (flg) mat_pardiso->dparm[33-1] = dcntl;
  ierr = PetscOptionsReal("-mat_pardiso_dparm_34","","None",mat_pardiso->dparm[34-1],&dcntl,&flg);CHKERRQ(ierr);
  if (flg) mat_pardiso->dparm[34-1] = dcntl;
  ierr = PetscOptionsReal("-mat_pardiso_dparm_35","","None",mat_pardiso->dparm[35-1],&dcntl,&flg);CHKERRQ(ierr);
  if (flg) mat_pardiso->dparm[35-1] = dcntl;

  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

PetscErrorCode MatFactorPARDISOInitialize_Private(Mat A, MatFactorType ftype, Mat_PARDISO *mat_pardiso)
{
  PetscErrorCode ierr;
  PetscInt       i;


  PetscFunctionBegin;
  for ( i = 0; i < IPARM_SIZE; i++ ){
    mat_pardiso->iparm[i] = 0;
  }

  /* Note: no attempt to set reasonable defaults has been made
     for the DPARM settings, so the user should set all of them */
  for ( i = 0; i < DPARM_SIZE; i++ ){
    mat_pardiso->dparm[i] = 0.0;
  }

  for ( i = 0; i < IPARM_SIZE; i++ ){
    mat_pardiso->pt[i] = 0;
  }

  /* Default options for both sym and unsym */
  mat_pardiso->iparm[ 1-1] =  1; /* Solver default parameters overridden with provided by iparm */
  mat_pardiso->iparm[ 2-1] =  2; /* Metis reordering */
  mat_pardiso->iparm[ 6-1] =  0; /* Write solution into x */
  mat_pardiso->iparm[ 8-1] =  2; /* Max number of iterative refinement steps */
  mat_pardiso->iparm[10-1] = 13; /* Perturb the pivot elements with 1E-13 */
  mat_pardiso->iparm[18-1] = -1; /* Output: Number of nonzeros in the factor LU */
  mat_pardiso->iparm[19-1] = -1; /* Output: Gflops for LU factorization */
  mat_pardiso->iparm[25-1] =  1; /* Parallel Forward/Backward Solve */
  mat_pardiso->iparm[35-1] =  1; /* Cluster Sparse Solver use C-style indexing for ia and ja arrays */
  mat_pardiso->iparm[40-1] =  0; /* Input: matrix/rhs/solution stored on master */
  
  mat_pardiso->CleanUp   = PETSC_FALSE;
  mat_pardiso->maxfct    = 1; /* Maximum number of numerical factorizations. */
  mat_pardiso->mnum      = 1; /* Which factorization to use. */
  mat_pardiso->msglvl    = 0; /* 0: do not print 1: Print statistical information in file */
  mat_pardiso->phase     = -1;
  mat_pardiso->err       = 0;

  mat_pardiso->solver    = 0; /* 0: direct solver 1: multi-recursive iterative solver (initialize dparm) */
  
  mat_pardiso->n         = A->rmap->N;
  mat_pardiso->nrhs      = 1;
  mat_pardiso->err       = 0;
  mat_pardiso->phase     = -1;
  
  if(ftype == MAT_FACTOR_LU){
    /* Default type for non-sym */
#if defined(PETSC_USE_COMPLEX)
    mat_pardiso->mtype     = 13;
#else
    mat_pardiso->mtype     = 11;
#endif

    mat_pardiso->iparm[11-1] =  1; /* Use nonsymmetric permutation and scaling MPS */
    mat_pardiso->iparm[13-1] =  1; /* Switch on Maximum Weighted Matching algorithm (default for non-symmetric) */

  } else {
    /* Default type for sym */
#if defined(PETSC_USE_COMPLEX)
    mat_pardiso ->mtype    = 3;
#else
    mat_pardiso ->mtype    = -2;
#endif
    mat_pardiso->iparm[11-1] = 0;  /* Use nonsymmetric permutation and scaling MPS */
    mat_pardiso->iparm[13-1] = 1;  /* Switch on Maximum Weighted Matching algorithm (default for non-symmetric) */
    mat_pardiso->iparm[21-1] =  1; /* Apply 1x1 and 2x2 Bunch-Kaufman pivoting during the factorization process */
  }
  ierr = PetscMalloc1(A->rmap->N*sizeof(INT_TYPE), &mat_pardiso->perm);CHKERRQ(ierr);
  for(i = 0; i < A->rmap->N; i++){
    mat_pardiso->perm[i] = 0;
  }

  PARDISO_INIT(mat_pardiso->pt, &mat_pardiso->mtype, &mat_pardiso->solver, mat_pardiso->iparm, mat_pardiso->dparm, &mat_pardiso->err);

  if (mat_pardiso->err < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by PARDISO after initialization: err=%d\n. Please check manual",mat_pardiso->err);

  PetscFunctionReturn(0);
}

/*
 * Symbolic decomposition. Pardiso analysis phase.
 */
PetscErrorCode MatFactorSymbolic_AIJPARDISO_Private(Mat F,Mat A,const MatFactorInfo *info)
{
  Mat_PARDISO *mat_pardiso = (Mat_PARDISO*)F->spptr;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  mat_pardiso->matstruc = DIFFERENT_NONZERO_PATTERN;

  /* Set PARDISO options from the options database */
  ierr = PetscSetPARDISOFromOptions(F,A);CHKERRQ(ierr);

  ierr = MatCopy_PARDISO(A, MAT_INITIAL_MATRIX, &mat_pardiso->nz, &mat_pardiso->ia, &mat_pardiso->ja, &mat_pardiso->a);CHKERRQ(ierr);
  mat_pardiso->n = A->rmap->N;

  /* analysis phase */
  mat_pardiso->phase = JOB_ANALYSIS;

   zeroToOneIndexing(mat_pardiso);
#if defined(PETSC_USE_DEBUG)
    pardiso_chkmatrix(
    &mat_pardiso->mtype,
    &mat_pardiso->n,
    mat_pardiso->a,
    mat_pardiso->ia,
    mat_pardiso->ja,
    &mat_pardiso->err
    );
#endif
  PARDISO (
    mat_pardiso->pt, 
    &mat_pardiso->maxfct,
    &mat_pardiso->mnum,
    &mat_pardiso->mtype,
    &mat_pardiso->phase,
    &mat_pardiso->n,
    mat_pardiso->a,
    mat_pardiso->ia,
    mat_pardiso->ja,
    mat_pardiso->perm,
    &mat_pardiso->nrhs,
    mat_pardiso->iparm,
    &mat_pardiso->msglvl,
    NULL,
    NULL,
    &mat_pardiso->err,
    mat_pardiso->dparm);
   oneToZeroIndexing(mat_pardiso);
  if (mat_pardiso->err < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by PARDISO: err=%d\n. Please check manual",mat_pardiso->err);

  mat_pardiso->CleanUp = PETSC_TRUE;

  if(F->factortype == MAT_FACTOR_LU){
    F->ops->lufactornumeric = MatFactorNumeric_PARDISO;
  } else {
    F->ops->choleskyfactornumeric = MatFactorNumeric_PARDISO;
  }
  F->ops->solve           = MatSolve_PARDISO;
  F->ops->solvetranspose  = MatSolveTranspose_PARDISO;
  F->ops->matsolve        = MatMatSolve_PARDISO;
  PetscFunctionReturn(0);
}

PetscErrorCode MatLUFactorSymbolic_AIJPARDISO(Mat F,Mat A,IS r,IS c,const MatFactorInfo *info)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatFactorSymbolic_AIJPARDISO_Private(F, A, info);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatCholeskyFactorSymbolic_AIJPARDISO(Mat F,Mat A,IS r,const MatFactorInfo *info)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatFactorSymbolic_AIJPARDISO_Private(F, A, info);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatView_PARDISO(Mat A, PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscBool         iascii;
  PetscViewerFormat format;
  Mat_PARDISO   *mat_pardiso=(Mat_PARDISO*)A->spptr;
  PetscInt          i;

  PetscFunctionBegin;
  /* check if matrix is pardiso type */
  if (A->ops->solve != MatSolve_PARDISO) PetscFunctionReturn(0);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO) {
      ierr = PetscViewerASCIIPrintf(viewer,"PARDISO run parameters:\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"PARDISO phase:             %d \n",mat_pardiso->phase);CHKERRQ(ierr);
      for(i = 1; i <= 64; i++){
        ierr = PetscViewerASCIIPrintf(viewer,"PARDISO iparm[%d]:     %d \n",i, mat_pardiso->iparm[i - 1]);CHKERRQ(ierr);
      }
      if (mat_pardiso->iparm[32-1]){
        for(i = 1; i <= 64; i++){
          ierr = PetscViewerASCIIPrintf(viewer,"PARDISO dparm[%d]:     %f \n",i, (double) mat_pardiso->dparm[i - 1]);CHKERRQ(ierr);
        }
      }
      ierr = PetscViewerASCIIPrintf(viewer,"PARDISO maxfct:     %d \n", mat_pardiso->maxfct);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"PARDISO mnum:     %d \n", mat_pardiso->mnum);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"PARDISO mtype:     %d \n", mat_pardiso->mtype);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"PARDISO n:     %d \n", mat_pardiso->n);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"PARDISO nrhs:     %d \n", mat_pardiso->nrhs);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"PARDISO msglvl:     %d \n", mat_pardiso->msglvl);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}


PetscErrorCode MatGetInfo_PARDISO(Mat A, MatInfoType flag, MatInfo *info)
{
  Mat_PARDISO *mat_pardiso =(Mat_PARDISO*)A->spptr;

  PetscFunctionBegin;
  info->block_size        = 1.0;
  info->nz_allocated      = mat_pardiso->nz + 0.0;
  info->nz_unneeded       = 0.0;
  info->assemblies        = 0.0;
  info->mallocs           = 0.0;
  info->memory            = 0.0;
  info->fill_ratio_given  = 0;
  info->fill_ratio_needed = 0;
  info->factor_mallocs    = 0;
  PetscFunctionReturn(0);
}

PetscErrorCode MatPardisoSetCntl_PARDISO(Mat F,PetscInt icntl,PetscInt ival)
{
  Mat_PARDISO *mat_pardiso =(Mat_PARDISO*)F->spptr;

  int err = 0;

  PetscFunctionBegin;
  if(icntl <= 64){
    mat_pardiso->iparm[icntl - 1] = ival;
  } else {
    if(icntl == 65)
      omp_set_num_threads((int)ival);
    else if(icntl == 66)
      mat_pardiso->maxfct = ival;
    else if(icntl == 67)
      mat_pardiso->mnum = ival;
    else if(icntl == 68)
      mat_pardiso->msglvl = ival;
    else if(icntl == 69){
      void *pt[IPARM_SIZE];
      mat_pardiso->mtype = ival;
      PARDISO_INIT(pt, &mat_pardiso->mtype, &mat_pardiso->solver, mat_pardiso->iparm, mat_pardiso->dparm, &err);
    }
  }
  PetscFunctionReturn(0);
}

/*@
  MatPardisoSetCntl - Set Pardiso parameters

   Logically Collective on Mat

   Input Parameters:
+  F - the factored matrix obtained by calling MatGetFactor()
.  icntl - index of Pardiso parameter
-  ival - value of Pardiso parameter

  Options Database:
.   -mat_pardiso_<icntl> <ival>

   Level: beginner

   References: Pardiso Users' Guide

.seealso: MatGetFactor()
@*/
PetscErrorCode MatPardisoSetCntl(Mat F,PetscInt icntl,PetscInt ival)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTryMethod(F,"MatPardisoSetCntl_C",(Mat,PetscInt,PetscInt),(F,icntl,ival));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
  MATSOLVERPARDISO -  A matrix type providing direct solvers (LU) for
  sequential matrices via the external package PARDISO.

  Works with MATSEQAIJ matrices

  Use -pc_type <lu,cholesky> -pc_factor_mat_solver_package pardiso to us this direct solver

  Options Database Keys:
+ -mat_pardiso_solver - Solver (0: direct, 1: multi-recursive iterative)
+ -mat_pardiso_65 - Number of threads to use
. -mat_pardiso_66 - Maximum number of factors with identical sparsity structure that must be kept in memory at the same time
. -mat_pardiso_67 - Indicates the actual matrix for the solution phase
. -mat_pardiso_68 - Message level information
. -mat_pardiso_69 - Defines the matrix type. IMPORTANT: When you set this flag, iparm parameters are going to be set to the default ones for the matrix type
. -mat_pardiso_1 - Use default values
. -mat_pardiso_2 - Fill-in reducing ordering for the input matrix
. -mat_pardiso_4 - Preconditioned CGS/CG
. -mat_pardiso_5 - User permutation
. -mat_pardiso_6 - Write solution on x
. -mat_pardiso_8 - Iterative refinement step
. -mat_pardiso_10 - Pivoting perturbation
. -mat_pardiso_11 - Scaling vectors
. -mat_pardiso_12 - Solve with transposed or conjugate transposed matrix A
. -mat_pardiso_13 - Improved accuracy using (non-) symmetric weighted matching
. -mat_pardiso_18 - Numbers of non-zero elements
. -mat_pardiso_19 - Report Gflops for LU factorization
. -mat_pardiso_21 - Pivoting for symmetric indefinite matrices
. -mat_pardiso_24 - Parallel factorization control
. -mat_pardiso_25 - Parallel forward/backward solve control
. -mat_pardiso_30 - Supernode size
. -mat_pardiso_31 - Partial solve and computing selected components of the solution vectors
. -mat_pardiso_32 - Use multi-recursive solver (use -mat_pardiso_solver 1 to set default values for dparm)
. -mat_pardiso_34 - Optimal number of threads for conditional numerical reproducibility (CNR) mode
. -mat_pardiso_dparm_1 - set DPARM(1). This pattern applies for other DPARM values (see Pardiso manual).

  Level: beginner

  For more information please check pardiso manual

.seealso: PCFactorSetMatSolverType(), MatSolverType

M*/
static PetscErrorCode MatFactorGetSolverType_pardiso(Mat A,MatSolverType *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERPARDISO;
  PetscFunctionReturn(0);
}

/* MatGetFactor for Seq sbAIJ matrices */
PETSC_EXTERN PetscErrorCode MatGetFactor_sbaij_pardiso(Mat A,MatFactorType ftype,Mat *F)
{
  Mat             B;
  PetscErrorCode  ierr;
  Mat_PARDISO     *mat_pardiso;
  PetscBool       isSeqSBAIJ;
  PetscInt        bs;

  PetscFunctionBegin;

  /* Create the factorization matrix */
  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQSBAIJ,&isSeqSBAIJ);CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject)A),&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
  ierr = MatSetType(B,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr = MatSeqSBAIJSetPreallocation(B,1,0,NULL);CHKERRQ(ierr);
  ierr = MatGetBlockSize(A,&bs); CHKERRQ(ierr);

  if(bs != 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Matrice MATSEQSBAIJ with block size other than 1 is not supported by Pardiso");
  if(ftype != MAT_FACTOR_CHOLESKY) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Matrice MATSEQAIJ should be used only with MAT_FACTOR_CHOLESKY.");
  
  B->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_AIJPARDISO;
  B->factortype                  = MAT_FACTOR_CHOLESKY;
  B->ops->destroy                = MatDestroy_PARDISO;
  B->ops->view                   = MatView_PARDISO;
  B->factortype                  = ftype;
  B->ops->getinfo                = MatGetInfo_PARDISO;
  B->assembled                   = PETSC_TRUE;           /* required by -ksp_view */

  ierr = PetscNewLog(B,&mat_pardiso);CHKERRQ(ierr);
  B->spptr = mat_pardiso;
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatFactorGetSolverType_C",MatFactorGetSolverType_pardiso);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatPardisoSetCntl_C",MatPardisoSetCntl_PARDISO);CHKERRQ(ierr);
  ierr = MatFactorPARDISOInitialize_Private(A, ftype, mat_pardiso);CHKERRQ(ierr);
  *F = B;
  PetscFunctionReturn(0);
}

/* MatGetFactor for Seq AIJ matrices */
PETSC_EXTERN PetscErrorCode MatGetFactor_aij_pardiso(Mat A,MatFactorType ftype,Mat *F)
{
  Mat             B;
  PetscErrorCode  ierr;
  Mat_PARDISO *mat_pardiso;
  PetscBool       isSeqAIJ;

  PetscFunctionBegin;
  /* Create the factorization matrix */
  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQAIJ,&isSeqAIJ);CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject)A),&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
  ierr = MatSetType(B,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(B,0,NULL);CHKERRQ(ierr);

  if(ftype != MAT_FACTOR_LU) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Matrice MATSEQAIJ should be used only with MAT_FACTOR_LU.");

  B->ops->lufactorsymbolic = MatLUFactorSymbolic_AIJPARDISO;
  B->factortype            = MAT_FACTOR_LU;
  B->ops->destroy          = MatDestroy_PARDISO;
  B->ops->view             = MatView_PARDISO;
  B->factortype            = ftype;
  B->ops->getinfo          = MatGetInfo_PARDISO;
  B->assembled             = PETSC_TRUE;           /* required by -ksp_view */

  ierr = PetscNewLog(B,&mat_pardiso);CHKERRQ(ierr);
  B->spptr = mat_pardiso;
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatFactorGetSolverType_C",MatFactorGetSolverType_pardiso);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatPardisoSetCntl_C",MatPardisoSetCntl_PARDISO);CHKERRQ(ierr);
  ierr = MatFactorPARDISOInitialize_Private(A, ftype, mat_pardiso);CHKERRQ(ierr);

  *F = B;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_Pardiso(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSolverTypeRegister(MATSOLVERPARDISO,MATSEQAIJ,   MAT_FACTOR_LU,      MatGetFactor_aij_pardiso  );CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERPARDISO,MATSEQSBAIJ, MAT_FACTOR_CHOLESKY,MatGetFactor_sbaij_pardiso);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef PARDISO
#undef PARDISO_INIT
