
/*
   This provides a matrix that applies a VecScatter to a vector.
*/

#include <petsc/private/matimpl.h>        /*I "petscmat.h" I*/
#include <petsc/private/vecimpl.h>

typedef struct {
  VecScatter scatter;
} Mat_Scatter;

/*@
    MatScatterGetVecScatter - Returns the user-provided scatter set with MatScatterSetVecScatter()

    Not Collective, but not cannot use scatter if not used collectively on Mat

    Input Parameter:
.   mat - the matrix, should have been created with MatCreateScatter() or have type MATSCATTER

    Output Parameter:
.   scatter - the scatter context

    Level: intermediate

.seealso: MatCreateScatter(), MatScatterSetVecScatter(), MATSCATTER
@*/
PetscErrorCode  MatScatterGetVecScatter(Mat mat,VecScatter *scatter)
{
  Mat_Scatter *mscatter;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidPointer(scatter,2);
  mscatter = (Mat_Scatter*)mat->data;
  *scatter = mscatter->scatter;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_Scatter(Mat mat)
{
  PetscErrorCode ierr;
  Mat_Scatter    *scatter = (Mat_Scatter*)mat->data;

  PetscFunctionBegin;
  ierr = VecScatterDestroy(&scatter->scatter);CHKERRQ(ierr);
  ierr = PetscFree(mat->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_Scatter(Mat A,Vec x,Vec y)
{
  Mat_Scatter    *scatter = (Mat_Scatter*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckFalse(!scatter->scatter,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"Need to first call MatScatterSetScatter()");
  ierr = VecZeroEntries(y);CHKERRQ(ierr);
  ierr = VecScatterBegin(scatter->scatter,x,y,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(scatter->scatter,x,y,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAdd_Scatter(Mat A,Vec x,Vec y,Vec z)
{
  Mat_Scatter    *scatter = (Mat_Scatter*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckFalse(!scatter->scatter,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"Need to first call MatScatterSetScatter()");
  if (z != y) {ierr = VecCopy(y,z);CHKERRQ(ierr);}
  ierr = VecScatterBegin(scatter->scatter,x,z,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(scatter->scatter,x,z,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTranspose_Scatter(Mat A,Vec x,Vec y)
{
  Mat_Scatter    *scatter = (Mat_Scatter*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckFalse(!scatter->scatter,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"Need to first call MatScatterSetScatter()");
  ierr = VecZeroEntries(y);CHKERRQ(ierr);
  ierr = VecScatterBegin(scatter->scatter,x,y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(scatter->scatter,x,y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTransposeAdd_Scatter(Mat A,Vec x,Vec y,Vec z)
{
  Mat_Scatter    *scatter = (Mat_Scatter*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckFalse(!scatter->scatter,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"Need to first call MatScatterSetScatter()");
  if (z != y) {ierr = VecCopy(y,z);CHKERRQ(ierr);}
  ierr = VecScatterBegin(scatter->scatter,x,z,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(scatter->scatter,x,z,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static struct _MatOps MatOps_Values = {NULL,
                                       NULL,
                                       NULL,
                                       MatMult_Scatter,
                               /*  4*/ MatMultAdd_Scatter,
                                       MatMultTranspose_Scatter,
                                       MatMultTransposeAdd_Scatter,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 10*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 15*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 20*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 24*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 29*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 34*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 39*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 44*/ NULL,
                                       NULL,
                                       MatShift_Basic,
                                       NULL,
                                       NULL,
                               /* 49*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 54*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 59*/ NULL,
                                       MatDestroy_Scatter,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 64*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 69*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 74*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 79*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 84*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 89*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 94*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                /*99*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*104*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*109*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*114*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*119*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*124*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*129*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*134*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*139*/ NULL,
                                       NULL,
                                       NULL
};

/*MC
   MATSCATTER - MATSCATTER = "scatter" - A matrix type that simply applies a VecScatterBegin/End()

  Level: advanced

.seealso: MatCreateScatter(), MatScatterSetVecScatter(), MatScatterGetVecScatter()

M*/

PETSC_EXTERN PetscErrorCode MatCreate_Scatter(Mat A)
{
  Mat_Scatter    *b;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMemcpy(A->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);
  ierr = PetscNewLog(A,&b);CHKERRQ(ierr);

  A->data = (void*)b;

  ierr = PetscLayoutSetUp(A->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->cmap);CHKERRQ(ierr);

  A->assembled    = PETSC_TRUE;
  A->preallocated = PETSC_FALSE;

  ierr = PetscObjectChangeTypeName((PetscObject)A,MATSCATTER);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <petsc/private/sfimpl.h>
/*@C
   MatCreateScatter - Creates a new matrix based on a VecScatter

  Collective

   Input Parameters:
+  comm - MPI communicator
-  scatter - a VecScatterContext

   Output Parameter:
.  A - the matrix

   Level: intermediate

   PETSc requires that matrices and vectors being used for certain
   operations are partitioned accordingly.  For example, when
   creating a scatter matrix, A, that supports parallel matrix-vector
   products using MatMult(A,x,y) the user should set the number
   of local matrix rows to be the number of local elements of the
   corresponding result vector, y. Note that this is information is
   required for use of the matrix interface routines, even though
   the scatter matrix may not actually be physically partitioned.

  Developer Notes: This directly accesses information inside the VecScatter associated with the matrix-vector product
   for this matrix. This is not desirable..

.seealso: MatScatterSetVecScatter(), MatScatterGetVecScatter(), MATSCATTER
@*/
PetscErrorCode  MatCreateScatter(MPI_Comm comm,VecScatter scatter,Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,scatter->vscat.to_n,scatter->vscat.from_n,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATSCATTER);CHKERRQ(ierr);
  ierr = MatScatterSetVecScatter(*A,scatter);CHKERRQ(ierr);
  ierr = MatSetUp(*A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    MatScatterSetVecScatter - sets that scatter that the matrix is to apply as its linear operator

   Collective on Mat

    Input Parameters:
+   mat - the scatter matrix
-   scatter - the scatter context create with VecScatterCreate()

   Level: advanced

.seealso: MatCreateScatter(), MATSCATTER
@*/
PetscErrorCode  MatScatterSetVecScatter(Mat mat,VecScatter scatter)
{
  Mat_Scatter    *mscatter = (Mat_Scatter*)mat->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidHeaderSpecific(scatter,PETSCSF_CLASSID,2);
  PetscCheckSameComm((PetscObject)scatter,2,(PetscObject)mat,1);
  PetscCheckFalse(mat->rmap->n != scatter->vscat.to_n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Number of local rows in matrix %" PetscInt_FMT " not equal local scatter size %" PetscInt_FMT,mat->rmap->n,scatter->vscat.to_n);
  PetscCheckFalse(mat->cmap->n != scatter->vscat.from_n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Number of local columns in matrix %" PetscInt_FMT " not equal local scatter size %" PetscInt_FMT,mat->cmap->n,scatter->vscat.from_n);

  ierr = PetscObjectReference((PetscObject)scatter);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&mscatter->scatter);CHKERRQ(ierr);

  mscatter->scatter = scatter;
  PetscFunctionReturn(0);
}

