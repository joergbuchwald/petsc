#requirespackage   'PETSC_HAVE_PARDISO'

ALL: lib

CFLAGS   = ${PARDISO_INCLUDE}
FFLAGS   =
SOURCEC	 = pardiso.c
SOURCEF	 =
SOURCEH	 =
LIBBASE	 = libpetscmat
DIRS	 =
MANSEC	 = Mat
LOCDIR	 = src/mat/impls/aij/seq/pardiso/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
