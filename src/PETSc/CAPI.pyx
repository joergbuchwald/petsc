#---------------------------------------------------------------------

cdef extern from "petsc.h":
    int PyPetscINCREF"PetscObjectReference"(PetscObject)

cdef int addref(void *o) except -1:
    if o == NULL: return 0
    CHKERR( PyPetscINCREF(<PetscObject>o) )
    return 0

#---------------------------------------------------------------------

# -- Error --

cdef public api int PyPetscError_Set(int ierr):
    return SETERR(ierr)

# -- Comm --

cdef public api object PyPetscComm_New(MPI_Comm arg):
    cdef Comm retv = Comm()
    retv.comm = arg
    return retv

cdef public api MPI_Comm PyPetscComm_Get(object arg) except *:
    cdef MPI_Comm retv = MPI_COMM_NULL
    cdef Comm ob = <Comm?> arg
    retv = ob.comm
    return retv

cdef public api MPI_Comm* PyPetscComm_GetPtr(object arg) except NULL:
    cdef MPI_Comm *retv = NULL
    cdef Comm ob = <Comm?> arg
    retv = &ob.comm
    return retv

# -- Object --

cdef public api PetscObject PyPetscObject_Get(object arg) except ? NULL:
    cdef PetscObject retv = NULL
    cdef Object ob = <Object?> arg
    retv = ob.obj[0]
    return retv

cdef public api PetscObject* PyPetscObject_GetPtr(object arg) except NULL:
    cdef PetscObject *retv = NULL
    cdef Object ob = <Object?> arg
    retv = ob.obj
    return retv

# -- Viewer --

cdef public api object PyPetscViewer_New(PetscViewer arg):
    cdef Viewer retv = Viewer()
    addref(arg); retv.vwr = arg
    return retv

cdef public api PetscViewer PyPetscViewer_Get(object arg) except ? NULL:
    cdef PetscViewer retv = NULL
    cdef Viewer ob = <Viewer?> arg
    retv = ob.vwr
    return retv

# -- IS --

cdef public api object PyPetscIS_New(PetscIS arg):
    cdef IS retv = IS()
    addref(arg); retv.iset = arg
    return retv

cdef public api PetscIS PyPetscIS_Get(object arg) except? NULL:
    cdef PetscIS retv = NULL
    cdef IS ob = <IS?> arg
    retv = ob.iset
    return retv

# -- LGMap --

cdef public api object PyPetscLGMap_New(PetscLGMap arg):
    cdef LGMap retv = LGMap()
    addref(arg); retv.lgm = arg
    return retv

cdef public api PetscLGMap PyPetscLGMap_Get(object arg) except ? NULL:
    cdef PetscLGMap retv = NULL
    cdef LGMap ob = <LGMap?> arg
    retv = ob.lgm
    return retv

# -- Vec --

cdef public api object PyPetscVec_New(PetscVec arg):
    cdef Vec retv = Vec()
    addref(arg); retv.vec = arg
    return retv

cdef public api PetscVec PyPetscVec_Get(object arg) except ? NULL:
    cdef PetscVec retv = NULL
    cdef Vec ob = <Vec?> arg
    retv = ob.vec
    return retv

# -- Mat --

cdef public api object PyPetscMat_New(PetscMat arg):
    cdef Mat retv = Mat()
    addref(arg); retv.mat = arg
    return retv

cdef public api PetscMat PyPetscMat_Get(object arg) except ? NULL:
    cdef PetscMat retv = NULL
    cdef Mat ob = <Mat?> arg
    retv = ob.mat
    return retv

# -- PC --

cdef public api object PyPetscPC_New(PetscPC arg):
    cdef PC retv = PC()
    addref(arg); retv.pc = arg
    return retv

cdef public api PetscPC PyPetscPC_Get(object arg) except ? NULL:
    cdef PetscPC retv = NULL
    cdef PC ob = <PC?> arg
    retv = ob.pc
    return retv

# -- KSP --

cdef public api object PyPetscKSP_New(PetscKSP arg):
    cdef KSP retv = KSP()
    addref(arg); retv.ksp = arg
    return retv

cdef public api PetscKSP PyPetscKSP_Get(object arg) except ? NULL:
    cdef PetscKSP retv = NULL
    cdef KSP ob = <KSP?> arg
    retv = ob.ksp
    return retv

# -- SNES --

cdef public api object PyPetscSNES_New(PetscSNES arg):
    cdef SNES retv = SNES()
    addref(arg); retv.snes = arg
    return retv

cdef public api PetscSNES PyPetscSNES_Get(object arg) except ? NULL:
    cdef PetscSNES retv = NULL
    cdef SNES ob = <SNES?> arg
    retv = ob.snes
    return retv

# -- TS --

cdef public api object PyPetscTS_New(PetscTS arg):
    cdef TS retv = TS()
    addref(arg); retv.ts = arg
    return retv

cdef public api PetscTS PyPetscTS_Get(object arg) except ? NULL:
    cdef PetscTS retv = NULL
    cdef TS ob = <TS?> arg
    retv = ob.ts
    return retv

#---------------------------------------------------------------------
