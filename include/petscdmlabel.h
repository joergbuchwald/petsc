#if !defined(PETSCDMLABEL_H)
#define PETSCDMLABEL_H
#include <petscis.h>

/*S
  DMLabel - Object which encapsulates a subset of the mesh from this DM

  Level: developer

.seealso:  DM, DMPlexCreate(), DMPlexCreateLabel()
S*/
typedef struct _p_DMLabel *DMLabel;
PETSC_EXTERN PetscErrorCode DMLabelCreate(MPI_Comm, const char [], DMLabel *);
PETSC_EXTERN PetscErrorCode DMLabelView(DMLabel, PetscViewer);
PETSC_EXTERN PetscErrorCode DMLabelReset(DMLabel);
PETSC_EXTERN PetscErrorCode DMLabelDestroy(DMLabel *);
PETSC_EXTERN PetscErrorCode DMLabelGetDefaultValue(DMLabel, PetscInt *);
PETSC_EXTERN PetscErrorCode DMLabelSetDefaultValue(DMLabel, PetscInt);
PETSC_EXTERN PetscErrorCode DMLabelDuplicate(DMLabel, DMLabel *);
PETSC_EXTERN PetscErrorCode DMLabelCompare(MPI_Comm, DMLabel, DMLabel, PetscBool *, char **message);
PETSC_EXTERN PetscErrorCode DMLabelGetValue(DMLabel, PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode DMLabelSetValue(DMLabel, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMLabelClearValue(DMLabel, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMLabelAddStratum(DMLabel, PetscInt);
PETSC_EXTERN PetscErrorCode DMLabelAddStrata(DMLabel, PetscInt, const PetscInt[]);
PETSC_EXTERN PetscErrorCode DMLabelAddStrataIS(DMLabel, IS);
PETSC_EXTERN PetscErrorCode DMLabelInsertIS(DMLabel, IS, PetscInt);
PETSC_EXTERN PetscErrorCode DMLabelGetNumValues(DMLabel, PetscInt *);
PETSC_EXTERN PetscErrorCode DMLabelGetStratumBounds(DMLabel, PetscInt, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode DMLabelGetValueIS(DMLabel, IS *);
PETSC_EXTERN PetscErrorCode DMLabelGetNonEmptyStratumValuesIS(DMLabel, IS *);
PETSC_EXTERN PetscErrorCode DMLabelGetValueIndex(DMLabel, PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode DMLabelStratumHasPoint(DMLabel, PetscInt, PetscInt, PetscBool *);
PETSC_EXTERN PetscErrorCode DMLabelHasStratum(DMLabel, PetscInt, PetscBool *);
PETSC_EXTERN PetscErrorCode DMLabelGetStratumSize(DMLabel, PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode DMLabelGetStratumIS(DMLabel, PetscInt, IS *);
PETSC_EXTERN PetscErrorCode DMLabelSetStratumIS(DMLabel, PetscInt, IS);
PETSC_EXTERN PetscErrorCode DMLabelSetStratumBounds(DMLabel, PetscInt, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMLabelClearStratum(DMLabel, PetscInt);
PETSC_EXTERN PetscErrorCode DMLabelGetStratumPointIndex(DMLabel, PetscInt, PetscInt, PetscInt *);

PETSC_EXTERN PetscErrorCode DMLabelComputeIndex(DMLabel);
PETSC_EXTERN PetscErrorCode DMLabelCreateIndex(DMLabel, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMLabelDestroyIndex(DMLabel);
PETSC_EXTERN PetscErrorCode DMLabelHasValue(DMLabel, PetscInt, PetscBool *);
PETSC_EXTERN PetscErrorCode DMLabelHasPoint(DMLabel, PetscInt, PetscBool *);
PETSC_EXTERN PetscErrorCode DMLabelGetBounds(DMLabel, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode DMLabelFilter(DMLabel, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMLabelPermute(DMLabel, IS, DMLabel *);
PETSC_EXTERN PetscErrorCode DMLabelDistribute(DMLabel, PetscSF, DMLabel *);
PETSC_EXTERN PetscErrorCode DMLabelGather(DMLabel, PetscSF, DMLabel *);
PETSC_EXTERN PetscErrorCode DMLabelConvertToSection(DMLabel, PetscSection *, IS *);

PETSC_EXTERN PetscErrorCode PetscSectionCreateGlobalSectionLabel(PetscSection, PetscSF, PetscBool, DMLabel, PetscInt, PetscSection *);

#define PETSCSECTIONSYMLABEL "label"
PETSC_EXTERN PetscErrorCode PetscSectionSymCreateLabel(MPI_Comm,DMLabel,PetscSectionSym *);
PETSC_EXTERN PetscErrorCode PetscSectionSymLabelSetLabel(PetscSectionSym,DMLabel);
PETSC_EXTERN PetscErrorCode PetscSectionSymLabelSetStratum(PetscSectionSym,PetscInt,PetscInt,PetscInt,PetscInt,PetscCopyMode,const PetscInt **,const PetscScalar **);

#endif
