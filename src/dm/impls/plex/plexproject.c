#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/

#include <petsc/private/petscfeimpl.h>

/*@
  DMPlexGetActivePoint - Get the point on which projection is currently working

  Not collective

  Input Parameter:
. dm   - the DM

  Output Parameter:
. point - The mesh point involved in the current projection

  Level: developer

.seealso: DMPlexSetActivePoint()
@*/
PetscErrorCode DMPlexGetActivePoint(DM dm, PetscInt *point)
{
  PetscFunctionBeginHot;
  *point = ((DM_Plex *) dm->data)->activePoint;
  PetscFunctionReturn(0);
}

/*@
  DMPlexSetActivePoint - Set the point on which projection is currently working

  Not collective

  Input Parameters:
+ dm   - the DM
- point - The mesh point involved in the current projection

  Level: developer

.seealso: DMPlexGetActivePoint()
@*/
PetscErrorCode DMPlexSetActivePoint(DM dm, PetscInt point)
{
  PetscFunctionBeginHot;
  ((DM_Plex *) dm->data)->activePoint = point;
  PetscFunctionReturn(0);
}

/*
  DMProjectPoint_Func_Private - Interpolate the given function in the output basis on the given point

  Input Parameters:
+ dm     - The output DM
. ds     - The output DS
. dmIn   - The input DM
. dsIn   - The input DS
. time   - The time for this evaluation
. fegeom - The FE geometry for this point
. fvgeom - The FV geometry for this point
. isFE   - Flag indicating whether each output field has an FE discretization
. sp     - The output PetscDualSpace for each field
. funcs  - The evaluation function for each field
- ctxs   - The user context for each field

  Output Parameter:
. values - The value for each dual basis vector in the output dual space

  Level: developer

.seealso: DMProjectPoint_Field_Private()
*/
static PetscErrorCode DMProjectPoint_Func_Private(DM dm, PetscDS ds, DM dmIn, PetscDS dsIn, PetscReal time, PetscFEGeom *fegeom, PetscFVCellGeom *fvgeom, PetscBool isFE[], PetscDualSpace sp[],
                                                  PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *), void **ctxs,
                                                  PetscScalar values[])
{
  PetscInt       coordDim, Nf, *Nc, f, spDim, d, v, tp;
  PetscBool      isAffine, isCohesive, transform;
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  ierr = DMGetCoordinateDim(dmIn, &coordDim);CHKERRQ(ierr);
  ierr = DMHasBasisTransform(dmIn, &transform);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(ds, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetComponents(ds, &Nc);CHKERRQ(ierr);
  ierr = PetscDSIsCohesive(ds, &isCohesive);CHKERRQ(ierr);
  /* Get values for closure */
  isAffine = fegeom->isAffine;
  for (f = 0, v = 0, tp = 0; f < Nf; ++f) {
    void * const ctx = ctxs ? ctxs[f] : NULL;
    PetscBool    cohesive;

    if (!sp[f]) continue;
    ierr = PetscDSGetCohesive(ds, f, &cohesive);CHKERRQ(ierr);
    ierr = PetscDualSpaceGetDimension(sp[f], &spDim);CHKERRQ(ierr);
    if (funcs[f]) {
      if (isFE[f]) {
        PetscQuadrature   allPoints;
        PetscInt          q, dim, numPoints;
        const PetscReal   *points;
        PetscScalar       *pointEval;
        PetscReal         *x;
        DM                rdm;

        ierr = PetscDualSpaceGetDM(sp[f],&rdm);CHKERRQ(ierr);
        ierr = PetscDualSpaceGetAllData(sp[f], &allPoints, NULL);CHKERRQ(ierr);
        ierr = PetscQuadratureGetData(allPoints,&dim,NULL,&numPoints,&points,NULL);CHKERRQ(ierr);
        ierr = DMGetWorkArray(rdm,numPoints*Nc[f],MPIU_SCALAR,&pointEval);CHKERRQ(ierr);
        ierr = DMGetWorkArray(rdm,coordDim,MPIU_REAL,&x);CHKERRQ(ierr);
        for (q = 0; q < numPoints; q++, tp++) {
          const PetscReal *v0;

          if (isAffine) {
            const PetscReal *refpoint = &points[q*dim];
            PetscReal        injpoint[3] = {0., 0., 0.};

            if (dim != fegeom->dim) {
              if (isCohesive) {
                /* We just need to inject into the higher dimensional space assuming the last dimension is collapsed */
                for (d = 0; d < dim; ++d) injpoint[d] = refpoint[d];
                refpoint = injpoint;
              } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Reference spatial dimension %D != %D dual basis spatial dimension", fegeom->dim, dim);
            }
            CoordinatesRefToReal(coordDim, fegeom->dim, fegeom->xi, fegeom->v, fegeom->J, refpoint, x);
            v0 = x;
          } else {
            v0 = &fegeom->v[tp*coordDim];
          }
          if (transform) {ierr = DMPlexBasisTransformApplyReal_Internal(dmIn, v0, PETSC_TRUE, coordDim, v0, x, dm->transformCtx);CHKERRQ(ierr); v0 = x;}
          ierr = (*funcs[f])(coordDim, time, v0, Nc[f], &pointEval[Nc[f]*q], ctx);CHKERRQ(ierr);
        }
        /* Transform point evaluations pointEval[q,c] */
        ierr = PetscDualSpacePullback(sp[f], fegeom, numPoints, Nc[f], pointEval);CHKERRQ(ierr);
        ierr = PetscDualSpaceApplyAll(sp[f], pointEval, &values[v]);CHKERRQ(ierr);
        ierr = DMRestoreWorkArray(rdm,coordDim,MPIU_REAL,&x);CHKERRQ(ierr);
        ierr = DMRestoreWorkArray(rdm,numPoints*Nc[f],MPIU_SCALAR,&pointEval);CHKERRQ(ierr);
        v += spDim;
        if (isCohesive && !cohesive) {
          for (d = 0; d < spDim; d++, v++) values[v] = values[v - spDim];
        }
      } else {
        for (d = 0; d < spDim; ++d, ++v) {
          ierr = PetscDualSpaceApplyFVM(sp[f], d, time, fvgeom, Nc[f], funcs[f], ctx, &values[v]);CHKERRQ(ierr);
        }
      }
    } else {
      for (d = 0; d < spDim; d++, v++) values[v] = 0.;
      if (isCohesive && !cohesive) {
        for (d = 0; d < spDim; d++, v++) values[v] = 0.;
      }
    }
  }
  PetscFunctionReturn(0);
}

/*
  DMProjectPoint_Field_Private - Interpolate a function of the given field, in the input basis, using the output basis on the given point

  Input Parameters:
+ dm             - The output DM
. ds             - The output DS
. dmIn           - The input DM
. dsIn           - The input DS
. dmAux          - The auxiliary DM, which is always for the input space
. dsAux          - The auxiliary DS, which is always for the input space
. time           - The time for this evaluation
. localU         - The local solution
. localA         - The local auziliary fields
. cgeom          - The FE geometry for this point
. sp             - The output PetscDualSpace for each field
. p              - The point in the output DM
. T              - Input basis and derivatives for each field tabulated on the quadrature points
. TAux           - Auxiliary basis and derivatives for each aux field tabulated on the quadrature points
. funcs          - The evaluation function for each field
- ctxs           - The user context for each field

  Output Parameter:
. values         - The value for each dual basis vector in the output dual space

  Note: Not supported for FV

  Level: developer

.seealso: DMProjectPoint_Field_Private()
*/
static PetscErrorCode DMProjectPoint_Field_Private(DM dm, PetscDS ds, DM dmIn, DMEnclosureType encIn, PetscDS dsIn, DM dmAux, DMEnclosureType encAux, PetscDS dsAux, PetscReal time, Vec localU, Vec localA, PetscFEGeom *cgeom, PetscDualSpace sp[], PetscInt p,
                                                   PetscTabulation *T, PetscTabulation *TAux,
                                                   void (**funcs)(PetscInt, PetscInt, PetscInt,
                                                                  const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                  const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                  PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), void **ctxs,
                                                   PetscScalar values[])
{
  PetscSection       section, sectionAux = NULL;
  PetscScalar       *u, *u_t = NULL, *u_x, *a = NULL, *a_t = NULL, *a_x = NULL, *bc;
  PetscScalar       *coefficients   = NULL, *coefficientsAux   = NULL;
  PetscScalar       *coefficients_t = NULL, *coefficientsAux_t = NULL;
  const PetscScalar *constants;
  PetscReal         *x;
  PetscInt          *uOff, *uOff_x, *aOff = NULL, *aOff_x = NULL, *Nc;
  PetscFEGeom        fegeom;
  const PetscInt     dE = cgeom->dimEmbed;
  PetscInt           numConstants, Nf, NfIn, NfAux = 0, f, spDim, d, v, inp, tp = 0;
  PetscBool          isAffine, isCohesive, transform;
  PetscErrorCode     ierr;

  PetscFunctionBeginHot;
  ierr = PetscDSGetNumFields(ds, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetComponents(ds, &Nc);CHKERRQ(ierr);
  ierr = PetscDSIsCohesive(ds, &isCohesive);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(dsIn, &NfIn);CHKERRQ(ierr);
  ierr = PetscDSGetComponentOffsets(dsIn, &uOff);CHKERRQ(ierr);
  ierr = PetscDSGetComponentDerivativeOffsets(dsIn, &uOff_x);CHKERRQ(ierr);
  ierr = PetscDSGetEvaluationArrays(dsIn, &u, &bc /*&u_t*/, &u_x);CHKERRQ(ierr);
  ierr = PetscDSGetWorkspace(dsIn, &x, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscDSGetConstants(dsIn, &numConstants, &constants);CHKERRQ(ierr);
  ierr = DMHasBasisTransform(dmIn, &transform);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dmIn, &section);CHKERRQ(ierr);
  ierr = DMGetEnclosurePoint(dmIn, dm, encIn, p, &inp);CHKERRQ(ierr);
  ierr = DMPlexVecGetClosure(dmIn, section, localU, inp, NULL, &coefficients);CHKERRQ(ierr);
  if (dmAux) {
    PetscInt subp;

    ierr = DMGetEnclosurePoint(dmAux, dm, encAux, p, &subp);CHKERRQ(ierr);
    ierr = PetscDSGetNumFields(dsAux, &NfAux);CHKERRQ(ierr);
    ierr = DMGetLocalSection(dmAux, &sectionAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponentOffsets(dsAux, &aOff);CHKERRQ(ierr);
    ierr = PetscDSGetComponentDerivativeOffsets(dsAux, &aOff_x);CHKERRQ(ierr);
    ierr = PetscDSGetEvaluationArrays(dsAux, &a, NULL /*&a_t*/, &a_x);CHKERRQ(ierr);
    ierr = DMPlexVecGetClosure(dmAux, sectionAux, localA, subp, NULL, &coefficientsAux);CHKERRQ(ierr);
  }
  /* Get values for closure */
  isAffine = cgeom->isAffine;
  fegeom.dim      = cgeom->dim;
  fegeom.dimEmbed = cgeom->dimEmbed;
  if (isAffine) {
    fegeom.v    = x;
    fegeom.xi   = cgeom->xi;
    fegeom.J    = cgeom->J;
    fegeom.invJ = cgeom->invJ;
    fegeom.detJ = cgeom->detJ;
  }
  for (f = 0, v = 0; f < Nf; ++f) {
    PetscQuadrature  allPoints;
    PetscInt         q, dim, numPoints;
    const PetscReal *points;
    PetscScalar     *pointEval;
    PetscBool        cohesive;
    DM               dm;

    if (!sp[f]) continue;
    ierr = PetscDSGetCohesive(ds, f, &cohesive);CHKERRQ(ierr);
    ierr = PetscDualSpaceGetDimension(sp[f], &spDim);CHKERRQ(ierr);
    if (!funcs[f]) {
      for (d = 0; d < spDim; d++, v++) values[v] = 0.;
      if (isCohesive && !cohesive) {
        for (d = 0; d < spDim; d++, v++) values[v] = 0.;
      }
      continue;
    }
    ierr = PetscDualSpaceGetDM(sp[f],&dm);CHKERRQ(ierr);
    ierr = PetscDualSpaceGetAllData(sp[f], &allPoints, NULL);CHKERRQ(ierr);
    ierr = PetscQuadratureGetData(allPoints,&dim,NULL,&numPoints,&points,NULL);CHKERRQ(ierr);
    ierr = DMGetWorkArray(dm,numPoints*Nc[f],MPIU_SCALAR,&pointEval);CHKERRQ(ierr);
    for (q = 0; q < numPoints; ++q, ++tp) {
      if (isAffine) {
        CoordinatesRefToReal(dE, cgeom->dim, fegeom.xi, cgeom->v, fegeom.J, &points[q*dim], x);
      } else {
        fegeom.v    = &cgeom->v[tp*dE];
        fegeom.J    = &cgeom->J[tp*dE*dE];
        fegeom.invJ = &cgeom->invJ[tp*dE*dE];
        fegeom.detJ = &cgeom->detJ[tp];
      }
      ierr = PetscFEEvaluateFieldJets_Internal(dsIn, NfIn, 0, tp, T, &fegeom, coefficients, coefficients_t, u, u_x, u_t);CHKERRQ(ierr);
      if (dsAux) {ierr = PetscFEEvaluateFieldJets_Internal(dsAux, NfAux, 0, tp, TAux, &fegeom, coefficientsAux, coefficientsAux_t, a, a_x, a_t);CHKERRQ(ierr);}
      if (transform) {ierr = DMPlexBasisTransformApplyReal_Internal(dmIn, fegeom.v, PETSC_TRUE, dE, fegeom.v, fegeom.v, dm->transformCtx);CHKERRQ(ierr);}
      (*funcs[f])(dE, NfIn, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, a_t, a_x, time, fegeom.v, numConstants, constants, &pointEval[Nc[f]*q]);
    }
    ierr = PetscDualSpaceApplyAll(sp[f], pointEval, &values[v]);CHKERRQ(ierr);
    ierr = DMRestoreWorkArray(dm,numPoints*Nc[f],MPIU_SCALAR,&pointEval);CHKERRQ(ierr);
    v += spDim;
    /* TODO: For now, set both sides equal, but this should use info from other support cell */
    if (isCohesive && !cohesive) {
      for (d = 0; d < spDim; d++, v++) values[v] = values[v - spDim];
    }
  }
  ierr = DMPlexVecRestoreClosure(dmIn, section, localU, inp, NULL, &coefficients);CHKERRQ(ierr);
  if (dmAux) {ierr = DMPlexVecRestoreClosure(dmAux, sectionAux, localA, p, NULL, &coefficientsAux);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode DMProjectPoint_BdField_Private(DM dm, PetscDS ds, DM dmIn, PetscDS dsIn, DM dmAux, DMEnclosureType encAux, PetscDS dsAux, PetscReal time, Vec localU, Vec localA, PetscFEGeom *fgeom, PetscDualSpace sp[], PetscInt p,
                                                     PetscTabulation *T, PetscTabulation *TAux,
                                                     void (**funcs)(PetscInt, PetscInt, PetscInt,
                                                                    const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                    const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                    PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), void **ctxs,
                                                     PetscScalar values[])
{
  PetscSection       section, sectionAux = NULL;
  PetscScalar       *u, *u_t = NULL, *u_x, *a = NULL, *a_t = NULL, *a_x = NULL, *bc;
  PetscScalar       *coefficients   = NULL, *coefficientsAux   = NULL;
  PetscScalar       *coefficients_t = NULL, *coefficientsAux_t = NULL;
  const PetscScalar *constants;
  PetscReal         *x;
  PetscInt          *uOff, *uOff_x, *aOff = NULL, *aOff_x = NULL, *Nc;
  PetscFEGeom        fegeom, cgeom;
  const PetscInt     dE = fgeom->dimEmbed;
  PetscInt           numConstants, Nf, NfAux = 0, f, spDim, d, v, tp = 0;
  PetscBool          isAffine;
  PetscErrorCode     ierr;

  PetscFunctionBeginHot;
  PetscCheckFalse(dm != dmIn,PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Not yet upgraded to use different input DM");
  ierr = PetscDSGetNumFields(ds, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetComponents(ds, &Nc);CHKERRQ(ierr);
  ierr = PetscDSGetComponentOffsets(ds, &uOff);CHKERRQ(ierr);
  ierr = PetscDSGetComponentDerivativeOffsets(ds, &uOff_x);CHKERRQ(ierr);
  ierr = PetscDSGetEvaluationArrays(ds, &u, &bc /*&u_t*/, &u_x);CHKERRQ(ierr);
  ierr = PetscDSGetWorkspace(ds, &x, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscDSGetConstants(ds, &numConstants, &constants);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
  ierr = DMPlexVecGetClosure(dmIn, section, localU, p, NULL, &coefficients);CHKERRQ(ierr);
  if (dmAux) {
    PetscInt subp;

    ierr = DMGetEnclosurePoint(dmAux, dm, encAux, p, &subp);CHKERRQ(ierr);
    ierr = PetscDSGetNumFields(dsAux, &NfAux);CHKERRQ(ierr);
    ierr = DMGetLocalSection(dmAux, &sectionAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponentOffsets(dsAux, &aOff);CHKERRQ(ierr);
    ierr = PetscDSGetComponentDerivativeOffsets(dsAux, &aOff_x);CHKERRQ(ierr);
    ierr = PetscDSGetEvaluationArrays(dsAux, &a, NULL /*&a_t*/, &a_x);CHKERRQ(ierr);
    ierr = DMPlexVecGetClosure(dmAux, sectionAux, localA, subp, NULL, &coefficientsAux);CHKERRQ(ierr);
  }
  /* Get values for closure */
  isAffine = fgeom->isAffine;
  fegeom.n  = NULL;
  fegeom.J  = NULL;
  fegeom.v  = NULL;
  fegeom.xi = NULL;
  cgeom.dim      = fgeom->dim;
  cgeom.dimEmbed = fgeom->dimEmbed;
  if (isAffine) {
    fegeom.v    = x;
    fegeom.xi   = fgeom->xi;
    fegeom.J    = fgeom->J;
    fegeom.invJ = fgeom->invJ;
    fegeom.detJ = fgeom->detJ;
    fegeom.n    = fgeom->n;

    cgeom.J     = fgeom->suppJ[0];
    cgeom.invJ  = fgeom->suppInvJ[0];
    cgeom.detJ  = fgeom->suppDetJ[0];
  }
  for (f = 0, v = 0; f < Nf; ++f) {
    PetscQuadrature   allPoints;
    PetscInt          q, dim, numPoints;
    const PetscReal   *points;
    PetscScalar       *pointEval;
    DM                dm;

    if (!sp[f]) continue;
    ierr = PetscDualSpaceGetDimension(sp[f], &spDim);CHKERRQ(ierr);
    if (!funcs[f]) {
      for (d = 0; d < spDim; d++, v++) values[v] = 0.;
      continue;
    }
    ierr = PetscDualSpaceGetDM(sp[f],&dm);CHKERRQ(ierr);
    ierr = PetscDualSpaceGetAllData(sp[f], &allPoints, NULL);CHKERRQ(ierr);
    ierr = PetscQuadratureGetData(allPoints,&dim,NULL,&numPoints,&points,NULL);CHKERRQ(ierr);
    ierr = DMGetWorkArray(dm,numPoints*Nc[f],MPIU_SCALAR,&pointEval);CHKERRQ(ierr);
    for (q = 0; q < numPoints; ++q, ++tp) {
      if (isAffine) {
        CoordinatesRefToReal(dE, fgeom->dim, fegeom.xi, fgeom->v, fegeom.J, &points[q*dim], x);
      } else {
        fegeom.v    = &fgeom->v[tp*dE];
        fegeom.J    = &fgeom->J[tp*dE*dE];
        fegeom.invJ = &fgeom->invJ[tp*dE*dE];
        fegeom.detJ = &fgeom->detJ[tp];
        fegeom.n    = &fgeom->n[tp*dE];

        cgeom.J     = &fgeom->suppJ[0][tp*dE*dE];
        cgeom.invJ  = &fgeom->suppInvJ[0][tp*dE*dE];
        cgeom.detJ  = &fgeom->suppDetJ[0][tp];
      }
      /* TODO We should use cgeom here, instead of fegeom, however the geometry coming in through fgeom does not have the support cell geometry */
      ierr = PetscFEEvaluateFieldJets_Internal(ds, Nf, 0, tp, T, &cgeom, coefficients, coefficients_t, u, u_x, u_t);CHKERRQ(ierr);
      if (dsAux) {ierr = PetscFEEvaluateFieldJets_Internal(dsAux, NfAux, 0, tp, TAux, &cgeom, coefficientsAux, coefficientsAux_t, a, a_x, a_t);CHKERRQ(ierr);}
      (*funcs[f])(dE, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, a_t, a_x, time, fegeom.v, fegeom.n, numConstants, constants, &pointEval[Nc[f]*q]);
    }
    ierr = PetscDualSpaceApplyAll(sp[f], pointEval, &values[v]);CHKERRQ(ierr);
    ierr = DMRestoreWorkArray(dm,numPoints*Nc[f],MPIU_SCALAR,&pointEval);CHKERRQ(ierr);
    v += spDim;
  }
  ierr = DMPlexVecRestoreClosure(dmIn, section, localU, p, NULL, &coefficients);CHKERRQ(ierr);
  if (dmAux) {ierr = DMPlexVecRestoreClosure(dmAux, sectionAux, localA, p, NULL, &coefficientsAux);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode DMProjectPoint_Private(DM dm, PetscDS ds, DM dmIn, DMEnclosureType encIn, PetscDS dsIn, DM dmAux, DMEnclosureType encAux, PetscDS dsAux, PetscFEGeom *fegeom, PetscInt effectiveHeight, PetscReal time, Vec localU, Vec localA, PetscBool hasFE, PetscBool hasFV, PetscBool isFE[],
                                             PetscDualSpace sp[], PetscInt p, PetscTabulation *T, PetscTabulation *TAux,
                                             DMBoundaryConditionType type, void (**funcs)(void), void **ctxs, PetscBool fieldActive[], PetscScalar values[])
{
  PetscFVCellGeom fvgeom;
  PetscInt        dim, dimEmbed;
  PetscErrorCode  ierr;

  PetscFunctionBeginHot;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &dimEmbed);CHKERRQ(ierr);
  if (hasFV) {ierr = DMPlexComputeCellGeometryFVM(dm, p, &fvgeom.volume, fvgeom.centroid, NULL);CHKERRQ(ierr);}
  switch (type) {
  case DM_BC_ESSENTIAL:
  case DM_BC_NATURAL:
    ierr = DMProjectPoint_Func_Private(dm, ds, dmIn, dsIn, time, fegeom, &fvgeom, isFE, sp, (PetscErrorCode (**)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *)) funcs, ctxs, values);CHKERRQ(ierr);break;
  case DM_BC_ESSENTIAL_FIELD:
  case DM_BC_NATURAL_FIELD:
    ierr = DMProjectPoint_Field_Private(dm, ds, dmIn, encIn, dsIn, dmAux, encAux, dsAux, time, localU, localA, fegeom, sp, p, T, TAux,
                                        (void (**)(PetscInt, PetscInt, PetscInt,
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[])) funcs, ctxs, values);CHKERRQ(ierr);break;
  case DM_BC_ESSENTIAL_BD_FIELD:
    ierr = DMProjectPoint_BdField_Private(dm, ds, dmIn, dsIn, dmAux, encAux, dsAux, time, localU, localA, fegeom, sp, p, T, TAux,
                                          (void (**)(PetscInt, PetscInt, PetscInt,
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                     PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[])) funcs, ctxs, values);CHKERRQ(ierr);break;
  default: SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Unknown boundary condition type: %d", (int) type);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceGetAllPointsUnion(PetscInt Nf, PetscDualSpace *sp, PetscInt dim, void (**funcs)(void), PetscQuadrature *allPoints)
{
  PetscReal      *points;
  PetscInt       f, numPoints;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  numPoints = 0;
  for (f = 0; f < Nf; ++f) {
    if (funcs[f]) {
      PetscQuadrature fAllPoints;
      PetscInt        fNumPoints;

      ierr = PetscDualSpaceGetAllData(sp[f],&fAllPoints, NULL);CHKERRQ(ierr);
      ierr = PetscQuadratureGetData(fAllPoints, NULL, NULL, &fNumPoints, NULL, NULL);CHKERRQ(ierr);
      numPoints += fNumPoints;
    }
  }
  ierr = PetscMalloc1(dim*numPoints,&points);CHKERRQ(ierr);
  numPoints = 0;
  for (f = 0; f < Nf; ++f) {
    if (funcs[f]) {
      PetscQuadrature fAllPoints;
      PetscInt        qdim, fNumPoints, q;
      const PetscReal *fPoints;

      ierr = PetscDualSpaceGetAllData(sp[f],&fAllPoints, NULL);CHKERRQ(ierr);
      ierr = PetscQuadratureGetData(fAllPoints, &qdim, NULL, &fNumPoints, &fPoints, NULL);CHKERRQ(ierr);
      PetscCheckFalse(qdim != dim,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Spatial dimension %D for dual basis does not match input dimension %D", qdim, dim);
      for (q = 0; q < fNumPoints*dim; ++q) points[numPoints*dim+q] = fPoints[q];
      numPoints += fNumPoints;
    }
  }
  ierr = PetscQuadratureCreate(PETSC_COMM_SELF,allPoints);CHKERRQ(ierr);
  ierr = PetscQuadratureSetData(*allPoints,dim,0,numPoints,points,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMGetFirstLabeledPoint - Find first labeled point p_o in odm such that the corresponding point p in dm has the specified height. Return p and the corresponding ds.

  Input Parameters:
  dm - the DM
  odm - the enclosing DM
  label - label for DM domain, or NULL for whole domain
  numIds - the number of ids
  ids - An array of the label ids in sequence for the domain
  height - Height of target cells in DMPlex topology

  Output Parameters:
  point - the first labeled point
  ds - the ds corresponding to the first labeled point

  Level: developer
*/
PetscErrorCode DMGetFirstLabeledPoint(DM dm, DM odm, DMLabel label, PetscInt numIds, const PetscInt ids[], PetscInt height, PetscInt *point, PetscDS *ds)
{
  DM              plex;
  DMEnclosureType enc;
  PetscInt        ls = -1;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (point) *point = -1;
  if (!label) PetscFunctionReturn(0);
  ierr = DMGetEnclosureRelation(dm, odm, &enc);CHKERRQ(ierr);
  ierr = DMConvert(dm, DMPLEX, &plex);CHKERRQ(ierr);
  for (PetscInt i = 0; i < numIds; ++i) {
    IS       labelIS;
    PetscInt num_points, pStart, pEnd;
    ierr = DMLabelGetStratumIS(label, ids[i], &labelIS);CHKERRQ(ierr);
    if (!labelIS) continue; /* No points with that id on this process */
    ierr = DMPlexGetHeightStratum(plex, height, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = ISGetSize(labelIS, &num_points);CHKERRQ(ierr);
    if (num_points) {
      const PetscInt *points;
      ierr = ISGetIndices(labelIS, &points);CHKERRQ(ierr);
      for (PetscInt i=0; i<num_points; i++) {
        PetscInt point;
        ierr = DMGetEnclosurePoint(dm, odm, enc, points[i], &point);CHKERRQ(ierr);
        if (pStart <= point && point < pEnd) {
          ls = point;
          if (ds) {ierr = DMGetCellDS(dm, ls, ds);CHKERRQ(ierr);}
        }
      }
      ierr = ISRestoreIndices(labelIS, &points);CHKERRQ(ierr);
    }
    ierr = ISDestroy(&labelIS);CHKERRQ(ierr);
    if (ls >= 0) break;
  }
  if (point) *point = ls;
  ierr = DMDestroy(&plex);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  This function iterates over a manifold, and interpolates the input function/field using the basis provided by the DS in our DM

  There are several different scenarios:

  1) Volumetric mesh with volumetric auxiliary data

     Here minHeight=0 since we loop over cells.

  2) Boundary mesh with boundary auxiliary data

     Here minHeight=1 since we loop over faces. This normally happens since we hang cells off of our boundary meshes to facilitate computation.

  3) Volumetric mesh with boundary auxiliary data

     Here minHeight=1 and auxbd=PETSC_TRUE since we loop over faces and use data only supported on those faces. This is common when imposing Dirichlet boundary conditions.

  4) Volumetric input mesh with boundary output mesh

     Here we must get a subspace for the input DS

  The maxHeight is used to support enforcement of constraints in DMForest.

  If localU is given and not equal to localX, we call DMPlexInsertBoundaryValues() to complete it.

  If we are using an input field (DM_BC_ESSENTIAL_FIELD or DM_BC_NATURAL_FIELD), we need to evaluate it at all the quadrature points of the dual basis functionals.
    - We use effectiveHeight to mean the height above our incoming DS. For example, if the DS is for a submesh then the effective height is zero, whereas if the DS
      is for the volumetric mesh, but we are iterating over a surface, then the effective height is nonzero. When the effective height is nonzero, we need to extract
      dual spaces for the boundary from our input spaces.
    - After extracting all quadrature points, we tabulate the input fields and auxiliary fields on them.

  We check that the #dof(closure(p)) == #dual basis functionals(p) for a representative p in the iteration

  If we have a label, we iterate over those points. This will probably break the maxHeight functionality since we do not check the height of those points.
*/
static PetscErrorCode DMProjectLocal_Generic_Plex(DM dm, PetscReal time, Vec localU,
                                                  PetscInt Ncc, const PetscInt comps[], DMLabel label, PetscInt numIds, const PetscInt ids[],
                                                  DMBoundaryConditionType type, void (**funcs)(void), void **ctxs,
                                                  InsertMode mode, Vec localX)
{
  DM                 plex, dmIn, plexIn, dmAux = NULL, plexAux = NULL, tdm;
  DMEnclosureType    encIn, encAux;
  PetscDS            ds = NULL, dsIn = NULL, dsAux = NULL;
  Vec                localA = NULL, tv;
  IS                 fieldIS;
  PetscSection       section;
  PetscDualSpace    *sp, *cellsp, *spIn, *cellspIn;
  PetscTabulation *T = NULL, *TAux = NULL;
  PetscInt          *Nc;
  PetscInt           dim, dimEmbed, depth, htInc = 0, htIncIn = 0, htIncAux = 0, minHeight, maxHeight, h, regionNum, Nf, NfIn, NfAux = 0, NfTot, f;
  PetscBool         *isFE, hasFE = PETSC_FALSE, hasFV = PETSC_FALSE, isCohesive = PETSC_FALSE, transform;
  DMField            coordField;
  DMLabel            depthLabel;
  PetscQuadrature    allPoints = NULL;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (localU) {ierr = VecGetDM(localU, &dmIn);CHKERRQ(ierr);}
  else        {dmIn = dm;}
  ierr = DMGetAuxiliaryVec(dm, label, numIds ? ids[0] : 0, &localA);CHKERRQ(ierr);
  if (localA) {ierr = VecGetDM(localA, &dmAux);CHKERRQ(ierr);} else {dmAux = NULL;}
  ierr = DMConvert(dm, DMPLEX, &plex);CHKERRQ(ierr);
  ierr = DMConvert(dmIn, DMPLEX, &plexIn);CHKERRQ(ierr);
  ierr = DMGetEnclosureRelation(dmIn, dm, &encIn);CHKERRQ(ierr);
  ierr = DMGetEnclosureRelation(dmAux, dm, &encAux);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetVTKCellHeight(plex, &minHeight);CHKERRQ(ierr);
  ierr = DMGetBasisTransformDM_Internal(dm, &tdm);CHKERRQ(ierr);
  ierr = DMGetBasisTransformVec_Internal(dm, &tv);CHKERRQ(ierr);
  ierr = DMHasBasisTransform(dm, &transform);CHKERRQ(ierr);
  /* Auxiliary information can only be used with interpolation of field functions */
  if (dmAux) {
    ierr = DMConvert(dmAux, DMPLEX, &plexAux);CHKERRQ(ierr);
    if (type == DM_BC_ESSENTIAL_FIELD || type == DM_BC_ESSENTIAL_BD_FIELD || type == DM_BC_NATURAL_FIELD) {
      PetscCheckFalse(!localA,PETSC_COMM_SELF, PETSC_ERR_USER, "Missing localA vector");
    }
  }
  /* Determine height for iteration of all meshes */
  {
    DMPolytopeType ct, ctIn, ctAux;
    PetscInt       minHeightIn, minHeightAux, lStart, pStart, pEnd, p, pStartIn, pStartAux;
    PetscInt       dim = -1, dimIn, dimAux;

    ierr = DMPlexGetSimplexOrBoxCells(plex, minHeight, &pStart, &pEnd);CHKERRQ(ierr);
    if (pEnd > pStart) {
      ierr = DMGetFirstLabeledPoint(dm, dm, label, numIds, ids, minHeight, &lStart, NULL);CHKERRQ(ierr);
      p    = lStart < 0 ? pStart : lStart;
      ierr = DMPlexGetCellType(plex, p, &ct);CHKERRQ(ierr);
      dim  = DMPolytopeTypeGetDim(ct);
      ierr = DMPlexGetVTKCellHeight(plexIn, &minHeightIn);CHKERRQ(ierr);
      ierr = DMPlexGetSimplexOrBoxCells(plexIn, minHeightIn, &pStartIn, NULL);CHKERRQ(ierr);
      ierr = DMPlexGetCellType(plexIn, pStartIn, &ctIn);CHKERRQ(ierr);
      dimIn = DMPolytopeTypeGetDim(ctIn);
      if (dmAux) {
        ierr = DMPlexGetVTKCellHeight(plexAux, &minHeightAux);CHKERRQ(ierr);
        ierr = DMPlexGetSimplexOrBoxCells(plexAux, minHeightAux, &pStartAux, NULL);CHKERRQ(ierr);
        ierr = DMPlexGetCellType(plexAux, pStartAux, &ctAux);CHKERRQ(ierr);
        dimAux = DMPolytopeTypeGetDim(ctAux);
      } else dimAux = dim;
    }
    if (dim < 0) {
      DMLabel spmap = NULL, spmapIn = NULL, spmapAux = NULL;

      /* Fall back to determination based on being a submesh */
      ierr = DMPlexGetSubpointMap(plex,   &spmap);CHKERRQ(ierr);
      ierr = DMPlexGetSubpointMap(plexIn, &spmapIn);CHKERRQ(ierr);
      if (plexAux) {ierr = DMPlexGetSubpointMap(plexAux, &spmapAux);CHKERRQ(ierr);}
      dim    = spmap    ? 1 : 0;
      dimIn  = spmapIn  ? 1 : 0;
      dimAux = spmapAux ? 1 : 0;
    }
    {
      PetscInt dimProj = PetscMin(PetscMin(dim, dimIn), dimAux);

      PetscCheckFalse(PetscAbsInt(dimProj - dim) > 1 || PetscAbsInt(dimProj - dimIn) > 1 || PetscAbsInt(dimProj - dimAux) > 1,PETSC_COMM_SELF, PETSC_ERR_SUP, "Do not currently support differences of more than 1 in dimension");
      if (dimProj < dim) minHeight = 1;
      htInc    =  dim    - dimProj;
      htIncIn  =  dimIn  - dimProj;
      htIncAux =  dimAux - dimProj;
    }
  }
  ierr = DMPlexGetDepth(plex, &depth);CHKERRQ(ierr);
  ierr = DMPlexGetDepthLabel(plex, &depthLabel);CHKERRQ(ierr);
  ierr = DMPlexGetMaxProjectionHeight(plex, &maxHeight);CHKERRQ(ierr);
  maxHeight = PetscMax(maxHeight, minHeight);
  PetscCheck(maxHeight >= 0 && maxHeight <= dim,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Maximum projection height %D not in [0, %D)", maxHeight, dim);
  ierr = DMGetFirstLabeledPoint(dm, dm, label, numIds, ids, 0, NULL, &ds);CHKERRQ(ierr);
  if (!ds) {ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);}
  ierr = DMGetFirstLabeledPoint(dmIn, dm, label, numIds, ids, 0, NULL, &dsIn);CHKERRQ(ierr);
  if (!dsIn) {ierr = DMGetDS(dmIn, &dsIn);CHKERRQ(ierr);}
  ierr = PetscDSGetNumFields(ds, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(dsIn, &NfIn);CHKERRQ(ierr);
  ierr = DMGetNumFields(dm, &NfTot);CHKERRQ(ierr);
  ierr = DMFindRegionNum(dm, ds, &regionNum);CHKERRQ(ierr);
  ierr = DMGetRegionNumDS(dm, regionNum, NULL, &fieldIS, NULL);CHKERRQ(ierr);
  ierr = PetscDSIsCohesive(ds, &isCohesive);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &dimEmbed);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
  if (dmAux) {
    ierr = DMGetDS(dmAux, &dsAux);CHKERRQ(ierr);
    ierr = PetscDSGetNumFields(dsAux, &NfAux);CHKERRQ(ierr);
  }
  ierr = PetscDSGetComponents(ds, &Nc);CHKERRQ(ierr);
  ierr = PetscMalloc3(Nf, &isFE, Nf, &sp, NfIn, &spIn);CHKERRQ(ierr);
  if (maxHeight > 0) {ierr = PetscMalloc2(Nf, &cellsp, NfIn, &cellspIn);CHKERRQ(ierr);}
  else               {cellsp = sp; cellspIn = spIn;}
  if (localU && localU != localX) {ierr = DMPlexInsertBoundaryValues(plex, PETSC_TRUE, localU, time, NULL, NULL, NULL);CHKERRQ(ierr);}
  /* Get cell dual spaces */
  for (f = 0; f < Nf; ++f) {
    PetscDiscType disctype;

    ierr = PetscDSGetDiscType_Internal(ds, f, &disctype);CHKERRQ(ierr);
    if (disctype == PETSC_DISC_FE) {
      PetscFE fe;

      isFE[f] = PETSC_TRUE;
      hasFE   = PETSC_TRUE;
      ierr = PetscDSGetDiscretization(ds, f, (PetscObject *) &fe);CHKERRQ(ierr);
      ierr = PetscFEGetDualSpace(fe, &cellsp[f]);CHKERRQ(ierr);
    } else if (disctype == PETSC_DISC_FV) {
      PetscFV fv;

      isFE[f] = PETSC_FALSE;
      hasFV   = PETSC_TRUE;
      ierr = PetscDSGetDiscretization(ds, f, (PetscObject *) &fv);CHKERRQ(ierr);
      ierr = PetscFVGetDualSpace(fv, &cellsp[f]);CHKERRQ(ierr);
    } else {
      isFE[f]   = PETSC_FALSE;
      cellsp[f] = NULL;
    }
  }
  for (f = 0; f < NfIn; ++f) {
    PetscDiscType disctype;

    ierr = PetscDSGetDiscType_Internal(dsIn, f, &disctype);CHKERRQ(ierr);
    if (disctype == PETSC_DISC_FE) {
      PetscFE fe;

      ierr = PetscDSGetDiscretization(dsIn, f, (PetscObject *) &fe);CHKERRQ(ierr);
      ierr = PetscFEGetDualSpace(fe, &cellspIn[f]);CHKERRQ(ierr);
    } else if (disctype == PETSC_DISC_FV) {
      PetscFV fv;

      ierr = PetscDSGetDiscretization(dsIn, f, (PetscObject *) &fv);CHKERRQ(ierr);
      ierr = PetscFVGetDualSpace(fv, &cellspIn[f]);CHKERRQ(ierr);
    } else {
      cellspIn[f] = NULL;
    }
  }
  ierr = DMGetCoordinateField(dm,&coordField);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) {
    if (!htInc) {sp[f] = cellsp[f];}
    else        {ierr = PetscDualSpaceGetHeightSubspace(cellsp[f], htInc, &sp[f]);CHKERRQ(ierr);}
  }
  if (type == DM_BC_ESSENTIAL_FIELD || type == DM_BC_ESSENTIAL_BD_FIELD || type == DM_BC_NATURAL_FIELD) {
    PetscFE          fem, subfem;
    PetscDiscType    disctype;
    const PetscReal *points;
    PetscInt         numPoints;

    PetscCheckFalse(maxHeight > minHeight,PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Field projection not supported for face interpolation");
    ierr = PetscDualSpaceGetAllPointsUnion(Nf, sp, dim-htInc, funcs, &allPoints);CHKERRQ(ierr);
    ierr = PetscQuadratureGetData(allPoints, NULL, NULL, &numPoints, &points, NULL);CHKERRQ(ierr);
    ierr = PetscMalloc2(NfIn, &T, NfAux, &TAux);CHKERRQ(ierr);
    for (f = 0; f < NfIn; ++f) {
      if (!htIncIn) {spIn[f] = cellspIn[f];}
      else          {ierr = PetscDualSpaceGetHeightSubspace(cellspIn[f], htIncIn, &spIn[f]);CHKERRQ(ierr);}

      ierr = PetscDSGetDiscType_Internal(dsIn, f, &disctype);CHKERRQ(ierr);
      if (disctype != PETSC_DISC_FE) continue;
      ierr = PetscDSGetDiscretization(dsIn, f, (PetscObject *) &fem);CHKERRQ(ierr);
      if (!htIncIn) {subfem = fem;}
      else          {ierr = PetscFEGetHeightSubspace(fem, htIncIn, &subfem);CHKERRQ(ierr);}
      ierr = PetscFECreateTabulation(subfem, 1, numPoints, points, 1, &T[f]);CHKERRQ(ierr);
    }
    for (f = 0; f < NfAux; ++f) {
      ierr = PetscDSGetDiscType_Internal(dsAux, f, &disctype);CHKERRQ(ierr);
      if (disctype != PETSC_DISC_FE) continue;
      ierr = PetscDSGetDiscretization(dsAux, f, (PetscObject *) &fem);CHKERRQ(ierr);
      if (!htIncAux) {subfem = fem;}
      else           {ierr = PetscFEGetHeightSubspace(fem, htIncAux, &subfem);CHKERRQ(ierr);}
      ierr = PetscFECreateTabulation(subfem, 1, numPoints, points, 1, &TAux[f]);CHKERRQ(ierr);
    }
  }
  /* Note: We make no attempt to optimize for height. Higher height things just overwrite the lower height results. */
  for (h = minHeight; h <= maxHeight; h++) {
    PetscInt     hEff     = h - minHeight + htInc;
    PetscInt     hEffIn   = h - minHeight + htIncIn;
    PetscInt     hEffAux  = h - minHeight + htIncAux;
    PetscDS      dsEff    = ds;
    PetscDS      dsEffIn  = dsIn;
    PetscDS      dsEffAux = dsAux;
    PetscScalar *values;
    PetscBool   *fieldActive;
    PetscInt     maxDegree;
    PetscInt     pStart, pEnd, p, lStart, spDim, totDim, numValues;
    IS           heightIS;

    if (h > minHeight) {
      for (f = 0; f < Nf; ++f) {ierr = PetscDualSpaceGetHeightSubspace(cellsp[f], hEff, &sp[f]);CHKERRQ(ierr);}
    }
    ierr = DMPlexGetSimplexOrBoxCells(plex, h, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = DMGetFirstLabeledPoint(dm, dm, label, numIds, ids, h, &lStart, NULL);CHKERRQ(ierr);
    ierr = DMLabelGetStratumIS(depthLabel, depth - h, &heightIS);CHKERRQ(ierr);
    if (pEnd <= pStart) {
      ierr = ISDestroy(&heightIS);CHKERRQ(ierr);
      continue;
    }
    /* Compute totDim, the number of dofs in the closure of a point at this height */
    totDim = 0;
    for (f = 0; f < Nf; ++f) {
      PetscBool cohesive;

      if (!sp[f]) continue;
      ierr = PetscDSGetCohesive(ds, f, &cohesive);CHKERRQ(ierr);
      ierr = PetscDualSpaceGetDimension(sp[f], &spDim);CHKERRQ(ierr);
      totDim += spDim;
      if (isCohesive && !cohesive) totDim += spDim;
    }
    p    = lStart < 0 ? pStart : lStart;
    ierr = DMPlexVecGetClosure(plex, section, localX, p, &numValues, NULL);CHKERRQ(ierr);
    PetscCheckFalse(numValues != totDim,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "The output section point (%D) closure size %D != dual space dimension %D at height %D in [%D, %D]", p, numValues, totDim, h, minHeight, maxHeight);
    if (!totDim) {
      ierr = ISDestroy(&heightIS);CHKERRQ(ierr);
      continue;
    }
    if (htInc) {ierr = PetscDSGetHeightSubspace(ds, hEff, &dsEff);CHKERRQ(ierr);}
    /* Compute totDimIn, the number of dofs in the closure of a point at this height */
    if (localU) {
      PetscInt totDimIn, pIn, numValuesIn;

      totDimIn = 0;
      for (f = 0; f < NfIn; ++f) {
        PetscBool cohesive;

        if (!spIn[f]) continue;
        ierr = PetscDSGetCohesive(dsIn, f, &cohesive);CHKERRQ(ierr);
        ierr = PetscDualSpaceGetDimension(spIn[f], &spDim);CHKERRQ(ierr);
        totDimIn += spDim;
        if (isCohesive && !cohesive) totDimIn += spDim;
      }
      ierr = DMGetEnclosurePoint(dmIn, dm, encIn, lStart < 0 ? pStart : lStart, &pIn);CHKERRQ(ierr);
      ierr = DMPlexVecGetClosure(plexIn, NULL, localU, pIn, &numValuesIn, NULL);CHKERRQ(ierr);
      PetscCheckFalse(numValuesIn != totDimIn,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "The input section point (%D) closure size %D != dual space dimension %D at height %D", pIn, numValuesIn, totDimIn, htIncIn);
      if (htIncIn) {ierr = PetscDSGetHeightSubspace(dsIn, hEffIn, &dsEffIn);CHKERRQ(ierr);}
    }
    if (htIncAux) {ierr = PetscDSGetHeightSubspace(dsAux, hEffAux, &dsEffAux);CHKERRQ(ierr);}
    /* Loop over points at this height */
    ierr = DMGetWorkArray(dm, numValues, MPIU_SCALAR, &values);CHKERRQ(ierr);
    ierr = DMGetWorkArray(dm, NfTot, MPI_INT, &fieldActive);CHKERRQ(ierr);
    {
      const PetscInt *fields;

      ierr = ISGetIndices(fieldIS, &fields);CHKERRQ(ierr);
      for (f = 0; f < NfTot; ++f) {fieldActive[f] = PETSC_FALSE;}
      for (f = 0; f < Nf; ++f) {fieldActive[fields[f]] = (funcs[f] && sp[f]) ? PETSC_TRUE : PETSC_FALSE;}
      ierr = ISRestoreIndices(fieldIS, &fields);CHKERRQ(ierr);
    }
    if (label) {
      PetscInt i;

      for (i = 0; i < numIds; ++i) {
        IS              pointIS, isectIS;
        const PetscInt *points;
        PetscInt        n;
        PetscFEGeom  *fegeom = NULL, *chunkgeom = NULL;
        PetscQuadrature quad = NULL;

        ierr = DMLabelGetStratumIS(label, ids[i], &pointIS);CHKERRQ(ierr);
        if (!pointIS) continue; /* No points with that id on this process */
        ierr = ISIntersect(pointIS,heightIS,&isectIS);CHKERRQ(ierr);
        ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
        if (!isectIS) continue;
        ierr = ISGetLocalSize(isectIS, &n);CHKERRQ(ierr);
        ierr = ISGetIndices(isectIS, &points);CHKERRQ(ierr);
        ierr = DMFieldGetDegree(coordField,isectIS,NULL,&maxDegree);CHKERRQ(ierr);
        if (maxDegree <= 1) {
          ierr = DMFieldCreateDefaultQuadrature(coordField,isectIS,&quad);CHKERRQ(ierr);
        }
        if (!quad) {
          if (!h && allPoints) {
            quad = allPoints;
            allPoints = NULL;
          } else {
            ierr = PetscDualSpaceGetAllPointsUnion(Nf,sp,isCohesive ? dim-htInc-1 : dim-htInc,funcs,&quad);CHKERRQ(ierr);
          }
        }
        ierr = DMFieldCreateFEGeom(coordField, isectIS, quad, (htInc && h == minHeight) ? PETSC_TRUE : PETSC_FALSE, &fegeom);CHKERRQ(ierr);
        for (p = 0; p < n; ++p) {
          const PetscInt  point = points[p];

          ierr = PetscArrayzero(values, numValues);CHKERRQ(ierr);
          ierr = PetscFEGeomGetChunk(fegeom,p,p+1,&chunkgeom);CHKERRQ(ierr);
          ierr = DMPlexSetActivePoint(dm, point);CHKERRQ(ierr);
          ierr = DMProjectPoint_Private(dm, dsEff, plexIn, encIn, dsEffIn, plexAux, encAux, dsEffAux, chunkgeom, htInc, time, localU, localA, hasFE, hasFV, isFE, sp, point, T, TAux, type, funcs, ctxs, fieldActive, values);
          if (ierr) {
            PetscErrorCode ierr2;
            ierr2 = DMRestoreWorkArray(dm, numValues, MPIU_SCALAR, &values);CHKERRQ(ierr2);
            ierr2 = DMRestoreWorkArray(dm, Nf, MPI_INT, &fieldActive);CHKERRQ(ierr2);
            CHKERRQ(ierr);
          }
          if (transform) {ierr = DMPlexBasisTransformPoint_Internal(plex, tdm, tv, point, fieldActive, PETSC_FALSE, values);CHKERRQ(ierr);}
          ierr = DMPlexVecSetFieldClosure_Internal(plex, section, localX, fieldActive, point, Ncc, comps, label, ids[i], values, mode);CHKERRQ(ierr);
        }
        ierr = PetscFEGeomRestoreChunk(fegeom,p,p+1,&chunkgeom);CHKERRQ(ierr);
        ierr = PetscFEGeomDestroy(&fegeom);CHKERRQ(ierr);
        ierr = PetscQuadratureDestroy(&quad);CHKERRQ(ierr);
        ierr = ISRestoreIndices(isectIS, &points);CHKERRQ(ierr);
        ierr = ISDestroy(&isectIS);CHKERRQ(ierr);
      }
    } else {
      PetscFEGeom    *fegeom = NULL, *chunkgeom = NULL;
      PetscQuadrature quad = NULL;
      IS              pointIS;

      ierr = ISCreateStride(PETSC_COMM_SELF,pEnd-pStart,pStart,1,&pointIS);CHKERRQ(ierr);
      ierr = DMFieldGetDegree(coordField,pointIS,NULL,&maxDegree);CHKERRQ(ierr);
      if (maxDegree <= 1) {
        ierr = DMFieldCreateDefaultQuadrature(coordField,pointIS,&quad);CHKERRQ(ierr);
      }
      if (!quad) {
        if (!h && allPoints) {
          quad = allPoints;
          allPoints = NULL;
        } else {
          ierr = PetscDualSpaceGetAllPointsUnion(Nf, sp, dim-htInc, funcs, &quad);CHKERRQ(ierr);
        }
      }
      ierr = DMFieldCreateFEGeom(coordField, pointIS, quad, (htInc && h == minHeight) ? PETSC_TRUE : PETSC_FALSE, &fegeom);CHKERRQ(ierr);
      for (p = pStart; p < pEnd; ++p) {
        ierr = PetscArrayzero(values, numValues);CHKERRQ(ierr);
        ierr = PetscFEGeomGetChunk(fegeom,p-pStart,p-pStart+1,&chunkgeom);CHKERRQ(ierr);
        ierr = DMPlexSetActivePoint(dm, p);CHKERRQ(ierr);
        ierr = DMProjectPoint_Private(dm, dsEff, plexIn, encIn, dsEffIn, plexAux, encAux, dsEffAux, chunkgeom, htInc, time, localU, localA, hasFE, hasFV, isFE, sp, p, T, TAux, type, funcs, ctxs, fieldActive, values);
        if (ierr) {
          PetscErrorCode ierr2;
          ierr2 = DMRestoreWorkArray(dm, numValues, MPIU_SCALAR, &values);CHKERRQ(ierr2);
          ierr2 = DMRestoreWorkArray(dm, Nf, MPI_INT, &fieldActive);CHKERRQ(ierr2);
          CHKERRQ(ierr);
        }
        if (transform) {ierr = DMPlexBasisTransformPoint_Internal(plex, tdm, tv, p, fieldActive, PETSC_FALSE, values);CHKERRQ(ierr);}
        ierr = DMPlexVecSetFieldClosure_Internal(plex, section, localX, fieldActive, p, Ncc, comps, NULL, -1, values, mode);CHKERRQ(ierr);
      }
      ierr = PetscFEGeomRestoreChunk(fegeom,p-pStart,pStart-p+1,&chunkgeom);CHKERRQ(ierr);
      ierr = PetscFEGeomDestroy(&fegeom);CHKERRQ(ierr);
      ierr = PetscQuadratureDestroy(&quad);CHKERRQ(ierr);
      ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
    }
    ierr = ISDestroy(&heightIS);CHKERRQ(ierr);
    ierr = DMRestoreWorkArray(dm, numValues, MPIU_SCALAR, &values);CHKERRQ(ierr);
    ierr = DMRestoreWorkArray(dm, Nf, MPI_INT, &fieldActive);CHKERRQ(ierr);
  }
  /* Cleanup */
  if (type == DM_BC_ESSENTIAL_FIELD || type == DM_BC_ESSENTIAL_BD_FIELD || type == DM_BC_NATURAL_FIELD) {
    for (f = 0; f < NfIn;  ++f) {ierr = PetscTabulationDestroy(&T[f]);CHKERRQ(ierr);}
    for (f = 0; f < NfAux; ++f) {ierr = PetscTabulationDestroy(&TAux[f]);CHKERRQ(ierr);}
    ierr = PetscFree2(T, TAux);CHKERRQ(ierr);
  }
  ierr = PetscQuadratureDestroy(&allPoints);CHKERRQ(ierr);
  ierr = PetscFree3(isFE, sp, spIn);CHKERRQ(ierr);
  if (maxHeight > 0) {ierr = PetscFree2(cellsp, cellspIn);CHKERRQ(ierr);}
  ierr = DMDestroy(&plex);CHKERRQ(ierr);
  ierr = DMDestroy(&plexIn);CHKERRQ(ierr);
  if (dmAux) {ierr = DMDestroy(&plexAux);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode DMProjectFunctionLocal_Plex(DM dm, PetscReal time, PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *), void **ctxs, InsertMode mode, Vec localX)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMProjectLocal_Generic_Plex(dm, time, NULL, 0, NULL, NULL, 0, NULL, DM_BC_ESSENTIAL, (void (**)(void)) funcs, ctxs, mode, localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMProjectFunctionLabelLocal_Plex(DM dm, PetscReal time, DMLabel label, PetscInt numIds, const PetscInt ids[], PetscInt Ncc, const PetscInt comps[], PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *), void **ctxs, InsertMode mode, Vec localX)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMProjectLocal_Generic_Plex(dm, time, NULL, Ncc, comps, label, numIds, ids, DM_BC_ESSENTIAL, (void (**)(void)) funcs, ctxs, mode, localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMProjectFieldLocal_Plex(DM dm, PetscReal time, Vec localU,
                                        void (**funcs)(PetscInt, PetscInt, PetscInt,
                                                       const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                       const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                       PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                        InsertMode mode, Vec localX)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMProjectLocal_Generic_Plex(dm, time, localU, 0, NULL, NULL, 0, NULL, DM_BC_ESSENTIAL_FIELD, (void (**)(void)) funcs, NULL, mode, localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMProjectFieldLabelLocal_Plex(DM dm, PetscReal time, DMLabel label, PetscInt numIds, const PetscInt ids[], PetscInt Ncc, const PetscInt comps[], Vec localU,
                                             void (**funcs)(PetscInt, PetscInt, PetscInt,
                                                            const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                            const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                            PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                             InsertMode mode, Vec localX)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMProjectLocal_Generic_Plex(dm, time, localU, Ncc, comps, label, numIds, ids, DM_BC_ESSENTIAL_FIELD, (void (**)(void)) funcs, NULL, mode, localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMProjectBdFieldLabelLocal_Plex(DM dm, PetscReal time, DMLabel label, PetscInt numIds, const PetscInt ids[], PetscInt Ncc, const PetscInt comps[], Vec localU,
                                               void (**funcs)(PetscInt, PetscInt, PetscInt,
                                                              const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                              const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                              PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                               InsertMode mode, Vec localX)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMProjectLocal_Generic_Plex(dm, time, localU, Ncc, comps, label, numIds, ids, DM_BC_ESSENTIAL_BD_FIELD, (void (**)(void)) funcs, NULL, mode, localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
