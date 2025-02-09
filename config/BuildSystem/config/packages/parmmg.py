import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit        = '010d4df5548fc1b44e4fbe20e96fd31bc78c8c23' # jolivet/fix-compilation-3.17.0-bis feb-12-2022
    self.download         = ['git://https://github.com/prj-/ParMmg.git','https://github.com/prj-/ParMmg/archive/'+self.gitcommit+'.tar.gz']
    self.versionname      = 'PMMG_VERSION_RELEASE'
    self.includes         = ['parmmg/libparmmg.h']
    self.liblist          = [['libparmmg.a']]
    self.functions        = ['PMMG_Free_all_var']
    self.precisions       = ['double']
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.externalpackagesdir = framework.require('PETSc.options.externalpackagesdir',self)
    self.compilerFlags = framework.require('config.compilerFlags',self)
    self.mathlib       = framework.require('config.packages.mathlib',self)
    self.mpi           = framework.require('config.packages.MPI',self)
    self.ptscotch      = framework.require('config.packages.PTScotch',self)
    self.metis         = framework.require('config.packages.metis',self)
    self.mmg           = framework.require('config.packages.mmg',self)
    self.deps          = [self.mpi,self.mathlib,self.ptscotch,self.metis,self.mmg]
    return

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-DDOWNLOAD_MMG=OFF')
    args.append('-DDOWNLOAD_METIS=OFF')
    args.append('-DUSE_VTK=OFF')
    args.append('-DUSE_POINTMAP=ON')
    args.append('-DSCOTCH_DIR:STRING="'+self.ptscotch.directory+'"')
    args.append('-DMETIS_DIR:STRING="'+self.metis.directory+'"')
    args.append('-DMMG_DIR:STRING="'+self.mmg.directory+'"')
    return args
