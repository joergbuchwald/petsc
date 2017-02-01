import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.includes     = []
    self.needsmath    = 1
    self.functions    = ['pardisoinit'] 
    self.liblist      = [['libpardiso.so']]   # You can supply the dynamic library that you download from the PARDISO site,
                                              # e.g. ./configure --with-pardiso-lib=/path/to/libpardiso500-GNU481-X86-64.so
                                              # The corresponding pardiso.lic file should be in your home directory.
    self.requires32bitint = 1                 # PARDISO works with many data types, but our wrapper currently only 
    self.precisions = ['double']              #  supports double precision scalars and 32-bit integers.
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.openmp     = framework.require('config.packages.openmp',self)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.deps       = [self.blasLapack]
    return

  def consistencyChecks(self):
    config.package.Package.consistencyChecks(self)
    if not self.openmp.found:
      raise RuntimeError('Pardiso requires OpenMP. Try reconfiguring with --with-openmp')
    return
