#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os
import PETSc.package
import md5

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.mpi           = self.framework.require('PETSc.packages.MPI',self)
    self.blasLapack    = self.framework.require('PETSc.packages.BlasLapack',self)
    self.download      = ['bk://petsc.bkbits.net/blacs-dev']
    self.deps          = [self.mpi,self.blasLapack]
    self.functions     = ['blacs_pinfo']
    self.includes      = []
    self.libdir        = ''
    return

  def getChecksum(self,source, chunkSize = 1024*1024):  #???
    '''Return the md5 checksum for a given file, which may also be specified by its filename
       - The chunkSize argument specifies the size of blocks read from the file'''
    if isinstance(source, file):
      f = source
    else:
      f = file(source)
    m = md5.new()
    size = chunkSize
    buf  = f.read(size)
    while buf:
      m.update(buf)
      buf = f.read(size)
    f.close()
    return m.hexdigest()

  def generateLibList(self,dir):
    alllibs = []
    alllibs.append(os.path.join(dir,'libblacs.a'))
    return alllibs
          
  def Install(self):
    # Get the BLACS directories
    blacsDir   = self.getDir()
    installDir = os.path.join(blacsDir, self.arch.arch)

    # Configure and build BLACS
    g = open(os.path.join(blacsDir,'Bmake.Inc'),'w')
    g.write('SHELL = /bin/sh\n')
    g.write('COMMLIB = MPI\n')
    g.write('SENDIS = -DSndIsLocBlk\n')
    g.write('WHATMPI = -DUseF77Mpi\n')
    g.write('DEBUGLVL = -DBlacsDebugLvl=1\n')
    g.write('BLACSdir = '+blacsDir+'\n')
    g.write('BLACSLIB = '+os.path.join(installDir,'libblacs.a')+'\n')
    g.write('MPIINCdir='+self.mpi.include[0]+'\n')
    g.write('MPILIB='+self.libraries.toString(self.mpi.lib)+'\n')
    g.write('SYSINC = -I$(MPIINCdir)\n')
    g.write('BTLIBS = $(BLACSLIB)  $(MPILIB) \n')
    if self.compilers.fortranManglingDoubleUnderscore:
      blah = 'f77IsF2C'
    elif self.compilers.fortranMangling == 'underscore':
      blah = 'Add_'
    elif self.compilers.fortranMangling == 'capitalize':
      blah = 'UpCase'
    else:
      blah = 'NoChange'
    g.write('INTFACE=-D'+blah+'\n')
    g.write('DEFS1 = -DSYSINC $(SYSINC) $(INTFACE) $(DEFBSTOP) $(DEFCOMBTOP) $(DEBUGLVL)\n')
    g.write('BLACSDEFS = $(DEFS1) $(SENDIS) $(BUFF) $(TRANSCOMM) $(WHATMPI) $(SYSERRORS)\n')
    self.setcompilers.pushLanguage('FC')  
    g.write('F77 ='+self.setcompilers.getCompiler()+'\n')
    g.write('F77FLAGS ='+self.setcompilers.getCompilerFlags()+'\n')
    g.write('F77LOADER ='+self.setcompilers.getLinker()+'\n')      
    g.write('F77LOADFLAGS ='+self.setcompilers.getLinkerFlags()+'\n')
    self.setcompilers.popLanguage()     
    self.setcompilers.pushLanguage('C')
    g.write('CC ='+self.setcompilers.getCompiler()+'\n')
    g.write('CCFLAGS ='+self.setcompilers.getCompilerFlags()+'\n')      
    g.write('CCLOADER ='+self.setcompilers.getLinker()+'\n')
    g.write('CCLOADFLAGS ='+self.setcompilers.getLinkerFlags()+'\n')
    self.setcompilers.popLanguage()
    g.write('ARCH ='+self.setcompilers.AR+'\n')
    g.write('ARCHFLAGS ='+self.setcompilers.AR_FLAGS+'\n')    
    g.write('RANLIB ='+self.setcompilers.RANLIB+'\n')    
    g.close()
    if not os.path.isdir(installDir):
      os.mkdir(installDir)
    if not os.path.isfile(os.path.join(installDir,'Bmake.Inc')) or not (self.getChecksum(os.path.join(installDir,'Bmake.Inc')) == self.getChecksum(os.path.join(blacsDir,'Bmake.Inc'))):
      try:
        self.logPrint("Compiling Blacs; this may take several minutes\n", debugSection='screen')
        output  = config.base.Configure.executeShellCommand('cd '+os.path.join(blacsDir,'SRC','MPI')+';make', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on BLACS: '+str(e))
    else:
      self.framework.log.write('Do NOT need to compile BLACS downloaded libraries\n')
    if not os.path.isfile(os.path.join(installDir,'libblacs.a')):
      self.framework.log.write('Error running make on BLACS   ******(libraries not installed)*******\n')
      self.framework.log.write('********Output of running make on BLACS follows *******\n')        
      self.framework.log.write(output)
      self.framework.log.write('********End of Output of running make on BLACS *******\n')
      raise RuntimeError('Error running make on BLACS, libraries not installed')
    try:
      output  = config.base.Configure.executeShellCommand('cp -f '+os.path.join(blacsDir,'Bmake.Inc')+' '+installDir, timeout=5, log = self.
framework.log)[0]
    except RuntimeError, e:
      pass
    self.framework.actions.addArgument('blacs', 'Install', 'Installed blacs into '+installDir)
    return self.getDir()

  def checkLib(self,lib,func,mangle,otherLibs = []):
    oldLibs = self.framework.argDB['LIBS']
    found = self.libraries.check(lib,func, otherLibs = otherLibs+self.mpi.lib+self.blasLapack.lib+self.compilers.flibs,fortranMangle=mangle)
    self.framework.argDB['LIBS']=oldLibs
    if found:
      self.framework.log.write('Found function '+str(func)+' in '+str(lib)+'\n')
    return found
  
  def configureLibrary(self): #almost same as package.py/configureLibrary()!
    '''Find an installation ando check if it can work with PETSc'''
    self.framework.log.write('==================================================================================\n')
    self.framework.log.write('Checking for a functional '+self.name+'\n')
    foundLibrary = 0
    foundHeader  = 0

    # get any libraries and includes we depend on
    libs         = []
    incls        = []
    for l in self.deps:
      if hasattr(l,'dlib'):    libs  += l.dlib
      if hasattr(l,self.includedir): incls += l.include
      
    for location, lib,incl in self.generateGuesses():
      if not isinstance(lib, list): lib = [lib]
      if not isinstance(incl, list): incl = [incl]
      self.framework.log.write('Checking for library '+location+': '+str(lib)+'\n')
      #if self.executeTest(self.libraries.check,[lib,self.functions],{'otherLibs' : libs}):
      if self.executeTest(self.checkLib,[lib,self.functions,1]):     
        self.lib = lib
        self.framework.log.write('Checking for headers '+location+': '+str(incl)+'\n')
        if (not self.includes) or self.executeTest(self.libraries.checkInclude, [incl, self.includes],{'otherIncludes' : incls}):
          self.include = incl
          self.found   = 1
          self.dlib    = self.lib+libs
          self.framework.packages.append(self)
          break
    if not self.found:
      raise RuntimeError('Could not find a functional '+self.name+'\n')

if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setup()
  framework.addChild(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
