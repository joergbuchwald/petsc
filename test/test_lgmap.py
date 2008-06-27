from petsc4py import PETSc
import unittest

# --------------------------------------------------------------------

class TestLGMapBase(object):

    def _mk_idx(self, comm):
        comm_size = comm.getSize()
        comm_rank = comm.getRank()
        lsize = 10
        first = lsize * comm_rank
        last  = first + lsize
        if comm_rank > 0:
            first -= 1
        if comm_rank < (comm_size-1):
            last += 1
        return list(range(first, last))

    def tearDown(self):
        self.lgmap = None

    def testGetSize(self):
        size = self.lgmap.getSize()
        self.assertTrue(size >=0)
       
    def testGetInfo(self):
        info = self.lgmap.getInfo()
        self.assertEqual(type(info), dict)
        if self.lgmap.getComm().getSize() == 1:
            self.assertEqual(info, {})
        else:
            self.assertTrue(len(info) > 1)
            self.assertTrue(len(info) < 4)

    def testApply(self):
        idxin  = list(range(self.lgmap.getSize()))
        idxout = self.lgmap.apply(idxin)
        self.lgmap.apply(idxin, idxout)
        invmap = self.lgmap.applyInverse(idxout)
        

    def testApplyIS(self):
        is_in  = PETSc.IS().createStride(self.lgmap.getSize())
        is_out = self.lgmap.apply(is_in)
        
    def testProperties(self):
        for prop in ('size', 'info'):
            self.assertTrue(hasattr(self.lgmap, prop))

# --------------------------------------------------------------------

class TestLGMap(TestLGMapBase, unittest.TestCase):

    def setUp(self):
        self.idx   = self._mk_idx(PETSc.COMM_WORLD)
        self.lgmap = PETSc.LGMap().create(self.idx)

class TestLGMapIS(TestLGMapBase, unittest.TestCase):

    def setUp(self):
        self.idx   = self._mk_idx(PETSc.COMM_WORLD)
        self.iset  = PETSc.IS().createGeneral(self.idx)
        self.lgmap = PETSc.LGMap().create(self.iset)

    def testSameComm(self):
        comm1 = self.lgmap.getComm()
        comm2 = self.iset.getComm()
        self.assertEqual(comm1, comm2)

# --------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
