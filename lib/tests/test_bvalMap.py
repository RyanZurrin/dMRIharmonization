#!/usr/bin/env python

# ===============================================================================
# dMRIharmonization (2018) pipeline is written by-
#
# TASHRIF BILLAH
# Brigham and Women's Hospital/Harvard Medical School
# tbillah@bwh.harvard.edu, tashrifbillah@gmail.com
#
# ===============================================================================
# See details at https://github.com/pnlbwh/dMRIharmonization
# Submit issues at https://github.com/pnlbwh/dMRIharmonization/issues
# View LICENSE at https://github.com/pnlbwh/dMRIharmonization/blob/master/LICENSE
# ===============================================================================

from test_util import *
from bvalMap import remapBval

from dipy.io import read_bvals_bvecs

class TestBmap(unittest.TestCase):

    def test_bvalMap(self):
        inPath= pjoin(FILEDIR, 'connectom_prisma/connectom/A/')
        inPrefix= pjoin(inPath, 'dwi_A_connectom_st_b1200')

        lowResImgPath = f'{inPrefix}.nii.gz'
        bvalPath = f'{inPrefix}.bval'
        lowResMaskPath = f'{inPrefix}_mask.nii.gz'

        # load signal attributes for pre-processing ----------------------------------------------------------------
        imgPath = nrrd2nifti(lowResImgPath)
        dwi = load(imgPath)

        maskPath = nrrd2nifti(lowResMaskPath)
        mask = load(maskPath)

        bvals, _ = read_bvals_bvecs(bvalPath, None)

        bNew= 1000.

        print('B value mapping ', imgPath)
        dwiNew, bvalsNew= remapBval(dwi.get_fdata(), mask.get_fdata(), bvals, bNew)

        outPrefix = imgPath.split('.nii')[0] + '_bmapped'
        save_nifti(f'{outPrefix}.nii.gz', dwiNew, dwi.affine, dwi.header)
        copyfile(f'{inPrefix}.bvec', f'{outPrefix}.bvec')
        write_bvals(f'{outPrefix}.bval', bvals)

if __name__ == '__main__':
    unittest.main()