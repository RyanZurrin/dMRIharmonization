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
from resampling import resampling


class TestResample(unittest.TestCase):

    def test_resample(self):
        inPath= pjoin(FILEDIR, 'connectom_prisma/connectom/A/')
        inPrefix= pjoin(inPath, 'dwi_A_connectom_st_b1200')

        lowResImgPath = f'{inPrefix}.nii.gz'
        bvalPath = f'{inPrefix}.bval'
        lowResMaskPath = f'{inPrefix}_mask.nii.gz'

        # load signal attributes for pre-processing ----------------------------------------------------------------
        imgPath = nrrd2nifti(lowResImgPath)
        lowRes = load(imgPath)
        lowResImg = lowRes.get_fdata().astype('float')
        lowResImgHdr = lowRes.header

        maskPath = nrrd2nifti(lowResMaskPath)
        lowRes = load(maskPath)
        lowResMask = lowRes.get_fdata()
        lowResMaskHdr = lowRes.header

        bvals, _ = read_bvals_bvecs(bvalPath, None)

        sp_high= array([1.5,1.5,1.5])

        print('Resampling ', imgPath)
        resampling(lowResImgPath, lowResMaskPath, lowResImg, lowResImgHdr, lowResMask, lowResMaskHdr, sp_high, bvals)
        outPrefix = f'{inPrefix}_resampled'
        copyfile(f'{inPrefix}.bvec', f'{outPrefix}.bvec')
        copyfile(f'{inPrefix}.bval', f'{outPrefix}.bval')


if __name__ == '__main__':
    unittest.main()


