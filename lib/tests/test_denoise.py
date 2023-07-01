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
from denoising import denoising



class TestDenoise(unittest.TestCase):

    def test_denoise(self):
        inPath = pjoin(FILEDIR, 'connectom_prisma/connectom/A/')
        inPrefix = pjoin(inPath, 'dwi_A_connectom_st_b1200')

        lowResImgPath = f'{inPrefix}.nii.gz'
        lowResMaskPath = f'{inPrefix}_mask.nii.gz'

        # load signal attributes for pre-processing ----------------------------------------------------------------
        imgPath = nrrd2nifti(lowResImgPath)
        dwi = load(imgPath)

        maskPath = nrrd2nifti(lowResMaskPath)
        mask = load(maskPath)

        print('Denoising ', imgPath)
        dwiNew, _= denoising(dwi.get_fdata(), mask.get_fdata())
        outPrefix = imgPath.split('.nii')[0] + '_denoised'
        save_nifti(f'{outPrefix}.nii.gz', dwiNew, dwi.affine, dwi.header)
        copyfile(f'{inPrefix}.bvec', f'{outPrefix}.bvec')
        copyfile(f'{inPrefix}.bval', f'{outPrefix}.bval')


if __name__ == '__main__':
    unittest.main()