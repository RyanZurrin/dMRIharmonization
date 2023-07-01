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

import multiprocessing
from conversion import nifti_write

from util import *
from denoising import denoising
from bvalMap import remapBval
from resampling import resampling
from dti import dti
from rish import rish

SCRIPTDIR= dirname(__file__)
config = ConfigParser()
config.read(pjoin(gettempdir(),f'harm_config_{getpid()}.ini'))

N_shm = int(config['DEFAULT']['N_shm'])
N_proc = int(config['DEFAULT']['N_proc'])
denoise= int(config['DEFAULT']['denoise'])
bvalMap= float(config['DEFAULT']['bvalMap'])
resample= config['DEFAULT']['resample']
if resample=='0':
    resample = 0
debug = int(config['DEFAULT']['debug'])
force = int(config['DEFAULT']['force'])

def write_bvals(bval_file, bvals):
    with open(bval_file, 'w') as f:
        f.write(('\n').join(str(b) for b in bvals))

def read_caselist(file):

    with open(file) as f:

        imgs = []
        masks = []
        content= f.read()
        for row in content.split():
            temp= [element for element in row.split(',') if element] # handling w/space
            imgs.append(temp[0])
            masks.append(temp[1])


    return (imgs, masks)


def dti_harm(imgPath, maskPath):

    directory = dirname(imgPath)
    inPrefix = imgPath.split('.nii')[0]
    prefix = psplit(inPrefix)[-1]

    outPrefix = pjoin(directory, 'dti', prefix)
    if not isfile(f'{outPrefix}_FA.nii.gz'):
        dti(imgPath, maskPath, inPrefix, outPrefix)

    outPrefix = pjoin(directory, 'harm', prefix)
    if not isfile(f'{outPrefix}_L0.nii.gz'):
        rish(imgPath, maskPath, inPrefix, outPrefix, N_shm)
    

# convert NRRD to NIFTI on the fly
def nrrd2nifti(imgPath):

    if imgPath.endswith('.nrrd'):
        niftiImgPrefix= imgPath.split('.nrrd')[0]
    elif imgPath.endswith('.nhdr'):
        niftiImgPrefix= imgPath.split('.nhdr')[0]
    else:
        return imgPath

    nifti_write(imgPath, niftiImgPrefix)

    return f'{niftiImgPrefix}.nii.gz'


def preprocessing(imgPath, maskPath):

    # load signal attributes for pre-processing
    imgPath= nrrd2nifti(imgPath)
    lowRes = load(imgPath)
    lowResImg = lowRes.get_data().astype('float')
    lowResImgHdr = lowRes.header

    maskPath= nrrd2nifti(maskPath)
    lowRes = load(maskPath)
    lowResMask = lowRes.get_data()
    lowResMaskHdr = lowRes.header

    lowResImg = applymask(lowResImg, lowResMask)

    # pre-processing

    # modifies data only
    if denoise:
        inPrefix = imgPath.split('.nii')[0]
        outPrefix = f'{inPrefix}_denoised'

        if force or not isfile(f'{outPrefix}.nii.gz'):
            print('Denoising ', imgPath)
            lowResImg, _ = denoising(lowResImg, lowResMask)
            save_nifti(f'{outPrefix}.nii.gz', lowResImg, lowRes.affine, lowResImgHdr)
            copyfile(f'{inPrefix}.bvec', f'{outPrefix}.bvec')
            copyfile(f'{inPrefix}.bval', f'{outPrefix}.bval')

        maskPath= maskPath
        imgPath = f'{outPrefix}.nii.gz'


    # modifies data, and bvals
    if bvalMap:
        inPrefix = imgPath.split('.nii')[0]
        outPrefix = f'{inPrefix}_bmapped'

        if force or not isfile(f'{outPrefix}.nii.gz'):
            print('B value mapping ', imgPath)
            bvals, _ = read_bvals_bvecs(f'{inPrefix}.bval', None)
            lowResImg, bvals = remapBval(lowResImg, lowResMask, bvals, bvalMap)
            save_nifti(f'{outPrefix}.nii.gz', lowResImg, lowRes.affine, lowResImgHdr)
            copyfile(f'{inPrefix}.bvec', f'{outPrefix}.bvec')
            write_bvals(f'{outPrefix}.bval', bvals)

        maskPath= maskPath
        imgPath = f'{outPrefix}.nii.gz'


    try:
        sp_high = np.array([float(i) for i in resample.split('x')])
    except:
        sp_high = lowResImgHdr['pixdim'][1:4]

    # modifies data, mask, and headers
    if resample and (abs(sp_high-lowResImgHdr['pixdim'][1:4])>10e-3).any():
        inPrefix = imgPath.split('.nii')[0]
        outPrefix = f'{inPrefix}_resampled'

        if force or not isfile(f'{outPrefix}.nii.gz'):
            print('Resampling ', imgPath)
            bvals, _ = read_bvals_bvecs(f'{inPrefix}.bval', None)
            imgPath, maskPath = resampling(imgPath, maskPath, lowResImg, lowResImgHdr, lowResMask, lowResMaskHdr, sp_high, bvals)
            copyfile(f'{inPrefix}.bvec', f'{outPrefix}.bvec')
            copyfile(f'{inPrefix}.bval', f'{outPrefix}.bval')
        else:
            maskPath= maskPath.split('.nii')[0]+ '_resampled.nii.gz'

        imgPath = f'{outPrefix}.nii.gz'


    return (imgPath, maskPath)



def common_processing(caselist):

    imgs, masks = read_caselist(caselist)

    # compute dti_harm of unprocessed data
    pool = multiprocessing.Pool(N_proc)
    for imgPath,maskPath in zip(imgs,masks):
        pool.apply_async(func= dti_harm, args= (imgPath,maskPath))
    pool.close()
    pool.join()


    pool = multiprocessing.Pool(N_proc)
    res = [
        pool.apply_async(func=preprocessing, args=(imgPath, maskPath))
        for imgPath, maskPath in zip(imgs, masks)
    ]
    attributes= [r.get() for r in res]

    pool.close()
    pool.join()


    with open(f'{caselist}.modified', 'w') as f:
        for i in range(len(imgs)):
            imgs[i] = attributes[i][0]
            masks[i] = attributes[i][1]
            f.write(f'{imgs[i]},{masks[i]}\n')
    # compute dti_harm of preprocessed data
    pool = multiprocessing.Pool(N_proc)
    for imgPath,maskPath in zip(imgs,masks):
        pool.apply_async(func= dti_harm, args= (imgPath,maskPath))
    pool.close()
    pool.join()


    return (imgs, masks)

