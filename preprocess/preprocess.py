import numpy as np
import nibabel as nib
import ants
import os
from fsl.wrappers import fslmaths, bet
from tqdm import tqdm
import multiprocessing
import glob

def nii2ants(image):
    ndim = image.ndim #must be 3D
    q_form = image.get_qform()
    spacing = image.header["pixdim"][1 : ndim + 1]
    origin = np.zeros((ndim))
    origin[:3] = q_form[:3, 3]
    direction = np.diag(np.ones(ndim))
    direction[:3, :3] = q_form[:3, :3] / spacing[:3]

    image = ants.from_numpy(
        data = image.get_fdata(),
        origin = origin.tolist(),
        spacing = spacing.tolist(),
        direction = direction )
    return image

def brain(i, image, z_list):
    '''
    Brain Extraction with FSL 

    Params:
    - image: nifti object, scan to brain extract
    Output: 
    - brain_image: nifti object, extracted brain
    '''
    affine = image.affine
    header = image.header
    tmpfile = f'{i}_tmpfile.nii.gz'
    image.to_filename(tmpfile)

    # FSL calls
    mask = fslmaths(image).thr('0.0000').uthr('100.00').bin().fillh().run()
    fslmaths(image).mas(mask).run(tmpfile)
    bet(tmpfile, tmpfile, fracintensity=0.01)
    mask = fslmaths(tmpfile).bin().fillh().run()
    image = fslmaths(image).mas(mask).run()
    image = nib.Nifti1Image(image.get_fdata(), affine, header)
    os.remove(tmpfile)
    z_list.append(header.get_data_shape()[2])
    return image

def rigid(fixed, moving):

        '''
        Rigid Registration with ANTs

        Params:
                - moving: ants image, image to move when registering
                - fixed: ants image, template image to register to

        Outputs: 
                - image: registered image
                - transforms: transformation affine matrix
        '''

        kwargs = {'-n': 'nearestNeighbor'}
        tx = ants.registration(fixed, moving, type_of_transform='Rigid', mask=None, grad_step=0.2, flow_sigma=3, total_sigma=0, 
                           aff_metric='mattes', aff_sampling=64, syn_metric ='mattes',**kwargs) 
                        
        image = tx['warpedmovout']
        transforms = tx['fwdtransforms']
        return image, transforms




def preprocess_medical_image(input_image, output_filename, z_list, i):
    brain_extracted_image = brain(i, input_image, z_list)
    # antsimage = nii2ants(brain_extracted_image)
    # registered_image, _ = rigid(template_image, antsimage)
    nib.save(brain_extracted_image, output_filename)
# #%% ICH
# if __name__ == "__main__":
#     z_list = multiprocessing.Manager().list()
#     PATH = '/mnt/hdd/smchou/hematoma/data/raw/nifti/ICH'
#     OUTPUT_PATH = '/mnt/hdd/smchou/hematoma/data/preprocessed/nifti/ICH/input'
#     # template_image = ants.image_read('/mnt/hdd/smchou/hematoma/data/raw/nifti/brainct_final/43529919.nii.gz')
#     print(f'in: {PATH} out: {OUTPUT_PATH}')
#     def preprocess_wrapper(i):
#         img = nib.load(os.path.join(PATH, i + "_ct.nii.gz"))
#         output_filename = os.path.join(OUTPUT_PATH, i + "_ct.nii.gz")
#         preprocess_medical_image(img, output_filename, z_list, i) # template_image,
        
#     with multiprocessing.Pool() as pool:
#         file_list = sorted([x.split('_')[0] for x in [os.path.basename(file) for file in glob.glob(os.path.join(PATH, '*_ct.nii.gz'))]])
#         for _ in tqdm(pool.imap_unordered(preprocess_wrapper, file_list), total=len(file_list)):
#             pass
    
#     print(z_list)
# #%% brainct_final
# if __name__ == "__main__":
#     z_list = multiprocessing.Manager().list()
#     PATH = '/mnt/hdd/smchou/hematoma/data/raw/nifti/brainct_final'
#     OUTPUT_PATH = '/mnt/hdd/smchou/hematoma/data/preprocessed/nifti/final_ct/input'
#     print(f'in: {PATH} out: {OUTPUT_PATH}')

#     def preprocess_wrapper(i):
#         img = nib.load(os.path.join(PATH, i))
#         output_filename = os.path.join(OUTPUT_PATH, i)
#         preprocess_medical_image(img, output_filename, z_list, i) # template_image,
        
#     with multiprocessing.Pool() as pool:
#         file_list = sorted(os.listdir(PATH))
#         for _ in tqdm(pool.imap_unordered(preprocess_wrapper, file_list), total=len(file_list)):
#             pass
    
#     print(z_list)

# #%% ICH_brm
# if __name__ == "__main__":
#     z_list = multiprocessing.Manager().list()
#     PATH = '/mnt/hdd/smchou/hematoma/data/raw/nifti/ICH_brm/indir'
#     OUTPUT_PATH = '/mnt/hdd/smchou/hematoma/data/preprocessed/nifti/ICH_brm/input'
#     print(f'in: {PATH} out: {OUTPUT_PATH}')

#     def preprocess_wrapper(i):
#         img = nib.load(os.path.join(PATH, i))
#         output_filename = os.path.join(OUTPUT_PATH, i)
#         preprocess_medical_image(img, output_filename, z_list, i) # template_image,
        
#     with multiprocessing.Pool() as pool:
#         file_list = sorted([os.path.basename(x) for x in glob.glob(os.path.join(PATH, '*_1.nii.gz'))])
#         for _ in tqdm(pool.imap_unordered(preprocess_wrapper, file_list), total=len(file_list)):
#             pass
    
#     print(z_list)
#%% finalct_rest
if __name__ == "__main__":
    z_list = multiprocessing.Manager().list()
    PATH = '/mnt/hdd/smchou/hematoma/data/raw/temp'
    OUTPUT_PATH = '/mnt/hdd/smchou/hematoma/data/raw/temp_preprocessed'
    print(f'in: {PATH} out: {OUTPUT_PATH}')
    def preprocess_wrapper(i):
        img = nib.load(os.path.join(PATH, i))
        output_filename = os.path.join(OUTPUT_PATH, i)
        preprocess_medical_image(img, output_filename, z_list, i) # template_image,
        
    with multiprocessing.Pool() as pool:
        file_list = sorted([os.path.basename(x) for x in glob.glob(os.path.join(PATH, '*.nii.gz'))])
        for _ in tqdm(pool.imap_unordered(preprocess_wrapper, file_list), total=len(file_list)):
            pass
    
    print(z_list)