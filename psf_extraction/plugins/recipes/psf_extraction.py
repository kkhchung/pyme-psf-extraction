# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 00:38:47 2019

@author: kkc29
"""

from PYME.recipes.base import ModuleBase, register_module, Filter
from PYME.recipes.traits import Input, Output, Float, Enum, CStr, Bool, Int, List, File

import numpy as np
from scipy import ndimage, optimize, signal, interpolate, stats
from PYME.IO.image import ImageStack
from PYME.IO.dataWrap import ListWrap
from PYME.IO import MetaDataHandler

from PYME.recipes import processing
import time

from skimage import feature
from matplotlib import pyplot

@register_module('CombineBeadStacks')
class CombineBeadStacks(ModuleBase):
    """
        Combine multiply bead stacks in the 4th dimension.
        X, Y, Z must be identical.
    """
    
    inputName = Input('dummy')
    
    files = List(File, ['', ''], 2)
    cache = File()
    
    outputName = Output('bead_images')
    
    def execute(self, namespace):
        
        ims = ImageStack(filename=self.files[0])
        dims = np.asarray(ims.data.shape, dtype=np.long)
        dims[3] = 0
        dtype_ = ims.data[:,0,0,0].dtype
        mdh = ims.mdh
        del ims
        
        for fil in self.files:
            ims = ImageStack(filename=fil)
            dims[3] += ims.data.shape[3]
            del ims
        
        if self.cache != '':
            raw_data = np.memmap(self.cache, dtype=dtype_, mode='w+', shape=tuple(dims))
        else:
            raw_data = np.zeros(shape=tuple(dims), dtype=dtype_)
        
        counter = 0
        for fil in self.files:
            ims = ImageStack(filename=fil)
            c_len = ims.data.shape[3]
            data = ims.data[:,:,:,:]
            data.shape += (1,) * (4 - data.ndim)
            raw_data[:,:,:,counter:counter+c_len] = data
            counter += c_len
            del ims
            
        new_mdh = None
        try:
            new_mdh = MetaDataHandler.NestedClassMDHandler(mdh)
            new_mdh["PSFExtraction.SourceFilenames"] = self.files
        except Exception as e:
            print(e)
            
        namespace[self.outputName] = ImageStack(data=raw_data, mdh=new_mdh)

@register_module('Cleanup')
class Cleanup(ModuleBase):
    """
        Currently includes some steps for data collected from Leica SMS (Kevin).
        
        * Removed hot pixel
        * Subtract AD offset
    """
    
    inputName = Input('input')
    outputName = Output('cleaned_image')
    
    def execute(self, namespace):
        ims = namespace[self.inputName]
        mdh = ims.mdh
        
        img = ims.data[:,:,:,:]
        img[img==2**16-1] = 0
              
        np.clip(img, mdh.Camera.ADOffset, None, img)
        img -= mdh.Camera.ADOffset
        
        new_mdh = None
        try:
            new_mdh = MetaDataHandler.NestedClassMDHandler(mdh)
            print "here"
            if not 'voxelsize.z' in new_mdh.keys() or np.allclose(new_mdh['voxelsize.z'], 0):
                print 'there'
                new_mdh['voxelsize.z'] = new_mdh['StackSettings.StepSize']
            if not "PSFExtraction.SourceFilenames" in new_mdh.keys():
                new_mdh["PSFExtraction.SourceFilenames"] = ims.filename
        except Exception as e:
            print(e)
            
        namespace[self.outputName] = ImageStack(img, new_mdh)
        

#@register_module('Binning')
#class Binning(ModuleBase):
#    """
#    """
#    
#    inputName = Input('input')
#    outputName = Output('cleaned_image')
#    
#    def execute(self, namespace):
#        ims = namespace[self.inputName]
#        
#        namespace[self.outputName] = None
        
@register_module('DetectPSF')
class DetectPSF(ModuleBase):
    """
        Detect PSF based on diff of gaussian
        Image dims in X, Y, Z, C where C are processed independently.
        Returns list of (X, Y, Z) per C
    """
    
    inputName = Input('input')
    
    min_sigma = Float(1.0)
    max_sigma = Float(3.0)
    sigma_ratio = Float(1.6)
    percent_threshold = Float(0.1)
    overlap = Float(0.5)
    exclude_border = Int(50)
    ignore_z = Bool(True)
    
    output_pos = Output('psf_pos')
#    output_img = Output('output')
    
    def execute(self, namespace):
        ims = namespace[self.inputName]
        
        pixel_size = ims.mdh['voxelsize.x']
        
        pos = list()
        counts = ims.data.shape[3]
        for c in np.arange(counts):
            mean_project = ims.data[:,:,:,c].mean(2).squeeze()
            mean_project[mean_project==2**16-1] = 200
            
            mean_project -= mean_project.min()
            mean_project /= mean_project.max()
            
            # if skimage is new enough to support exclude_border
            #blobs = feature.blob_dog(mean_project, self.min_sigma / pixel_size, self.max_sigma / pixel_size, overlap=self.overlap, threshold=self.percent_threshold*mean_project.max(), exclude_border=self.exclude_border)
            
            #otherwise:
            blobs = feature.blob_dog(mean_project, self.min_sigma / pixel_size, self.max_sigma / pixel_size, overlap=self.overlap, threshold=self.percent_threshold*mean_project.max())

            edge_mask = (blobs[:, 0] > self.exclude_border) & (blobs[:, 0] < mean_project.shape[0] - self.exclude_border)
            edge_mask &= (blobs[:, 1] > self.exclude_border) & (blobs[:, 1] < mean_project.shape[1] - self.exclude_border)
            blobs = blobs[edge_mask]
            # is list of x, y, sig
            if self.ignore_z:
                blobs = np.insert(blobs, 2, ims.data.shape[2]//2, axis=1)
            else:
                raise Exception("z centering not yet implemented")
            blobs = blobs.astype(np.int)
#            print blobs
                
            pos.append(blobs)
        namespace[self.output_pos] = pos

        if True:
            try:
#                from matplotlib import pyplot
                fig, axes = pyplot.subplots(1, counts, figsize=(4*counts, 3), squeeze=False)
                for c in np.arange(counts):
                    mean_project = ims.data[:,:,:,c].mean(2).squeeze()
                    mean_project[mean_project==2**16-1] = 200
                    axes[0, c].imshow(mean_project)
                    axes[0, c].set_axis_off()
                    for x, y, z, sig in pos[c]:
                        cir = pyplot.Circle((y, x), sig, color='red', linewidth=2, fill=False)
                        axes[0, c].add_patch(cir)
            except Exception as e:
                print e
                
#            overlay_image = np.zeros(mean_project.shape, dtype=bool)
#            for x, y, sig in blobs:
#                overlay_image[x, y] = True
#            namespace[self.outputName] = ImageStack(np.stack([mean_project[:,:,None,None], overlay_image[:,:,None,None]], 3))
    
@register_module('CropPSF')
class CropPSF(ModuleBase):
    """
        Crops out PSF based on positions given.
        Built-in filter by flattened index.
        Built-in filter for removing multiple peaked data.
        Filters work on flatten X, Y images
        Stacked in the 4 dimension
    """
    
    inputName = Input('input')
    input_pos = Input('psf_pos')
    
    ignore_pos = List(Int, [])
    threshold_reject = Float(0.5)
    com_reject = Float(2.0)
    
    half_roi_x = Int(20)
    half_roi_y = Int(20)
    half_roi_z = Int(60)
    
    output_images = Output('psf_cropped')
    
    def execute(self, namespace):
        ims = namespace[self.inputName]
        psf_pos = namespace[self.input_pos]
        
        res = np.zeros((self.half_roi_x*2+1, self.half_roi_y*2+1, self.half_roi_z*2+1, sum([ar.shape[0] for ar in psf_pos])))
        
        mask = np.ones(res.shape[3], dtype=bool)
        counter = 0
        for c in np.arange(ims.data.shape[3]):
            for i in np.arange(len(psf_pos[c])):
#                print psf_pos[c][i][:3]
                x, y, z = psf_pos[c][i][:3]
                x_slice = slice(x-self.half_roi_x, x+self.half_roi_x+1)
                y_slice = slice(y-self.half_roi_y, y+self.half_roi_y+1)
                z_slice = slice(z-self.half_roi_z, z+self.half_roi_z+1)
                res[:, :, :, counter] = ims.data[x_slice, y_slice, z_slice, c].squeeze()
                
                crop_flatten = res[:, :, :, counter].mean(2)
                failed = False
                labeled_image, labeled_counts = ndimage.label(crop_flatten > crop_flatten.max() * self.threshold_reject)
                if labeled_counts > 1:
                    failed = True
                else:
                    com = np.asarray(ndimage.center_of_mass(crop_flatten, labeled_image, 1))
                    img_center = np.asarray([(s-1)*0.5 for s in labeled_image.shape])
                    dist = np.linalg.norm(com - img_center)
#                    print(com, img_center, dist)
                    if dist > self.com_reject:
                        failed = True
                    
                if failed and counter not in self.ignore_pos:
                    self.ignore_pos.append(counter)                    
                
                counter += 1

        # To do: add metadata
#        mdh['ImageType=']='PSF'        
        print "images ignore: {}".format(self.ignore_pos)
        mask[self.ignore_pos] = False
        
        new_mdh = None
        try:
            new_mdh = MetaDataHandler.NestedClassMDHandler(ims.mdh)
            new_mdh["ImageType"] = 'PSF'
            if not "PSFExtraction.SourceFilenames" in new_mdh.keys():
                new_mdh["PSFExtraction.SourceFilenames"] = ims.filename
        except Exception as e:
            print(e)
            
        namespace[self.output_images] = ImageStack(data=res[:,:,:,mask], mdh=new_mdh)
        
@register_module('AlignPSF')
class AlignPSF(ModuleBase):
    """
        Align PSF stacks by redundant cross correlation.
    """
    
    inputName = Input('psf_cropped')
    normalize_z = Bool(True)
    tukey = Float(0.50)
    rcc_tolerance = Float(5.0)
    z_crop_half_roi = Int(15)
    peak_detect = Enum(['Gaussian', 'RBF'])
    debug = Bool(False)
    output_cross_corr_images = Output('cross_cor_img')
    output_cross_corr_images_fitted = Output('cross_cor_img_fitted')
    output_images = Output('psf_aligned')
    
    def execute(self, namespace):
        self._namespace = namespace
        ims = namespace[self.inputName]
        
        # X, Y, Z, 'C'
        psf_stack = ims.data[:,:,:,:]
        
        z_slice = slice(psf_stack.shape[2]//2-self.z_crop_half_roi, psf_stack.shape[2]//2+self.z_crop_half_roi+1)
        cleaned_psf_stack = self.normalize_images(psf_stack[:,:,z_slice,:])
            
        if self.tukey > 0:
            masks = [signal.tukey(dim_len, self.tukey) for dim_len in cleaned_psf_stack.shape[:3]]            
            masks = np.product(np.meshgrid(*masks, indexing='ij'), axis=0)            
            cleaned_psf_stack *= masks[:,:,:,None]
            
        drifts = self.calculate_shifts(cleaned_psf_stack, self.rcc_tolerance * 1E3 / ims.mdh['voxelsize.x'])
#        print drifts        
        
        namespace[self.output_images] = ImageStack(self.shift_images(cleaned_psf_stack if self.debug else psf_stack, drifts), mdh=ims.mdh)
        
    def normalize_images(self, psf_stack):
        # in case it is already bg subtracted
        cleaned_psf_stack = np.clip(psf_stack, 0, None)
        
        # substact bg per stack
        cleaned_psf_stack -= cleaned_psf_stack.min(axis=(0,1,2), keepdims=True)
        
        if self.normalize_z:
            # normalize intensity per plane
            cleaned_psf_stack /= cleaned_psf_stack.max(axis=(0,1), keepdims=True) / 1.05
        else:
            # normalize intensity per psf stack
            cleaned_psf_stack /= cleaned_psf_stack.max(axis=(0,1,2), keepdims=True) / 1.05
            
        cleaned_psf_stack -= 0.05
        np.clip(cleaned_psf_stack, 0, None, cleaned_psf_stack)
        
        return cleaned_psf_stack

    def calculate_shifts(self, psf_stack, drift_tolerance):
        n_steps = psf_stack.shape[3]
        coefs_size = n_steps * (n_steps-1) / 2
        coefs = np.zeros((coefs_size, n_steps-1))
        shifts = np.zeros((coefs_size, 3))
        
        output_cross_corr_images = np.zeros((psf_stack.shape[0], psf_stack.shape[1], psf_stack.shape[2], coefs_size), dtype=np.float)
        output_cross_corr_images_fitted = np.zeros((psf_stack.shape[0], psf_stack.shape[1], psf_stack.shape[2], coefs_size), dtype=np.float)
        
        counter = 0
        for i in np.arange(0, n_steps - 1):
            for j in np.arange(i+1, n_steps):
                coefs[counter, i:j] = 1
                
                print "compare {} to {}".format(i, j)
                correlate_result = signal.correlate(psf_stack[:,:,:,i], psf_stack[:,:,:,j], mode="same")
                correlate_result -= correlate_result.min()
                correlate_result /= correlate_result.max()
        
                threshold = 0.50
                correlate_result[correlate_result<threshold] = np.nan
                
                labeled_image, labeled_counts = ndimage.label(~np.isnan(correlate_result))
#                print(labeled_counts)
                # protects against > 1 peak in the cross correlation results
                # shouldn't happen anyway, but at least avoid fitting a single to multi-modal data
                if labeled_counts > 1:
                    max_order = np.argsort(ndimage.maximum(correlate_result, labeled_image, np.arange(labeled_counts)+1))+1
                    correlate_result[labeled_image!=max_order[0]] = np.nan
                    
                
                output_cross_corr_images[:,:,:,counter] = np.nan_to_num(correlate_result)
                
                dims = list()
                for _, dim in enumerate(correlate_result.shape):
                    dims.append(np.arange(dim))
                    dims[-1] = dims[-1] - dims[-1].mean()
                
#                peaks = np.nonzero(correlate_result==np.nanmax(correlate_result))
                if self.peak_detect == "Gaussian":
                    res = optimize.least_squares(guassian_nd_error,
                                                 [1, 0, 0, 5., 0, 5., 0, 30.],
                                                 args=(dims, correlate_result))                
                    output_cross_corr_images_fitted[:,:,:,counter] = gaussian_nd(res.x, dims)
#                    print("Gaussian")
#                    print("chi2: {}".format(np.sum(np.square(res.fun))/(res.fun.shape[0]-8)))
#                    print("fitted parameters: {}".format(res.x))
                        
    #                res = optimize.least_squares(guassian_sq_nd_error,
    #                                             [1, 0, 0, 3., 0, 3., 0, 20.],
    #                                             args=(dims, correlate_result))                
    #                output_cross_corr_images_fitted[:,:,:,counter] = gaussian_sq_nd(res.x, dims)
    #                print("Gaussian 2")
    #                print("chi2: {}".format(np.sum(np.square(res.fun))/(res.fun.shape[0]-8)))
    #                print("fitted parameters: {}".format(res.x))
    #                
    #                res = optimize.least_squares(lorentzian_nd_error,
    #                                             [1, 0, 0, 2., 0, 2., 0, 10.],
    #                                             args=(dims, correlate_result))                
    #                output_cross_corr_images_fitted[:,:,:,counter] = lorentzian_nd(res.x, dims)
    #                print("lorentzian")
    #                print("chi2: {}".format(np.sum(np.square(res.fun))/(res.fun.shape[0]-8)))
    #                print("fitted parameters: {}".format(res.x))
                    
                    shifts[counter, 0] = res.x[2]
                    shifts[counter, 1] = res.x[4]
                    shifts[counter, 2] = res.x[6]
                elif self.peak_detect == "RBF":
                    rbf_interpolator = build_rbf(dims, correlate_result)
                    res = optimize.minimize(rbf_nd_error, [correlate_result.shape[0]*0.5, correlate_result.shape[1]*0.5, correlate_result.shape[2]*0.5], args=rbf_interpolator)
                    output_cross_corr_images_fitted[:,:,:,counter] = rbf_nd(rbf_interpolator, dims)
#                    print(res.x)
                    shifts[counter, :] = res.x
                else:
                    raise Exception("peak founding method not recognised")

#                print("fitted parameters: {}".format(res.x))
        
                counter += 1
                
        self._namespace[self.output_cross_corr_images] = ImageStack(output_cross_corr_images)
        self._namespace[self.output_cross_corr_images_fitted] = ImageStack(output_cross_corr_images_fitted)
        
        drifts = np.matmul(np.linalg.pinv(coefs), shifts)
        residuals = np.matmul(coefs, drifts) - shifts
        residuals_dist = np.linalg.norm(residuals, axis=1)
        
#        shift_max = self.rcc_tolerance * 1E3 / mdh['voxelsize.x']
        shift_max = drift_tolerance
        # Sort and mask residual errors
        residuals_arg = np.argsort(-residuals_dist)        
        residuals_arg = residuals_arg[residuals_dist[residuals_arg] > shift_max]

        # Remove coefs rows
        # Descending from largest residuals to small
        # Only if matrix remains full rank
        coefs_temp = np.empty_like(coefs)
        counter = 0
        for i, index in enumerate(residuals_arg):
            coefs_temp[:] = coefs
            coefs_temp[index, :] = 0
            if np.linalg.matrix_rank(coefs_temp) == coefs.shape[1]:
                coefs[:] = coefs_temp
        #                print("index {} with residual of {} removed".format(index, residuals_dist[index]))
                counter += 1
            else:
                print("Could not remove all residuals over shift_max threshold.")
                break
        print("removed {} in total".format(counter))
        drifts = np.matmul(np.linalg.pinv(coefs), shifts)
       
        drifts = np.pad(drifts, [[1,0],[0,0]], 'constant', constant_values=0)
        np.cumsum(drifts, axis=0, out=drifts)
        
        psf_stack_mean = psf_stack / psf_stack.mean(axis=(0,1,2), keepdims=True)
        psf_stack_mean = psf_stack_mean.mean(axis=3)
        psf_stack_mean *= psf_stack_mean > psf_stack_mean.max() * 0.5        
        center_offset = ndimage.center_of_mass(psf_stack_mean) - np.asarray(psf_stack_mean.shape)*0.5
#        print(center_offset)
        
#        print drifts.shape
#        print stats.trim_mean(drifts, 0.25, axis=0)
#        drifts = drifts - stats.trim_mean(drifts, 0.25, axis=0)
        
        drifts = drifts - center_offset
                
        if True:
            try:
#                from matplotlib import pyplot
                fig, axes = pyplot.subplots(1, 2, figsize=(6,3))
#                new_residuals = np.matmul(coefs, drifts) - shifts
#                new_residuals_dist = np.linalg.norm(new_residuals, axis=1)
#                # print new_residuals_dist
#                pyplot.hist(new_residuals_dist[coefs.any(axis=1)], 100)
#                print drifts                
#                limits = np.max(np.abs(drifts), axis=0)
                
                axes[0].scatter(drifts[:,0], drifts[:,1], s=50)
#                axes[0].set_xlim(-limits[0], limits[0])
#                axes[0].set_ylim(-limits[1], limits[1])
                axes[0].set_xlabel('x')
                axes[0].set_ylabel('y')                
                axes[1].scatter(drifts[:,0], drifts[:,2], s=50)
                axes[1].set_xlabel('x')
                axes[1].set_ylabel('z')
                
                for ax in axes:
#                    ax.set_xlim(-1, 1)
#                    ax.set_ylim(-1, 1)
                    ax.axvline(0, color='red', ls='--')
                    ax.axhline(0, color='red', ls='--')                    
                
                fig.tight_layout()
            except Exception as e:
                print e
            
            
        return drifts
    
    def shift_images(self, psf_stack, shifts):
        kx = (np.fft.fftfreq(psf_stack.shape[0])) 
        ky = (np.fft.fftfreq(psf_stack.shape[1]))
        kz = (np.fft.fftfreq(psf_stack.shape[2]))
        kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')
        
        shifted_images = np.zeros_like(psf_stack)
        for i in np.arange(psf_stack.shape[3]):
            psf = psf_stack[:,:,:,i]
            ft_image = np.fft.fftn(psf)
            shift = shifts[i]
            shifted_images[:,:,:,i] = np.abs(np.fft.ifftn(ft_image*np.exp(-2j*np.pi*(kx*shift[0] + ky*shift[1] + kz*shift[2]))))
#            shifted_images.append(shifted_image)
        
        return shifted_images
#        namespace[self.output_images] = ImageStack(data=psf_stack, mdh=ims.mdh)
        
def guassian_nd_error(p, dims, data):
    mask = ~np.isnan(data)
    return (data - gaussian_nd(p, dims))[mask]/mask.sum()

def gaussian_nd(p, dims):
    # p = A, bg, x, sig_x, y, sig_y, z, sig_z, ...
    A, bg = p[:2]
    dims_nd = np.meshgrid(*dims, indexing='ij')
    exponent = 0
    for i, dim in enumerate(dims_nd):
        exponent += (dim-p[2+2*i])**2/(2*p[2+2*i+1]**2)
    return A * np.exp(-exponent) + bg

def guassian_sq_nd_error(p, dims, data):
    mask = ~np.isnan(data)
    return (data - gaussian_sq_nd(p, dims))[mask]/mask.sum()

def gaussian_sq_nd(p, dims):
    # p = A, bg, x, sig_x, y, sig_y, z, sig_z, ...
    A, bg = p[:2]
    dims_nd = np.meshgrid(*dims, indexing='ij')
    exponent = 0
    for i, dim in enumerate(dims_nd):
        exponent += (dim-p[2+2*i])**2/(2*p[2+2*i+1]**2)
    return A * np.exp(-exponent)**2 + bg

def lorentzian_nd_error(p, dims, data):
    mask = ~np.isnan(data)
    return (data - lorentzian_nd(p, dims))[mask]/mask.sum()

def lorentzian_nd(p, dims):
    # p = A, bg, x, sig_x, y, sig_y, z, sig_z, ...
    A, bg = p[:2]
    dims_nd = np.meshgrid(*dims, indexing='ij')
    exponent = 1
    for i, dim in enumerate(dims_nd):
#        exponent /= (1 + ((dim-p[2+2*i])/p[2+2*i+1])**2)
        exponent += ((dim-p[2+2*i])/p[2+2*i+1])**2
    return A / exponent + bg

def build_rbf(grids, data):
    grid_nd_list = np.meshgrid(*grids, indexing='ij')
    data = data.flatten()
    mask = ~np.isnan(data)
    data = data[mask]
    grid_nd_list_cleaned = [grid_nd.flatten()[mask] for grid_nd in grid_nd_list]
    grid_nd_list_cleaned.append(data)
    return interpolate.Rbf(*grid_nd_list_cleaned, function='multiquadric', epsilon=1.)

def rbf_nd_error(p, rbf_interpolator):
    return -rbf_interpolator(*p)

def rbf_nd(rbf_interpolator, dims):
    out_shape = [len(d) for d in dims]
    dims_nd_list = np.meshgrid(*dims, indexing='ij')
    dims_nd_list_cleaned = [dim_nd.flatten() for dim_nd in dims_nd_list]
    return rbf_interpolator(*dims_nd_list_cleaned).reshape(out_shape)

@register_module('AveragePSF')
class AveragePSF(ModuleBase):
    """
        Input stacks of PSF and return the (normalized) average PSF.
        Additional filter based on max error/residual between image and averaged image.
    """
    
    inputName = Input('psf_aligned')
    normalize_intensity = Bool(False)
#    normalize_z = Bool(False)
    output_var_image = Output('psf_var')
#    smoothing_method = Enum(['RBF', 'Gaussian'])
#    output_var_image_norm = Output('psf_var_norm')
    gaussian_filter = List(Float, [0, 0, 0], 3, 3)
    residual_threshold = Float(0.1)
    output_images = Output('psf_combined')
    
    def execute(self, namespace):        
        ims = namespace[self.inputName]
        psf_raw = ims.data[:,:,:,:]
        
        # always normalize first, since needed for statistics
        psf_raw_norm = psf_raw.copy()
#        if self.normalize_intensity == True:
        psf_raw_norm /= psf_raw_norm.max(axis=(0,1,2), keepdims=True)
        psf_raw_norm /= psf_raw_norm.sum(axis=(0,1), keepdims=True)
        psf_raw_norm -= psf_raw_norm.min()
        psf_raw_norm /= psf_raw_norm.max()
        
        residual_max = np.abs(psf_raw_norm - psf_raw_norm.mean(axis=3, keepdims=True)).max(axis=(0,1,2))
        print(residual_max)
        mask = residual_max < self.residual_threshold
        print "images ignore: {}".format(np.argwhere(~mask)[:,0])
        print mask
        psf_raw_norm = psf_raw_norm[:,:,:,mask]
        print(psf_raw_norm.shape)
        psf_raw_norm -= psf_raw_norm.min()
        psf_raw_norm /= psf_raw_norm.max()
        
        psf_var = psf_raw_norm.var(axis=3)  
        
#        psf_var_norm = psf_var / psf_combined.mean(axis=3)
        namespace[self.output_var_image] = ImageStack(psf_var, mdh=ims.mdh)
#        namespace[self.output_var_image_norm] = ImageStack(np.nan_to_num(psf_var_norm), mdh=ims.mdh)
        
        # if requested not to normalize, revert back to original data
        if not self.normalize_intensity:
            psf_raw_norm = psf_raw.copy()[:,:,:,mask]
        
        psf_combined = psf_raw_norm.mean(axis=3)
        psf_combined -= psf_combined.min()
        psf_combined /= psf_combined.max()
        
#        if self.smoothing_method == 'RBF':
#            dims = [np.arange(i) for i in psf_combined.shape]
            
#        elif self.smoothing_method == 'Gaussian' and
        if np.any(np.asarray(self.gaussian_filter)!=0):
            psf_processed = ndimage.gaussian_filter(psf_combined, self.gaussian_filter)
        else:
            psf_processed = psf_combined
            
        psf_processed -= psf_processed.min()
        psf_processed /= psf_processed.max()
        
        new_mdh = None
        try:
            new_mdh = MetaDataHandler.NestedClassMDHandler(ims.mdh)
            new_mdh["PSFExtraction.GaussianFilter"] = self.gaussian_filter
            new_mdh["PSFExtraction.NormalizeIntensity"] = self.normalize_intensity            
        except Exception as e:
            print(e)
        namespace[self.output_images] = ImageStack(psf_processed, mdh=new_mdh)
        
        if True:
            fig, axes = pyplot.subplots(2, 3, figsize=(9,6))
            axes[0,0].set_title('X')
            axes[0,0].plot(psf_raw_norm[:, psf_raw_norm.shape[1]//2, psf_raw_norm.shape[2]//2, :])
            axes[1,0].plot(psf_combined[:, psf_combined.shape[1]//2, psf_combined.shape[2]//2], lw=1, color='red')
            axes[1,0].plot(psf_processed[:, psf_processed.shape[1]//2, psf_processed.shape[2]//2], lw=1, ls='--', color='black')
            axes[0,1].set_title('Y')
            axes[0,1].plot(psf_raw_norm[psf_raw_norm.shape[0]//2, :, psf_raw_norm.shape[2]//2, :])
            axes[1,1].plot(psf_combined[psf_combined.shape[0]//2, :, psf_combined.shape[2]//2], lw=1, color='red')
            axes[1,1].plot(psf_processed[psf_processed.shape[0]//2, :, psf_processed.shape[2]//2], lw=1, ls='--', color='black')
            axes[0,2].set_title('Z')
            axes[0,2].plot(psf_raw_norm[psf_raw_norm.shape[0]//2, psf_raw_norm.shape[1]//2, :, :])
            axes[1,2].plot(psf_combined[psf_combined.shape[0]//2, psf_combined.shape[1]//2, :], lw=1, color='red')
            axes[1,2].plot(psf_processed[psf_processed.shape[0]//2, psf_processed.shape[1]//2, :], lw=1, ls='--', color='black')
            
            fig.tight_layout()
            
            fig, ax = pyplot.subplots(1, 1, figsize=(4,3))
            ax.hist(residual_max, bins=20)
            ax.axvline(self.residual_threshold, color='red', ls='--')

@register_module('InterpolatePSF')
class InterpolatePSF(ModuleBase):
    """
        Interpolate PSF with RBF.
        Very stupid. Very slow. Performed on local pixels and combine by tiling.
        Only uses the first color channel
    """
    
    inputName = Input('input')
    rbf_radius = Float(250.0)
    target_voxelsize = List(Float, [100., 100., 100.])

    output_images = Output('psf_interpolated')
    
    def execute(self, namespace):        
        ims = namespace[self.inputName]
        data = ims.data[:,:,:,0]
        
        dims_original = list()
        voxelsize = [ims.mdh.voxelsize.x, ims.mdh.voxelsize.y, ims.mdh.voxelsize.z]
        for dim, dim_len in enumerate(data.shape):
            d = np.linspace(0, dim_len-1, dim_len) * voxelsize[dim] * 1E3
            d -= d.mean()        
            dims_original.append(d)
        X, Y, Z = np.meshgrid(*dims_original, indexing='ij')
        
        dims_interpolated = list()
        for dim, dim_len in enumerate(data.shape):
            tar_len = int(np.ceil((voxelsize[dim]*1E3 * dim_len) / self.target_voxelsize[dim]))
            d = np.arange(tar_len) * self.target_voxelsize[dim]
            d -= d.mean()
            dims_interpolated.append(d)
        
        X_interp, Y_interp, Z_interp = np.meshgrid(*dims_interpolated, indexing='ij')
        pts_interp = zip(*[X_interp.flatten(), Y_interp.flatten(), Z_interp.flatten()])
        
        results = np.zeros(X_interp.size)
        for i, pt in enumerate(pts_interp):
            results[i] = self.InterpolateAt(pt, X, Y, Z, data[:,:,:])
            if i % 100 == 0:
                print("{} out of {} completed.".format(i, results.shape[0]))
        
        results = results.reshape(len(dims_interpolated[0]), len(dims_interpolated[1]), len(dims_interpolated[2]))
#        return results
        new_mdh = None
        try:
            new_mdh = MetaDataHandler.NestedClassMDHandler(ims.mdh)            
            new_mdh["voxelsize.x"] = self.target_voxelsize[0] * 1E-3
            new_mdh["voxelsize.y"] = self.target_voxelsize[1] * 1E-3
            new_mdh["voxelsize.z"] = self.target_voxelsize[2] * 1E-3
            new_mdh['Interpolation.Method'] = 'RBF'
            new_mdh['Interpolation.RbfRadius'] = self.rbf_radius
        except Exception as e:
            print(e)
        namespace[self.output_images] = ImageStack(data=results, mdh=new_mdh)
        
    def InterpolateAt(self, pt, X, Y, Z, data, radius=250.):
        X_subset, Y_subset, Z_subset, data_subset = self.GetPointsInNeighbourhood(pt, X, Y, Z, data, radius)
        rbf = interpolate.Rbf(X_subset, Y_subset, Z_subset, data_subset, function="cubic", smooth=1E3)#, norm=euclidean_norm_numpy)
    #     print pt
        return rbf(*pt)
    
    def GetPointsInNeighbourhood(self, center, X, Y, Z, data, radius=250.):
        distance = np.sqrt((X-center[0])**2 + (Y-center[1])**2 + (Z-center[2])**2)
    #     print distance.shape
        mask = distance < 250.
        
        return X[mask], Y[mask], Z[mask], data[mask]