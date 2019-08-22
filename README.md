# pyme-psf-extraction
PYME module for PSF extraction

* PSF detected with Difference of Gaussian method from `skimage.feature.blob_dog`.
* PSF detection can be tuned with parameters in the **DetectPSF** module. In addition, additional filters are available to reject bad PSFs.
    * **CropPSF** can reject ROI with multiple peaks or if the ROI center of mass deviates from its geometric center.
	* **CropPSF** also allows manual rejection.
	* **AlignPSF** will reject a poor cross correlation, although not a ROI.
	* **AveragePSF** will calculate maximum error/residual of ROIs against the averaged PSF and can be filtered out.
* PSFs are aligned using (redundant) cross-correlation. All pairs of PSFs are compared.
* Various graphs output are provided for debugging. Rough and not labeled.
    * **DetectPSF** shows the flatten bead stacks with the detected PSF marked by a circle.
    * **AlignPSF** shows the measured offsets in x, y, and z.
	* **AveragePSF** shows a histogram of max error/residual and the threshold used.
	* **AveragePSF** shows cross-sections of the individual PSF stacks, useful for checking for alignment or outliers issues. The averaged PSF before (*in red*) and after (*in black*) smoothing is also displayed.

## Usage
* Use **CombineBeadStacks** to combine multiple bead stacks for analysis. Currently the only way to combine PSF extracted from different files.

* Typically to extract PSF, chain these modules in order:
    1. **Cleanup**
	2. **DetectPSF**
	3. **CropPSF**
	4. **AlignPSF**
	5. **AveragePSF**

* Use **InterpolatePSF** to resample PSF. Testing. Very slow.

## To do's
* Documentation
* Change some of the parameters from pixel size to nanometer
* Metadata
* Label graphs