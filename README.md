# pyme-psf-extraction
A bead PSF extraction recipe module written for [PYME](https://python-microscopy.org/).

Designed to crop, align and average bead PSFs.
Some capability for rejecting bad bead PSFs are provided. Deconvolution not part of this module but is available in PYME.

## System requirements
* Windows, Linux or OS X
* Python 2.7
* PYME (>19.02.27) and dependencies

  
- Tested on Windows 10 with PYME (19.02.27)
## Installation

1. Clone repository.
2. Run the following in the project folder. 
	```
		python setup.py develop
	```
3. Start `hd5view` (PYME).

(Runtime < 5 min)

## Demo
1. Open this simulated image of beads ([beads_simulated.h5](/psf_extraction/example/beads_simulated.h5)) with `dh5view` (PYME).
2. Load and run this demo recipe ([bead_psf_extraction.yaml](/psf_extraction/example/bead_psf_extraction.yaml)).
3. Open the extracted PSF by clicking the output (default: `psf_combined`) on the final **AveragePSF** module.

(Runtime < 5 min)

## Instructions
1. Refer to [PYME documention](https://python-microscopy.org/doc/index.html) for general use of PYME.
2. Open image file with `dh5view` (PYME).
3. Chain the modules in this order:
	1. **CombineBeadStacks** (optional)
	2. **Cleanup**
	3. **DetectPSF**
	4. **CropPSF**
	5. **AlignPSF**
	6. **AveragePSF**
	
	Detailed description of each module and their inputs, outputs and parameters are accessible in PYME.
4. Run recipe.
5. Click on the extracted PSF output from **AveragePSF** and save as necessary.

## To do's
* Change some of the parameters from pixel size to nanometer
* Add more metadata