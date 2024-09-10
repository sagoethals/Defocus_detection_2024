# Defocus_detection
Codes corresponding to the paper "Non linear spatial integration allows the retina to detect the sign of defocus in natural scenes" by Goethals et al., 2024.

### Requirements

In addition to standard scientific packages (scipy, matplotlib), this code requires:
* Seaborn (for plotting)
* PyRetinaSystemIdentification (for CNN predictions)

Electrophysiological data recorded in retinal ganglion cells and defocused natural images
(DOI to be inserted here) must be
[downloaded](here the link to the online repo with data) into the folder
`/Paper defocus/`.

### Defocused images

Defocused images are stored in /Eye model simulations/convolved images/. These are the images tha were projected on the retinas during MEA experiments.
To obtain defocused images, we used the mouse PSFs in /Eye model simulations/mouse PSFs/ and the natural images in /Eye model simulations/original images and ran the code `MOUSE_Make_blurred_images.ipynb`.

## Simulating the mouse eye optics (`Figure_1.ipynb`)

### PSFs and defocused images

We plot the PSFs and defocused images. To obtain defocused images: `MOUSE_Make_blurred_images.ipynb`

## Some types of mouse RGCs detect the sign of defocus (`Figure_2.ipynb` and `Figure_3.ipynb`)

### Cell typing

Codes for typing for each experiment are in /RGC typing/. The RGC types were then homogenized accross experiments with the code `Merge_RGC_types_among_experiments.ipynb`.

### RGC response to defocus

Pre-processing of MEA data : `MEA_1_Select_clusters_makeSTA.ipynb`
Computing spike count in response to defocused images : `MEA_2_Blurred_images_spike_counts.ipynb`
Computing PSTH in response to defocused images : `MEA_PSTH.ipynb`

### CNN model

TODO

## Defocus detector cells encode local spatial contrast (`Figure_4.ipynb`)
  
### LSC in mouse defocused images 

* we obtained the receptive fields (ellipses) as the 2-sigma contour of the STAs : `MOUSE_get_ellipses.ipynb`
* we calculated the LSC in the ellipses, for all OFF slow and ON-OFF local RGC and for the 4 images used in experiments: `MOUSE_measure_LSC.ipynb`
Third
* we used the model of Liu et al., 2022 to predict RGC's responses : `Liu_2022_data.ipynb` to store the data appropriately and `Liu_2022_model.ipynb` to predict

### CNN predictions

TODO

## Local spatial contrast allows defocus detectors to detect the sign of defocus (`Figure_5.ipynb`)

* we calculated the LSC in the 4 images, for all RGCs, and with varying spherical aberrations :`MOUSE_measure_LSC_SAs.ipynb`
  
## Local spatial contrast in human retinal images (`Figure_6.ipynb`)

* we define a generic receptive field and calculate the LSC for the 4 images used in experiments, with varying spherical aberrations : `HUMAN_measure_LSC_SA.ipynb`
* we do the same in hundreds of images to obtain distributions : `HUMAN_measure_LSC_many_images_SA.ipynb`
* we do the same for near vision (`HUMAN_measure_LSC_NV.ipynb` and `HUMAN_measure_LSC_many_images_NV.ipynb`)


