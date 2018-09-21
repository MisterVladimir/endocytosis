DONE
-------------
1. Load .csv and .zip ImageJ ROI (spun out into fijitools repository)
2. Load TIF images (portion of the ground truth data)
3. Simulate 2D images containing diffraction-limited spots modeled as Gaussians. Images can simulate EMCCD noise.
4. Storing camera metadata in YAML files.


TODO
-------------

0. Writing tests for image loading, and testing this feature again. Minor changes probably have broken this functionality.
1. Scripts to simulate images and save images (along with ground truth locations of simulated spots) as HDF5 files
2. Neural net to detect simulated spots
