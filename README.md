endocytosis
==================
**Author**: Vladimir Shteyn [GitHub](https://github.com/mistervladimir) [LinkedIn](https://www.linkedin.com/in/vladimir-shteyn/)

This is a work in progress. More 'How To'-type of information will be
provided once this is in a working state. That said, the eventual goal is
to segment fluorescently-labeled endocytic proteins imaged on a confocal
microscope using machine learning techniques. (Other particle tracking has
been tried, but these require manual filtering of the data, which is labor
intensive). As I'm a novice when it comes to machine learning (ML), my
first aim is to teach myself ML-based techniques, practicing on simulated
data e.g. point spread function models' images placed in a Poisson noise
background. It may be possible to use simulated to also train the
algorithm for actually detecting objects of interest in the microscopy
images. 

Prerequisites
------------------
numpy
scipy
tensorflow
h5py

[optional]
javabridge
python-bioformats

Instalation
------------------
pip install git+https://github.com/MisterVladimir/endocytosis/archive/0.1.tar.gz'

License
------------------
This project is licensed under the GNUv3 License - see the
[LICENSE.rst](LICENSE.rst) file for details. 

Acknowledgments
------------------
Many thanks to David Baddeley, Kenny Chung, and Rui Ma, my former colleagues
at Yale West Campus Nanobiology Institute who have helped me learn math,
statistics, and computer programming. Also a thanks to Jo