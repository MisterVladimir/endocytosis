endocytosis
==================
**Author**: Vladimir Shteyn 

**GitHub**: [GitHub](https://github.com/mistervladimir)

**LinkedIn**: [LinkedIn](https://www.linkedin.com/in/vladimir-shteyn/)

Introduction
------------------
This is a work in progress. More 'How To Use This Software'-type of information will be provided once this is in a more-complete state. That said, the eventual goal is to segment timelapse images (videos) of fluorescently-labeled endocytic proteins generated on a confocal microscope using machine learning techniques. (Other particle tracking methods have been tried, but these require extensive manual cleaning. In fact, it is this manually-cleaned data that will serve as ground truth). As I'm a novice in machine learning (ML), for now my aim is to teach myself ML-based techniques, first deploying them on simulated microscopy data.

Single molecule switching (SMS, otherwise known as [PALM](https://en.wikipedia.org/wiki/Photoactivated_localization_microscopy) or STORM) data may serve as a good heuristic for simulateable data. This is relatively easy to do: each object is a diffracion-limited spot (a [point spread function (PSF)](https://en.wikipedia.org/wiki/Point_spread_function)) modeled by a 2d Gaussian. EMCCD camera noise too may be modeled and serves as background for the Gaussian spots placed randomly in the image. Note that while SMS data is typically in timelapse form, for simplicity here we similuate video frames independently. 


Requirements
------------------
[required]

numpy

scipy

tensorflow

h5py

fijitools



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
statistics, and computer programming. Also a thanks to Joël Lemière, who spent
several years generating the ground truth data.
