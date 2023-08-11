# Detection of Fungal Disease in Citrus Fruit Based on Hyperspectral Imaging

The dataset will be made available for access after the publication of the article. Please wait for further updates.

## Dataset

The dataset can be obtained here. It contains recordings of:

- `Healthy citrus`		200 images
- `Phytophthora syringae`		185 images
- `Phytophthora citricola`		210 images
- `Phytophthora citrophthora`		205 images

<img src=".\images\image_1.png" style="zoom: 20%;" />

​						Fig. 1.RGB images of three citrus diseases synthesized from visible light spectra.



Relevant information about hyperspectral imaging devices：

- The portable snapshot hyperspectral imaging system used in this study is composed of `Specim FX 10e `(Spectral Imaging.Ltd,Finland), a dark box, and a computer installed with SpecView data collection software.  The camera is configured to capture images with dimensions of 1024×1024 pixels.  Each hyperspectral image has `224` channels, with a spectral resolution of 5.5 nm, a spectral sampling interval of 2.7 nm, and a spectral range from `400 to 1000 nanometers`, maintaining the visible light (VIS) range plus a lower near-infrared (NIR) range.

### Requirements

- Python 3.9.13
- PyTorch 1.12.0
- visdom
- Download the data set to a local folder

## Our model

This is the official implementation of the network, based on PyTorch.

The code is divided into subfolders, which correspond to the use cases:

- `checkpoint `contains the training process of all tasks. Here, you can find various pre-trained models, including the training results for RGB, raw spectral images, and dimensionality-reduced spectral images.

- `dataset` stores the dataset. Please extract the downloaded data into this directory.

- `Model`  contains the implementation of various models.

- `train.py` contains the training code, where you need to manually open the annotation selection model. The default model is `our_model`.

- `validAccResult.py` Validate the model effect through the test set. Replace the saved model with the corresponding position in the code.The default model is `Our_model_CARS.pth`.

