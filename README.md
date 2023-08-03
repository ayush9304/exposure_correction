# Exposure Enhancements

Multi-Scale Photo Exposure Correction.

Original Code [MATLAB] for Exposure Enhancement: [https://github.com/mahmoudnafifi/Exposure_Correction](https://github.com/mahmoudnafifi/Exposure_Correction)

## Requirements

- Python 3.9
- Tensorflow 2.x
- scikit-sparse (https://github.com/scikit-sparse/scikit-sparse)
- sparseqr (https://github.com/yig/PySPQR)

## Getting Started

- Create a virtual environment ```conda create -n venv python=3.9```
- Activate the virtual environment ```conda activate venv```
- Install JupyterLab if not already installed ```pip install -r requirements.txt```
- Follow [scikit-sparse instructions](https://github.com/scikit-sparse/scikit-sparse#installation) to install scikit-sparse
- Follow [sparseqr instructions](https://github.com/yig/PySPQR#installation) to install sparseqr
- Run JupyterLab ```jupyter-lab```

## File Structure

```
.
├── bgu
|   ├── Images
|   ├── bgu.py
|   ├── demo_bgu.ipynb
|   └── utils.py
├── data
|   ├── Input_Images
|   ├── Model_Outputs
|   └── Upsampled_Outputs
├── model
|   ├── customLayers
|   |   ├── extractPyrLayer.py
|   |   └── packingGLayer.py
|   ├── __init__.py
|   ├── dlnetwork.py
|   ├── README.txt
|   └── weights.h5
├── demo.ipynb
├── README.md
├── requirements.txt
└── utils.py
```
