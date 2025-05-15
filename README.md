# F2HGNN

1. Environment Requirements

- Ubuntu 18.04
- Anaconda with Python 3.8.0
- CUDA 11.2

Note: The specific python package list of our environment is included in the requirements.txt. The tensorflow version can be 2.5.0. The installation may need several minutes if there is no environmental conflicts.

1. Hardware requirements Needs a Linux server with larger than 45GB memory. GPU cores are optional but recommended.
2. Running

- Download datasets from their original sources
- Convert then into csv matrix formats (rows: users, columns: items, value: ratings)
- Execute "python Final.py"

Note: The logs will show the training RMSE and the test results. The estimated running time is from tens of minutes to hours, depending on the dataset size.

