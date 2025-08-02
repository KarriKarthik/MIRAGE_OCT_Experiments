# Note

The project code structure assumes that the database is already downloaded into Google Drive. In this project, the subfolders in the database have been zipped to reduce the storage requirement in Google Drive. Further information on how to download a Kaggle dataset into Google Colab is discussed [here](https://www.kaggle.com/discussions/general/74235). The code is organised in a systematic manner where each individual Python(.py) file can be run as a separate code block in a Python notebook. The project is implemented using the PyTorch deep learning library.  


# File Descriptions 

The source code is organized into the following files:

-   **`data_load.py`**: Contains code for importing Google Drive. Also included are command line commands in the form of multi-line comments and Python comments to load the data on a local SSD drive for faster data access. 
-   **`load.py`**: Contains code for importing the foundation model from Hugging Face.
-   **`model.py`**: Contains code for modifying the imported model for a classification task.
-   **`main.py`**: This file is the main file which includes the code for training the model.


[|IMPORTANT]
 Important 
