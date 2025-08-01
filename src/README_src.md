# Note

The project code structure assumes that the database is already downloaded into the google drive. Further information on how to download a kaggle dataset into Google 
Colab is discussed [here](https://www.kaggle.com/discussions/general/74235). The code is organised in a sytematic manner where each individual .py file can be run as
a saperate code block in a python notebook. The project is implemented using pytorch deep learning library.  


# File Descriptions 

The source code is organized into the following files:

-   **`data_load.py`**: Contains code for importing google drive. Also included are command line commands in the form of multi-line comments python comments to load the data in local SSD drive for faster data access. 
-   **`load.py`**: Contains code for importing the foundation model from hugging face.
-   **`model.py`**: Contains code for modyfying the imported model for classification task.
-   **`main.py`**: This file is the main file which includes the code for training the model.

---
