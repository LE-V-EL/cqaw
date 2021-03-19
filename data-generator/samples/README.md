# CQAW competition data - Quick Start

### Data classes

- Level 1
    - `LENGTH`: One line, representing a value by its length (in pixels). 
    - `ANGLE`: One angle, representing a value by its angle size (in degree). 
    - `LENGHTS`: Four lines in one figure, each representing a value by its length (in pixels). Individual lines in each figure are not labeled with names, and queries ask for the lengths of all lines in each figure, from left to right. 
    - `ANGLES`: Four angles in one figure, each representing a value by its angle size (in degree). Individual angles in each figure are not labeled with names, and queries ask for the size of all angles in each figure, from left to right. 
- Level 2
    - `SIMPLE_BAR`: Bar plots composed of 3-6 values. Individual bars in each training data are not labeled with names, and queries ask for the values reperesented by all bars in each figure, from left to right. 
    - `SIMPLE_PIE`: Pie plots composed of 3-6 values. Individual pies in each training data are not labeled with names, and queries ask for the values reperesented by all pies in each figure, in clockwise direction from the top. 
- Level 3
    - `ADVANCED_BAR`: Bar plots composed of 3-6 values. Individual bars in each training data are labeled with apple varieties, while individual bars in each test data are labeled with orange varieties. 
    - `ADVANCED_PIE`: Pie plots composed of 3-6 values. Individual pies in each training data are labeled with apple varieties, while individual pies in each test data are labeled with orange varieties. 

### Dataset format

Datasets from level 1, 2 and 3 are stored in separate *.zip* files. When you unzip each *.zip* file, you will find 200,000 - 400,000 *.png* files and two *.csv* files inside the top directory. 

*.png* files are 400*400 images, [0, 255] in greyscale. 

Two .csv files contain metadata for training and test dataset. You can match queries and labeled with images using the metadata. 

- *TRAIN_metadata.csv*
    - Header: [filename, level, classtype, query, label]
- *TEST_metadata.csv*
    - Header: [filename, level, classtype, query]

### Creating additional data

If you find additional data necessary, you can create a new dataset using the script on [this repository](http://link-is-dead.com). 
