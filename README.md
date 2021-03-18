# cqaw-test

Data generation code for CQAW competition. 
Combining codes from Rhoana/Perception and haehn/Instance-based-RN by Zhihao.

- Author: Hayoun Oh

- Thanks to Dr. Yunhai Wang and his group for providing mid-level dataset and generation code
Generating datasets for CQAW

## To generate your own dataset
1. `cd data-generator`
2. edit variables in `./generation_script.py` to customize dataset size and path to files
3. run `python generation_script.py` 

(If you get an ImportError for libL.so, apt install `libgl1-mesa-glx`) 
