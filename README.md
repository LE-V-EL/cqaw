# cqaw-test

Data generation code for CQAW competition. 
Combining codes from Rhoana/Perception and haehn/Instance-based-RN by Zhihao.

- Author: Hayoun Oh @https://mpsych.org/

- Thanks to Dr. Yunhai Wang and his group for providing mid-level dataset and generation code
Generating datasets for CQAW

## To generate your own dataset
1. Install conda
2. `conda env create --file environment.yml`
3. `conda activate cqaw-test`
4.  `cd data-generator`
5. Edit variables in `./generation_script.py` to customize dataset size and path to files
6. Run `python generation_script.py` 

(If you get an ImportError for libL.so, try apt installing `libgl1-mesa-glx`) 
