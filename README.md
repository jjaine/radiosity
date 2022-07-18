# radiosity

## Matlab
Matlab code for solving the radiosity equation for lighting in computer graphics virtual spaces (folder `matlab`). 

The code is related to the examples I show in the YouTube video https://youtu.be/krIVZvzlxUQ Note that the video language is Finnish but that it has English subtitles. 

1. Simplest example: empty room. 

First run "radiosity_emptyroom_Fcomp.m" in Matlab to create the geometric form factor matrix F, which will be saved to a file in the subfolder ./data/. Then you can run "radiosity_emptyroom_BW.m" to create a black-and-white image of a lit room, and "radiosity_emptyroom_color.m" for a color image. If you want more details (smaller patches), you can make n larger in "radiosity_emptyroom_Fcomp.m". Note that too large n value will cause your computer to run out of memory, so it is a good idea to increase n gradually. 

2. Another example: room with a dividing wall. 

First run "radiosity_wall_Fcomp.m" in Matlab to create the geometric form factor matrix F, which will be saved to a file in the subfolder ./data/. Then you can run "radiosity_wall_color.m" to create a color image. If you want more details (smaller patches), you can make halfn larger in "radiosity_wall_Fcomp.m". Note that too large halfn value will cause your computer to run out of memory, so it is a good idea to increase halfn gradually. 

3. Third example: room with a (levitating) table. 

First run "radiosity_table_Fcomp.m" in Matlab to create the geometric form factor matrix F, which will be saved to a file in the subfolder ./data/. Then you can run "radiosity_table_color.m" to create a color image. If you want more details (smaller patches), you can make halfn larger in "radiosity_wall_Fcomp.m". Note that too large halfn value will cause your computer to run out of memory, so it is a good idea to increase halfn gradually. 

## Python
Python code for solving the radiosity equation for lighting in computer graphics virtual spaces (folder `python`).

The code is converted to Python from the aforementioned Matlab code. 

Prerequisites: Python3, Matplotlib, SciPy

Installing prerequisites:
* create a Python virtual environment with `python3 -m venv radiosity_env`,
* activate the Python virtual environment with `source python/radiosity_env/bin/activate`,
* install Matplotlib `pip3 install matplotlib`,
* install SciPy `pip3 install scipy`, 
* (optional) patch Matplotlib with [PR#22046](https://github.com/matplotlib/matplotlib/pull/22046), for example by adding the changes by hand to the files in `radiosity_env/lib/python*/site-packages/`.

Then go to the sources folder with `cd python/src` for examples.

1. Simplest example: empty room.

First run "radiosity_emptyroom.py" with `python3 radiosity_emptyroom.py` to create the geometric form factor matrix F, which will be saved to a file in the subfolder `../data`. Then you can run "radiosity_emptyroom_color.py" with `python3 radiosity_emptyroom_color.py` to create a color image of a lit room. If you want more details (smaller patches), you can make `n` larger in "radiosity_emptyroom.py". Note that too large n value will cause your computer to run out of memory, so it is a good idea to increase n gradually.

2. Iterative version of the empty room.

Run "radiosity_emptyroom_iterative.py" with `python3 radiosity_emptyroom.py` to use the iterative method for solving radiosity. 