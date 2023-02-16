# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 15:21:08 2021

@author: ray
"""

import os
from pathlib import Path

'''
check all the path in list have existed
    input: 
        path_name -> list -> combine all path into a list

    output:
         Without return any variable, only show the message in console. 
'''

def path_ck(path_name:list, debug:bool=False):
    for name in path_name:
        if Path(name).exists() is False:
            os.makedirs(name)
            if debug:
                print('{} is not exist, build it.'.format(name))
        else:
            if debug:
                print('{} have existed.'.format(name))
            pass
    print('\t*path_ck done.')