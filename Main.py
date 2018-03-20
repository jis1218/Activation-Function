# coding: utf-8
'''
Created on 2018. 3. 20.

@author: Insup Jung
'''
from ActivationFunction.ReLULayer import ReLULayer
import numpy as np

if __name__ == '__main__':
    relu = ReLULayer()
    x = np.array([1.0, -0.5, 3.0, -4.0])
    mask = (x<=0)
    #print(mask)
    relu.foward(x)
    
    y = [1, 2, 3]
    show = (y<2)
    print(show)

    
    pass