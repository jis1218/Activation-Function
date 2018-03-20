# coding: utf-8
'''
Created on 2018. 3. 20.

@author: Insup Jung
'''
from _nsis import out

class SigmoidLayer(object):

    def __init__(self, params):
        self.out = None
        '''
        Constructor
        '''
    
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        
        return out
    
    def backward(self, dout):
        dx = dout*(1.0-self.out)*self.out
        
        return dx
        