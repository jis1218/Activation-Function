# coding: utf-8
'''
Created on 2018. 3. 20.

@author: Insup Jung
'''

class ReLULayer(object):
    '''
    classdocs
    '''


    def __init__(self):
        self.mask = None
        
    
    def forward(self, x):
        self.mask = (x <=0) #mask에는 x의 형상대로 true 또는 false가 들어간다.
        out = x.copy()
        print(out)
        out[self.mask]=0 #out중 true인 것에만 0을 넣어준다. numpy array의 특징
        print(out)
        print(self.mask)
        
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        
        return dx