# -*- coding: utf-8 -*-

import sys, math
import numpy as np

MAX_FILTER_LENGTH = 8

class FilterClass():
    def __init__(self, shape=None):
    # defult constructor initialize all variables in 0
    
        self.Ncoeff = 6 #number of coefficients
        self.i = 0

        if shape is None:
            self.data_shape = (MAX_FILTER_LENGTH,)
        elif type(shape) == int:
            self.data_shape = (MAX_FILTER_LENGTH, shape)
        else:
            self.data_shape = (MAX_FILTER_LENGTH, *shape)
        self.x = np.zeros(self.data_shape)
        self.y = np.zeros(self.data_shape)
        self.a = np.zeros(MAX_FILTER_LENGTH)
        self.b = np.zeros(MAX_FILTER_LENGTH)
    
        self.InitF = False

    def clear_filter(self):
        #clear the filter as constructor does
        self.x = np.zeros(self.data_shape)
        self.y = np.zeros(self.data_shape)
        self.a = np.zeros(MAX_FILTER_LENGTH)
        self.b = np.zeros(MAX_FILTER_LENGTH)

    #Build a butterwoth filter. T is the sample period,
    #cutoff is the cutoff frequency in hertz, N is the order (1,2,3 or 4)            
    def butterworth(self, T, cutoff, N):
        self.clear_filter()
        
        if N > 4:
            N = 4
        if N == 0:
            N = 1
        C = 1.0/math.tan(math.pi * cutoff * T)
        
        if N == 1:
            A = 1.0/(1.0+C)
            self.a[0] = A
            self.a[1] = A
            self.b[0] = 1.0
            self.b[1] = (1.0-C)*A
        
        elif N == 2:
            A = 1.0/(1.0+1.4142135623730950488016887242097*C+math.pow(C,2))
            self.a[0] = A
            self.a[1] = 2*A
            self.a[2] = A
            
            self.b[0] = 1.0
            self.b[1] = (2.0-2*math.pow(C,2))*A
            self.b[2] = (1.0-1.4142135623730950488016887242097*C+math.pow(C,2))*A
        
        elif N == 3:
            A=1.0/(1.0+2.0*C+2.0*math.pow(C,2)+math.pow(C,3))
            self.a[0]=A
            self.a[1]=3*A
            self.a[2]=3*A
            self.a[3]=A

            self.b[0]=1.0;
            self.b[1]=(3.0+2.0*C-2.0*math.pow(C,2)-3.0*math.pow(C,3))*A
            self.b[2]=(3.0-2.0*C-2.0*math.pow(C,2)+3.0*math.pow(C,3))*A
            self.b[3]=(1.0-2.0*C+2.0*math.pow(C,2)-math.pow(C,3))*A
            
        elif N == 4:
            A=1.0/(1+2.6131259*C+3.4142136*math.pow(C,2)+2.6131259*math.pow(C,3)+math.pow(C,4))
            self.a[0]=A
            self.a[1]=4.0*A
            self.a[2]=6.0*A
            self.a[3]=4.0*A
            self.a[4]=A

            self.b[0]=1.0
            self.b[1]=(4.0+2.0*2.6131259*C-2.0*2.6131259*math.pow(C,3)-4.0*math.pow(C,4))*A
            self.b[2]=(6.0*math.pow(C,4)-2.0*3.4142136*math.pow(C,2)+6.0)*A
            self.b[3]=(4.0-2.0*2.6131259*C+2.0*2.6131259*math.pow(C,3)-4.0*math.pow(C,4))*A
            self.b[4]=(1.0-2.6131259*C+3.4142136*math.pow(C,2)-2.6131259*math.pow(C,3)+math.pow(C,4))*A
            
    def initializeFilter(self,X):
        for i in range(MAX_FILTER_LENGTH):
            self.x[i] = X
            self.y[i] = X
            
    def applyFilter(self,X):
        #assumes the filter was already run so that the coefficients are able
        #receive the new input- return the filter signal
        self.x[1:5] = self.x[0:4]
        self.x[0] = X
        self.y[1:5] = self.y[0:4]
        self.y[0] = np.dot(self.x[:5].T, self.a[:5]).T - \
                    np.dot(self.y[1:5].T, self.b[1:5]).T
        return self.y[0]
