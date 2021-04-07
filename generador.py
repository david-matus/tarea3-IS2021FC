import matplotlib.pyplot as plt
import random as rnd
import scipy.fft
import numpy as np

def Grapher(graphMatrix,xTitle,yTitle,plotLabel):
    '''
    Graphs a 2-D plot of an X and Y vector pair using matplotlib.
    
    Parameters:
    graphMatrix: 2xn matrix
        a matrix containing [X,Y] coordinates to be plotted
    xTitle: str
        title for the x axis
    yTitle: str
        title for the y axis
    plotLabel: str
        label for the output plot
    
    Returns:
        a matplotlib graph of the desired coordinates
    '''
    #the vectors for X and Y (respectively) are plotted
    plt.plot(graphMatrix[0], graphMatrix[1])
    plt.xlabel(xTitle)
    plt.ylabel(yTitle)
    plt.title(plotLabel)
    plt.show()
    return

def RandSinMix(sinTotal,maxFreq,startX,endX,spacing):
    '''
    Generates a periodic function using a group of randomly
    generated sine waves.
    
    Parameters:
    sinTotal: int
        number of random sin waves to be included
    maxFreq: float
        maximum range for the angular frequency of sine waves
    startX: float
        first point on X on which the wave will be evaluated
    endX: float
        final point on X on which the wave will be evaluated
    spacing: float
        difference between each X value on which the wave will
        be evaluated
    
    Returns:
    [X,Y]: 2xn matrix
        a matrix containing [X,Y] coordinates of the wave
        X: 1xn array
            an array containing x coordinates of the wave
        Y: 1xn array
            an array containing y coordinates of the wave
    '''
    #determining if values increase or decrease
    direction=(startX<=endX)-(endX<startX)
    #X and Y vectors initialized
    size=int(abs((endX-startX)/spacing))
    X=[0]*size
    Y=[0]*size
    print('randomly chosen angular frequencies:')
    for i in range(sinTotal):
        #random frequency and phase chosen
        a=rnd.random()
        b=rnd.random()
        for j in range(size):
            #each XY pair is calculated
            X[j]=startX+spacing*direction*j
            Y[j]+=np.sin(a*maxFreq*X[j]+2*b*np.pi)
        print(a*maxFreq)
    return [X,Y]

def AddNoise(dataMatrix,maxNoise):
    '''
    Takes an input data set and adds uniformly random data as
    noise. Each Y value is randomly displaced from its original
    vertical position. No (x,y) relationship is modified.
    
    Parameters:
    graphMatrix: 2xn matrix
        a matrix containing [X,Y] coordinates of the dataset
    maxNoise: float
        maximum value a y point can be randomly displaced
    
    Returns:
    dataMatrix: 2xn matrix
        a matrix containing the new [X,Y] data points
    '''
    size=len(dataMatrix[1])
    for i in range(size):
        #wether to sum or substract is randomly chosen
        rngMidpoint=rnd.random()
        sign=(rngMidpoint<=0.5)-(0.5<rngMidpoint)
        #the Y datapoint is modified
        dataMatrix[1][i]+=sign*maxNoise*rnd.random()
    return dataMatrix

def SciPyFFT(dataMatrix):
    '''
    Takes an input data set and outputs the fourier transform
    of said dataset, using the scipy.fft library.
    
    Parameters:
    dataMatrix: 2xn matrix
        a matrix containing [X,Y] coordinates of the dataset,
        in that order
    
    Returns:
    [xtrans,ytrans]: array matrix
        a matrix containing numpy arrays corresponding to the
        fourier transform, arrays are in a 1:2 size relation
        xtrans: numpy array
            a 1xn numpy array of the frequencies involved in
            the transformation
        ytrans: numpy array
            a 1x2n numpy array of the intensity associated
            with each frequency
    '''
    size=len(dataMatrix[0])
    y=np.array(dataMatrix[1])
    #y is transformed and xtrans is generated using scipy.fft
    ytrans=scipy.fft.fft(y)
    xtrans=scipy.fft.fftfreq(size)[:size//2]
    return [xtrans,ytrans]

def FFTNoiseReduction(FFTdataMatrix,threshold):
    '''
    Takes an input Fourier frequency sectrum and adjusts noise
    according to an input threshold. Reduces all undesired
    frequencies to a zero intensity.
    
    Parameters:
    FFTdataMatrix: array matrix
        a matrix containing a numpy array with the frequencies
        of the transformation and a numpy array with their
        intensities, in that order. Array size is in a 1:2
        relation
    threshold: float
        minimum intensity a frequency has to achieve in order
        to avoid deletion
    
    Returns:
    outMatrix: array matrix
        a matrix containing a numpy array of the frequencies
        and a numpy array of the intensities of the adjusted
        FFT spectrum. Array size is in a 1:2 relation
    '''
    outMatrix=FFTdataMatrix
    position=0
    #removed variables are used only for ease of use purposes
    removedCount=0
    removedTotal=0
    #the module of each FFTy value is compared to threshold
    for i in FFTdataMatrix[1]:
        a=np.abs(i)
        #undesired frequencies are set to zero intensity
        outMatrix[1][position]=i*(a>=threshold)
        removedCount+=(a<threshold)
        removedTotal+=a*(a<threshold)
        position+=1
    removedAvg=removedTotal/removedCount
    print('average intensity of removed frequencies: ',removedAvg)
    return outMatrix

def SciPyInvFFT(dataMatrix,FFTdataMatrix):
    '''
    Takes an adjusted fourier spectrum and its pre-adjustment
    [X,Y] coordinate set and outputs the adjusted [X,Y]
    coordinate set. Does not perform any FFT calculations,
    transformation is left to previos work done on the dataset.
    
    Parameters:
    dataMatrix: 2xn matrix
        a matrix containing [X,Y] coordinates of the unmodified
        dataset
    maxNoise: float
        maximum value a y point can be randomly displaced
    
    Returns:
    [Xreal,Yreal]: 2xn matrix
        a matrix containing the adjusted [X,Y] data points
        Xreal: 1xn array
            an array containing the adjusted x coordinates
        Yreal: 1xn array
            an array containing the adjusted y coordinates
    '''
    #the adjusted FFT spectrum is inverted
    Yinv=scipy.fft.ifft(FFTdataMatrix[1])
    #the datasets are combined to create the output matrix
    Yreal=np.real(Yinv[0:len(Yinv)])
    Xreal=dataMatrix[0]
    return [Xreal,Yreal]

#signal is generated
sinTotal=int(input('number of sine functions: '))
maxFreq=float(input('maximum allowed frequency: '))
startX=float(input('starting point: '))
endX=float(input('end point: '))
spacing=float(input('sampling spacing: '))
[X,Y]=RandSinMix(sinTotal,maxFreq,startX,endX,spacing)
Grapher([X,Y],'X','Y','original wave')

#noise is added to the signal
maxNoise=float(input('maximum allowed noise: '))
[noisyX,noisyY]=AddNoise([X,Y],maxNoise)
Grapher([X,Y],'noisy X','noisy Y','noisy wave')

#FFT is applied to noisy signal
[Xtrans,Ytrans]=SciPyFFT([noisyX,noisyY])
size=len(Ytrans)
Grapher([Xtrans,2/size*np.abs(Ytrans[0:size//2])],'transformed X','transformed Y','FFT spectrum of noisy wave')

#FFT is cleaned to reduce noise
threshold=float(input('noise threshold: '))
[cleanXtrans,cleanYtrans]=FFTNoiseReduction([Xtrans,Ytrans],10)
Grapher([cleanXtrans,2/size*np.abs(cleanYtrans[0:size//2])],'adjusted transformed X','adjusted transformed Y','FFT spectrum of adjusted wave')

#clean FFT data is transformed to waveform
[cleanX,cleanY]=SciPyInvFFT([X,Y],[cleanXtrans,cleanYtrans])
Grapher([cleanX,cleanY],'adjusted X','adjusted Y','adjusted wave')
