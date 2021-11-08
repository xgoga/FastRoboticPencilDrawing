'''
MIT License

Copyright (c) 2021 
Michal Adamik, Jozef Goga, Jarmila Pavlovicova, Andrej Babinec, Ivan Sekaj

Faculty of Electrical Engineering and Information Technology 
of the Slovak University of Technology in Bratislava
Ilkovicova 3, 812 19 Bratislava 1, Slovak Republic

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import numpy as np
from PIL import Image
from PIL import ImageDraw
from image4layer import Image4Layer

# ----------------- DEFINITIONS ------------------
COLOUR_BLACK = (0, 0, 0, 255) # black background colour
COLOUR_WHITE = (255, 255, 255, 255) # white background colour

# ----------------- FUNCTIONS  ------------------
def selbest(Oldpop,Fvpop,Nums):
    """
    The function copies best chromosomes from the old population into the new 
    population required number of strings according to their fitness. The 
    number of the selected chromosomes depends on the input vector Nums. 
    The best chromosome is the chromosome with the lowest value of its 
    objective function.
    
    Args:
    
        Oldpop: The primary population
    
        Fvpop: The fitness vector of primary population (Oldpop)
    
        Nums:  Vector in the form: Nums=[number of copies of the best chromosome, ... ,
                                         number of copies of the i-th best chromosome, ...]
    
    Returns:
        
        Newpop: The selected population based on fitness and specified input vector Nums
        
        Newfit: The fitness vector of newly created population (Newpop)
        
    """
    
    if(len(Fvpop.shape)<=1):
        Fvpop = np.reshape(Fvpop, (1,len(Fvpop)))
    N = len(Nums)
    fit = np.sort(Fvpop)
    nix = np.argsort(Fvpop)
    Newpop0 = np.zeros((N, Oldpop.shape[1]))
    Newpop = np.zeros((int(np.sum(Nums)), Oldpop.shape[1]))
    Newfit = np.zeros((int(np.sum(Nums),)))
    
    for i in range(N):
        Newpop0[i,:] = Oldpop[nix[0,i],:]
    
    r = 0
    for i in range(N):
        for j in range(Nums[i]):
            Newpop[r,:] = Newpop0[i,:]
            Newfit[r] = fit[0,i]
            r += 1

    return [Newpop, Newfit]


def selsus(Oldpop,Fvpop,n):
    """
    The function selects from the old population a required number of 
    chromosomes using the "stochastic universal sampling" method. Under this 
    selection method the number of a parent copies in the selected new 
    population is proportional to its fitness.
    
    Args:
    
        Oldpop: The primary population
    
        Fvpop: The fitness vector of primary population (Oldpop)
    
        n:  Required number of selected chromosomes
    
    Returns:
        
        Newpop: The selected population based on stochastic universal sampling 
        
        Newfit: The fitness vector of newly created population (Newpop)
        
    """
    
    if(len(Fvpop.shape)<=1):
        Fvpop = np.reshape(Fvpop, (1,len(Fvpop)))
    Newpop = np.zeros((n, Oldpop.shape[1]))
    OldFvpop = np.copy(Fvpop)
    Newfit = np.zeros((n,))
    lpop, lstring = Oldpop.shape
    Fvpop = Fvpop - np.min(Fvpop) + 1
    sumfv = np.sum(Fvpop).astype(np.float32)
    w0 = np.zeros((lpop+1,))

    for i in range(lpop):
        men = Fvpop[0,i]*sumfv
        w0[i] = 1.0/men # creation of inverse weights 
    w0[i+1] = 0
    w = np.zeros((lpop+1,))
    for i in np.arange(lpop-1,-1,-1):
        w[i] = w[i+1] + w0[i]
    maxw = np.max(w)
    if (maxw==0):
        maxw = 0.00001
    w = (w/maxw)*100 # weigth vector
    pdel = 100.0/n 
    b0 = np.random.uniform()*pdel - 0.00001
    b = np.zeros((n,))
    for i in range(1,n+1):
        b[i-1] = (i-1)*pdel + b0
    for i in range(n):
        for j in range(lpop):
            if(b[i]<w[j] and b[i]>w[j+1]):
                break
        Newpop[i,:] = Oldpop[j,:]
        Newfit[i] = OldFvpop[0,j]
    
    return [Newpop,Newfit]



def genLinespop(popsize, Amps, Space):
    """
    The function generates a population of random real-coded chromosomes which 
    genes are limited by a two-row matrix Space. The first row of the matrix 
    Space consists of the lower limits and the second row consists of the upper
    limits of the possible values of genes representing coordinates of line 
    segments. The endpoints are generated by addition or substraction of 
    random real-numbers to the mutated genes. The absolute values of the added 
    values are limited by the vector Amp. 
    
    Args:
    
        popsize: The size of the population (number of chromosomes to be created)
    
        Amps: Matrix of endpoints generation boundaries in the form:
    			[real-number vector of lower limits;
                 real-number vector of upper limits];
    
        Space:  Matrix of gene boundaries in the form: 
	                [real-number vector of lower limits of genes;
                     real-number vector of upper limits of genes];
    
    Returns:
        
        Newpop: The created population of line segments inside 
                specified boudaries 
        
    """
    
    lpop, lstring = Space.shape
    Newpop = np.zeros((int(popsize),int(lstring)))
    if(len(Amps.shape)<=1):
        Amps = np.reshape(Amps, (1,len(Amps)))

    for r in range(int(popsize)):
        dX = Space[1,0] - Space[0,0]
        dY = Space[1,2] - Space[0,2]
        Newpop[r,0] = np.random.uniform()*dX + Space[0,0]
        Newpop[r,2] = np.random.uniform()*dY + Space[0,2]
        for s in [0,2]:
            if(Newpop[r,s]<Space[0,s]):
                Newpop[r,s] = Space[0,s]
            if(Newpop[r,s]>Space[1,s]):
                Newpop[r,s] = Space[1,s]
        
        Newpop[r,1] = Newpop[r,0] + np.random.randint(2*Amps[0,1]+1) - Amps[0,1]  
        Newpop[r,3] = Newpop[r,2] + np.random.randint(2*Amps[0,3]+1) - Amps[0,3] 
        
        for s in [1,3]:
            if(Newpop[r,s]<Space[0,s]):
                Newpop[r,s] = Space[0,s]
            if(Newpop[r,s]>Space[1,s]):
                Newpop[r,s] = Space[1,s]

    return Newpop


def mutLine(Oldpop,factor,Amps,Space):
    """
    The function mutates the population of chromosomes with the intensity
	proportional to the parameter rate from interval <0;1>. Only a few genes  
	from a few chromosomes are mutated in the population. The mutations are 
    realized by addition or substraction of random real-numbers to the mutated 
    genes. The absolute values of the added constants are limited by the vector 
    Amp. Next the mutated strings are limited using boundaries defined in 
	a two-row matrix Space. The first row of the matrix represents the lower 
	boundaries and the second row represents the upper boundaries of 
    corresponding genes.
    
    Args:
    
        Oldpop: The primary population
        
        factor: The mutation rate, 0 =< rate =< 1
    
        Amps: Matrix of gene mutation boundaries in the form:
    			[real-number vector of lower limits;
                 real-number vector of upper limits];
    
        Space:  Matrix of gene boundaries in the form: 
	                [real-number vector of lower limits of genes;
                     real-number vector of upper limits of genes];
    
    Returns:
        
        Newpop: The mutated population 
        
    """
    
    if(len(Amps.shape)<=1):
        Amps = np.reshape(Amps, (1,len(Amps)))
    lpop, lstring = Oldpop.shape
    
    if (factor>1):
        factor = 1
    if (factor<0):
        factor = 0
    
    n = int(np.ceil(lpop*lstring*factor*np.random.uniform()))
    Newpop = np.copy(Oldpop)
    
    for i in range(n):
        rN = np.random.randint(2)
        r = int(np.ceil(np.random.uniform()*lpop))-1
        if (rN==0): # x1, y1
            Newpop[r,0] = Oldpop[r,0] + (2.0*np.random.uniform()-1)*Amps[0,0]
            Newpop[r,2] = Oldpop[r,2] + (2.0*np.random.uniform()-1)*Amps[0,2]
        elif (rN==1): # x2, y2
            Newpop[r,1] = Oldpop[r,1] + (2.0*np.random.uniform()-1)*Amps[0,1]
            Newpop[r,3] = Oldpop[r,3] + (2.0*np.random.uniform()-1)*Amps[0,3]
        
        for s in range(4):
            if (Newpop[r,s]<Space[0,s]):
                Newpop[r,s] = Space[0,s]
            if (Newpop[r,s]>Space[1,s]):
                Newpop[r,s] = Space[1,s]

    return Newpop


def evalFitness(Pop, orimg, geimg, fitOld, LINE_WIDTH, BLEND_MODE):
    """
    Sequentially evaluate fitness function for each chromosome in the population.
    
    Args:
    
        Pop: The population to be evaluated
        
        orimg: Template image
        
        geimg: The current generated image
        
        fitOld: The last fitness vector (from previous generation)
        
        LINE_WIDTH: The width of the line segments (in pixels)
        
        BLEND_MODE: Mode for blending two image layers
    
    Returns:
        
        Fit: The computed fitness vector
        
    """
    
    if(len(Pop.shape)<=1):
        Pop = np.reshape(Pop, (1,len(Pop)))
    lpop, lstring = Pop.shape
    Fit = np.zeros((lpop,))
    
    for i in range(lpop):
        G = Pop[i,:]
        Fit[i] = evalPartialSimilarity(list([orimg, geimg, G, fitOld, LINE_WIDTH, BLEND_MODE]))     
    return Fit


def evalPartialSimilarity(imglist):
    """
    Evaluate partial similarity between original and generated image.
    
    Args:
        
        imglist: list of input parameters
            
            orimg: Template image
            
            geimg: The current generated image
            
            gen: The genetic information of current chromosome (coordinates)
            
            prevFit: The last value of the fitness function (from previous generation)
            
            LINE_WIDTH: The width of the line segments (in pixels)
            
            BLEND_MODE: Mode for blending two image layers
    
    Returns:
        
        The updated value of the fitness function
        
    """
    
    orimg = imglist[0]
    geimg = imglist[1]
    gen = imglist[2]
    prevFit = imglist[3]
    LINE_WIDTH = imglist[4]
    BLEND_MODE = imglist[5]
    
    # evaluate only part of the image
    minX = int(np.min(gen[0:2]))
    maxX = int(np.max(gen[0:2]))
    deltaX = int(maxX - minX)
    
    minY = int(np.min(gen[2:4]))
    maxY = int(np.max(gen[2:4]))
    deltaY = int(maxY - minY)
    
    if (deltaX == 0) and (deltaY != 0):
        deltaX = 1
    elif (deltaX != 0) and (deltaY == 0):
        deltaY = 1
    elif (deltaX == 0) and (deltaY == 0):
        return prevFit
    
    # first evaluation
    if prevFit is None:
        # evaluate full image

        # create a blank canvas
        draw = Image.new('RGBA', geimg.size, COLOUR_WHITE)
        draw = draw.convert('L')
        pdraw = ImageDraw.Draw(draw)
        
        line = ((gen[2], gen[0]),(gen[3], gen[1]))
        
        # create new binary image mask
        mask_img = Image.new('1', (geimg.size[0], geimg.size[1]), 0)
        ImageDraw.Draw(mask_img).line(line, fill=1, width=LINE_WIDTH)
        mask = np.array(mask_img)

        tgrey = orimg[:,:] * mask
        tgrey = tgrey[tgrey != 0]

        # if mask is an empty array
        if (tgrey.size == 0):
            return prevFit
        
        # compute the lighest shade of the line segment
        c = int(np.max(tgrey))
        
        # draw one line segment
        pdraw.line(line, fill=(c), width=LINE_WIDTH)
        
        # call blending mode function by name
        out = eval('Image4Layer.' + BLEND_MODE)(draw, geimg)

        # compute similarity between original and generated image
        return np.sqrt(np.sum((orimg - np.asarray(out, dtype=int))**2.0)/(out.size[0]*out.size[1])) / 255.0 
            
    else:
        # reconstruct previous similarity 
        tSum = ((prevFit*255.0)**2.0) * geimg.size[1] * geimg.size[0]  
        
        # create a blank canvas
        draw = Image.new('RGBA', (deltaY, deltaX), (255,255,255,0))
        draw = draw.convert('L')
        pdraw = ImageDraw.Draw(draw)
        line = ((gen[2]-minY, gen[0]-minX),(gen[3]-minY, gen[1]-minX))
        mask_img = Image.new('1', (draw.size[0], draw.size[1]), 0)
        ImageDraw.Draw(mask_img).line(line, fill=1, width=LINE_WIDTH)
        mask = np.array(mask_img)
        
        tgrey = orimg[minX:minX+deltaX, minY:minY+deltaY] * mask
        tgrey = tgrey[tgrey != 0]
        
        # if mask is an empty array
        if (tgrey.size == 0):
            return prevFit
        
        # compute the lighest shade of the line segment
        c = int(np.max(tgrey))
        
        # create new line segment
        pdraw.line(line, fill=(c), width=LINE_WIDTH)
        partgenImg = geimg.crop((minY, minX, minY + deltaY, minX + deltaX))
        
        # call blending mode function by name
        out = eval('Image4Layer.' + BLEND_MODE)(draw, partgenImg)

        # substract similarity between previously generated image and target image + add newly computed part
        newSum = tSum - np.sum((orimg[minX:minX+deltaX, minY:minY+deltaY] - np.asarray(partgenImg, dtype=int))**2.0) + np.sum((orimg[minX:minX+deltaX, minY:minY+deltaY] - np.asarray(out, dtype=int))**2.0)
        if (newSum < 0):
            return prevFit
        else:
            return np.sqrt(newSum/(geimg.size[0]*geimg.size[1])) / 255.0


def computeLineShade(orimg, gen, LINE_WIDTH):  
    """
    Extracts the shade of grey for line segment rendering from the image template.
    
    Args:
            
        orimg: Template image
        
        gen: The genetic information of current chromosome (coordinates)
        
        LINE_WIDTH: The width of the line segments (in pixels)
            
    Returns:
        
        The shade of grey for the specified line segment <0, 255>
        
    """
    
    line = ((gen[0][0], gen[0][1]),(gen[1][0], gen[1][1]))
    
    # create new binary image mask
    mask_img = Image.new('1', (orimg.shape[1], orimg.shape[0]), 0)
    ImageDraw.Draw(mask_img).line(line, fill=1, width=LINE_WIDTH)
    mask = np.array(mask_img)

    tgrey = orimg[:,:] * mask
    tgrey = tgrey[tgrey != 0]
    if (tgrey.size == 0):
        return 255
    
    # return the lightest shade
    return int(np.max(tgrey))




