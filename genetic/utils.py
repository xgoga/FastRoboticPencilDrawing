#import marshal
import numpy as np
from PIL import Image
from PIL import ImageDraw
from types import FunctionType
from image4layer import Image4Layer

# ----------------- DEFINITIONS ------------------
COLOUR_BLACK = (0, 0, 0, 255) # black background colour
COLOUR_WHITE = (255, 255, 255, 255) # white background colour

# ----------------- FUNCTIONS  ------------------

'''
 selbest - selection of best strings

	Description:
	The function copies from the old population into the new population
	required a number of strings according to their fitness. The number of the
	selected strings depends on the vector Nums as follows:
	Nums=[number of copies of the best string, ... ,
             number of copies of the i-th best string, ...]
	The best string is the string with the lowest value of its objective function.


	Syntax:

	Newpop=selbest(Oldpop,Oldfit,Nums);
	[Newpop,Newfit]=selbest(Oldpop,Oldfit,Nums);

	Newpop - new selected population
       Newfit - fitness vector of Newpop
	Oldpop - old population
	Oldfit - fitness vector of Oldpop
	Nums   - vector in the form: Nums=[number of copies of the best string, ... ,
                                          number of copies of the i-th best string, ...]

'''
def selbest(Oldpop,Fvpop,Nums):
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


'''
 selsus - stochastic universal selection


	Description:
	The function selects from the old population a required number of strings using
	the "stochastic universal sampling" method. Under this selection method the number of a 
       parent copies in the selected new population is proportional to its fitness. 
	
 
	Syntax:

	Newpop=selsus(Oldpop,Oldfit,Num);
	[Newpop,Newfit]=selsus(Oldpop,Oldfit,Num);

	       Newpop - new selected population
	       Newfit - fitness vector of Newpop
	       Oldpop - old population
	       Oldfit - fitness vector of Oldpop
	       Num    - required number of selected strings

'''
def selsus(Oldpop,Fvpop,n):
    if(len(Fvpop.shape)<=1):
        Fvpop = np.reshape(Fvpop, (1,len(Fvpop)))
    Newpop = np.zeros((n, Oldpop.shape[1]))
    OldFvpop = np.copy(Fvpop)
    Newfit = np.zeros((n,))
    lpop, lstring = Oldpop.shape
    Fvpop = Fvpop - np.min(Fvpop) + 1 # uprava na kladnu f., min=1
    sumfv = np.sum(Fvpop).astype(np.float32)
    w0 = np.zeros((lpop+1,))

    for i in range(lpop):
        men = Fvpop[0,i]*sumfv
        w0[i] = 1.0/men # tvorba inverznych vah
    w0[i+1] = 0
    w = np.zeros((lpop+1,))
    for i in np.arange(lpop-1,-1,-1):
        w[i] = w[i+1] + w0[i]
    maxw = np.max(w)
    if (maxw==0):
        maxw = 0.00001
    w = (w/maxw)*100 # vahovaci vektor
    pdel = 100.0/n # rovnomerne rozdeli interval na vyberove body 
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


''' Generate a population of lines '''
def genLinespop(popsize, Amps, Space):
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


''' Mutate a population of lines '''
def mutLine(Oldpop,factor,Amps,Space):
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


''' Sequentially evaluate fitness function '''
def evalFitness(Pop, orimg, geimg, fitOld, LINE_WIDTH, BLEND_MODE):
    if(len(Pop.shape)<=1):
        Pop = np.reshape(Pop, (1,len(Pop)))
    lpop, lstring = Pop.shape
    Fit = np.zeros((lpop,))
    
    for i in range(lpop):
        G = Pop[i,:]
        Fit[i] = evalPartialSimilarity(list([orimg, geimg, G, fitOld, LINE_WIDTH, BLEND_MODE]))     
    return Fit


''' Evaluate partial similarity between original and generated image '''
def evalPartialSimilarity(imglist):
    from PIL import Image, ImageDraw
    import numpy as np
    
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

        # make a blank image for the text, initialized to transparent color
        draw = Image.new('RGBA', geimg.size, COLOUR_WHITE) # canvas
        draw = draw.convert('L')
        pdraw = ImageDraw.Draw(draw)
        
        line = ((gen[2], gen[0]),(gen[3], gen[1]))
        
        # create new image ("1-bit pixels, black and white", (width, height), "default color")
        mask_img = Image.new('1', (geimg.size[0], geimg.size[1]), 0)
        ImageDraw.Draw(mask_img).line(line, fill=1, width=LINE_WIDTH)
        mask = np.array(mask_img)

        tgrey = orimg[:,:] * mask
        tgrey = tgrey[tgrey != 0]

        # if mask is empty array
        if (tgrey.size == 0):
            return prevFit
        
        # check most white colour on the line
        c = int(np.max(tgrey))
        
        # draw one line
        pdraw.line(line, fill=(c), width=LINE_WIDTH)
        
        # call blending mode function by name
        out = eval('Image4Layer.' + BLEND_MODE)(draw, geimg)

        # compute similarity between original and generated image
        return np.sqrt(np.sum((orimg - np.asarray(out, dtype=int))**2.0)/(out.size[0]*out.size[1])) / 255.0 
            
    else:
        # reconstruct previous similarity 
        tSum = ((prevFit*255.0)**2.0) * geimg.size[1] * geimg.size[0]  
        
        # make a blank image for the text, initialized to transparent color
        draw = Image.new('RGBA', (deltaY, deltaX), (255,255,255,0))
        draw = draw.convert('L')
        pdraw = ImageDraw.Draw(draw)
        line = ((gen[2]-minY, gen[0]-minX),(gen[3]-minY, gen[1]-minX))
        mask_img = Image.new('1', (draw.size[0], draw.size[1]), 0)
        ImageDraw.Draw(mask_img).line(line, fill=1, width=LINE_WIDTH)
        mask = np.array(mask_img)
        
        tgrey = orimg[minX:minX+deltaX, minY:minY+deltaY] * mask
        tgrey = tgrey[tgrey != 0]
        
        # if mask is empty array
        if (tgrey.size == 0):
            return prevFit
        
        # check most white colour on the line
        c = int(np.max(tgrey))
        
        # create new line
        pdraw.line(line, fill=(c), width=LINE_WIDTH)
        partgenImg = geimg.crop((minY, minX, minY + deltaY, minX + deltaX))
        
        # call blending mode function by name
        out = eval('Image4Layer.' + BLEND_MODE)(draw, partgenImg)

        # substract similarity between previous generated image and target, add newly computed part
        newSum = tSum - np.sum((orimg[minX:minX+deltaX, minY:minY+deltaY] - np.asarray(partgenImg, dtype=int))**2.0) + np.sum((orimg[minX:minX+deltaX, minY:minY+deltaY] - np.asarray(out, dtype=int))**2.0)
        if (newSum < 0):
            return prevFit
        else:
            return np.sqrt(newSum/(geimg.size[0]*geimg.size[1])) / 255.0


''' Compute mean colour of target image part specified by line '''
def computeLineShade(orimg, gen, LINE_WIDTH):   
    line = ((gen[0][0], gen[0][1]),(gen[1][0], gen[1][1]))
    
    # create new image ("1-bit pixels, black and white", (width, height), "default color")
    mask_img = Image.new('1', (orimg.shape[1], orimg.shape[0]), 0)
    ImageDraw.Draw(mask_img).line(line, fill=1, width=LINE_WIDTH)
    mask = np.array(mask_img)

    tgrey = orimg[:,:] * mask
    tgrey = tgrey[tgrey != 0]
    if (tgrey.size == 0):
        return 255
    
    # return the lightest shade
    return int(np.max(tgrey))

''' Compute euklidean distance between two vectors '''
def dist(x,y):   
    return np.sqrt(np.sum((np.asarray(x, dtype=float) - np.asarray(y, dtype=float))**2.0))









