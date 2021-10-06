import marshal
import numpy as np
from PIL import Image
from PIL import ImageDraw
from types import FunctionType
from image4layer import Image4Layer

# ----------------- DEFINITIONS ------------------
COLOUR_BLACK = (0, 0, 0, 255) # black background colour
COLOUR_WHITE = (255, 255, 255, 255) # white background colour

# ----------------- FUNCTIONS  ------------------

''' Compute euklidean distance between two vectors '''
def dist(x,y):   
    return np.sqrt(np.sum((np.asarray(x, dtype=float) - np.asarray(y, dtype=float))**2.0))

# check if triangle based on side conditions  
def isTriangle(a, b, c):  
    if (a + b <= c) or (a + c <= b) or (b + c <= a) : 
        return False
    else: 
        return True
    
''' Compute mean colour of target image part specified by polygon '''
def computePolyColor(orimg, polygon):
    img_array = np.asarray(orimg, dtype=int)
    mask_img = Image.new('1', (orimg.shape[1], orimg.shape[0]), 0)
    ImageDraw.Draw(mask_img).polygon(polygon, outline=1, fill=1)
    mask = np.array(mask_img)

    tred = img_array[:,:,0] * mask
    tred = tred[tred != 0]
    tgreen = img_array[:,:,1] * mask
    tgreen = tgreen[tgreen != 0]
    tblue = img_array[:,:,2] * mask
    tblue = tblue[tblue != 0]
    talpha = img_array[:,:,3] * mask
    talpha = talpha[talpha != 0]
    
    mred = int(np.mean(tred))
    mgreen = int(np.mean(tgreen))
    mblue = int(np.mean(tblue))
    malpha = int(np.mean(talpha))
    c = (mred, mgreen, mblue, malpha)

    return c

''' Compute mean colour of target image part specified by line '''
def computeLineColor(orimg, line, LINE_WIDTH):
    mred = int(np.mean((orimg[int(line[0][1]),int(line[0][0]),0], orimg[int(line[1][1]),int(line[1][0]),0])))
    mgreen = int(np.mean((orimg[int(line[0][1]),int(line[0][0]),1], orimg[int(line[1][1]),int(line[1][0]),1])))
    mblue = int(np.mean((orimg[int(line[0][1]),int(line[0][0]),2], orimg[int(line[1][1]),int(line[1][0]),2])))
    malpha = int(np.mean((orimg[int(line[0][1]),int(line[0][0]),3], orimg[int(line[1][1]),int(line[1][0]),3])))
    c = (mred, mgreen, mblue, malpha)

    return c

''' Compute mean colour of target image part specified by line '''
def computeLineColor_v2(orimg, line, LINE_WIDTH, LINE_MIN_INTENSITY, LINE_MAX_INTENSITY):
    cc = int(np.min((orimg[int(line[0][1]),int(line[0][0])], orimg[int(line[1][1]),int(line[1][0])])))
    # convert to specified range
    cr = np.interp(cc, [0,255],[LINE_MIN_INTENSITY, LINE_MAX_INTENSITY])
    
    # convert back to 0-255 value, with inverse model
    c = int((1.0-cr)*255)
    
    return c

''' Compute mean colour of target image part specified by line '''
def computeLineColor_v3(orimg, gen, LINE_WIDTH):   
    line = ((gen[0][0], gen[0][1]),(gen[1][0], gen[1][1]))
    
    # create new image ("1-bit pixels, black and white", (width, height), "default color")
    mask_img = Image.new('1', (orimg.shape[1], orimg.shape[0]), 0)
    ImageDraw.Draw(mask_img).line(line, fill=1, width=LINE_WIDTH)
    mask = np.array(mask_img)

    tgrey = orimg[:,:] * mask
    tgrey = tgrey[tgrey != 0]
    if (tgrey.size == 0):
        print(line)
        return 255
    
    # check most white colour on the line
    c = int(np.max(tgrey))
    
    return c
    
''' Generate a population of triangles '''
def genTrianglepop(popsize, Amps, Space):
    lpop, lstring = Space.shape
    Newpop = np.zeros((int(popsize),int(lstring)))
    if(len(Amps.shape)<=1):
        Amps = np.reshape(Amps, (1,len(Amps)))

    for r in range(int(popsize)):
        dX = Space[1,0] - Space[0,0]
        dY = Space[1,3] - Space[0,3]
        Newpop[r,0] = np.random.uniform()*dX + Space[0,0]
        Newpop[r,3] = np.random.uniform()*dY + Space[0,3]
        for s in [0,3]:
            if(Newpop[r,s]<Space[0,s]):
                Newpop[r,s] = Space[0,s]
            if(Newpop[r,s]>Space[1,s]):
                Newpop[r,s] = Space[1,s]
        
        Newpop[r,1] = Newpop[r,0] + np.random.randint(2*Amps[0,1]+1) - Amps[0,1]
        Newpop[r,2] = Newpop[r,0] + np.random.randint(2*Amps[0,2]+1) - Amps[0,2]
        
        Newpop[r,4] = Newpop[r,3] + np.random.randint(2*Amps[0,4]+1) - Amps[0,4]
        Newpop[r,5] = Newpop[r,3] + np.random.randint(2*Amps[0,5]+1) - Amps[0,5]  
        
        while(not isTriangle(dist((Newpop[r,0], Newpop[r,3]), (Newpop[r,1], Newpop[r,4])), dist((Newpop[r,1], Newpop[r,4]), (Newpop[r,2], Newpop[r,5])), dist((Newpop[r,2], Newpop[r,5]), (Newpop[r,0], Newpop[r,3])))):
            Newpop[r,1] = Newpop[r,0] + np.random.randint(2*Amps[0,1]+1) - Amps[0,1]
            Newpop[r,2] = Newpop[r,0] + np.random.randint(2*Amps[0,2]+1) - Amps[0,2]
            
            Newpop[r,4] = Newpop[r,3] + np.random.randint(2*Amps[0,4]+1) - Amps[0,4]
            Newpop[r,5] = Newpop[r,3] + np.random.randint(2*Amps[0,5]+1) - Amps[0,5]
        
        for s in [1,2,4,5]:
            if(Newpop[r,s]<Space[0,s]):
                Newpop[r,s] = Space[0,s]
            if(Newpop[r,s]>Space[1,s]):
                Newpop[r,s] = Space[1,s]

    return Newpop


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


''' Mutate a population of triangles '''
def mutTriangle(Oldpop,factor,Amps,Space):
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
        rN = np.random.randint(3)
        r = int(np.ceil(np.random.uniform()*lpop))-1
        if (rN==0): # x1, y1
            Newpop[r,0] = Oldpop[r,0] + (2.0*np.random.uniform()-1)*Amps[0,0]
            Newpop[r,3] = Oldpop[r,3] + (2.0*np.random.uniform()-1)*Amps[0,3]
        elif (rN==1): # x2, y2
            Newpop[r,1] = Oldpop[r,1] + (2.0*np.random.uniform()-1)*Amps[0,1]
            Newpop[r,4] = Oldpop[r,4] + (2.0*np.random.uniform()-1)*Amps[0,4]
        elif (rN==2): # x3, y3
            Newpop[r,2] = Oldpop[r,2] + (2.0*np.random.uniform()-1)*Amps[0,2]
            Newpop[r,5] = Oldpop[r,5] + (2.0*np.random.uniform()-1)*Amps[0,5]
        
        while(not isTriangle(dist((Newpop[r,0], Newpop[r,3]), (Newpop[r,1], Newpop[r,4])), dist((Newpop[r,1], Newpop[r,4]), (Newpop[r,2], Newpop[r,5])), dist((Newpop[r,2], Newpop[r,5]), (Newpop[r,0], Newpop[r,3])))):
            rN = np.random.randint(3)
            if (rN==0): # x1, y1
                Newpop[r,0] = Oldpop[r,0] + (2.0*np.random.uniform()-1)*Amps[0,0]
                Newpop[r,3] = Oldpop[r,3] + (2.0*np.random.uniform()-1)*Amps[0,3]
            elif (rN==1): # x2, y2
                Newpop[r,1] = Oldpop[r,1] + (2.0*np.random.uniform()-1)*Amps[0,1]
                Newpop[r,4] = Oldpop[r,4] + (2.0*np.random.uniform()-1)*Amps[0,4]
            elif (rN==2): # x3, y3
                Newpop[r,2] = Oldpop[r,2] + (2.0*np.random.uniform()-1)*Amps[0,2]
                Newpop[r,5] = Oldpop[r,5] + (2.0*np.random.uniform()-1)*Amps[0,5]
        
        for s in range(6):
            if (Newpop[r,s]<Space[0,s]):
                Newpop[r,s] = Space[0,s]
            if (Newpop[r,s]>Space[1,s]):
                Newpop[r,s] = Space[1,s]

    return Newpop


''' Evaluate partial similarity between original and generated image '''
def evalPartialSimilarity(imglist):
    from PIL import Image, ImageDraw
    import numpy as np
    
    orimg = imglist[0]
    geimg = imglist[1]
    gen = imglist[2]
    prevFit = imglist[3]
    
    # first evaluation
    if prevFit is None:
        # evaluate full image

        # make a blank image for the text, initialized to transparent color
        draw = Image.new('RGBA', geimg.size, (255,255,255,0))
        pdraw = ImageDraw.Draw(draw)
        
        p = ((gen[3], gen[0]),(gen[4], gen[1],),(gen[5], gen[2]))
        # create new image ("1-bit pixels, black and white", (width, height), "default color")
        mask_img = Image.new('1', (geimg.size[0], geimg.size[1]), 0)
        ImageDraw.Draw(mask_img).polygon(p, outline=1, fill=1)
        mask = np.array(mask_img)
        
        tred = orimg[:,:,0] * mask
        tred = tred[tred != 0]
        tgreen = orimg[:,:,1] * mask
        tgreen = tgreen[tgreen != 0]
        tblue = orimg[:,:,2] * mask
        tblue = tblue[tblue != 0]
        talpha = orimg[:,:,3] * mask
        talpha = talpha[talpha != 0]
        
        # if mask is empty array
        if (tred.size == 0) or (tgreen.size == 0) or (tblue.size == 0) or (talpha.size == 0):
            return prevFit

        mred = int(np.mean(tred))
        mgreen = int(np.mean(tgreen))
        mblue = int(np.mean(tblue))
        malpha = int(np.mean(talpha))
        c = (mred, mgreen, mblue, malpha)

        # generate one polygon
        pdraw.polygon(p, fill=c, outline=c)
        out = Image.alpha_composite(geimg, draw)
        # compute similarity between original and generated image
        return np.sqrt(np.sum((orimg - np.asarray(out, dtype=int))**2.0)/(out.size[0]*out.size[1]*4.0)) / 255.0 
            
    else:
        # reconstruct previous similarity 
        tSum = ((prevFit*255.0)**2.0) * geimg.size[1] * geimg.size[0] * 4  
        # evaluate only part of the image
        minX = int(np.min(gen[0:3]))
        maxX = int(np.max(gen[0:3]))
        deltaX = int(maxX - minX)
        
        minY = int(np.min(gen[3:6]))
        maxY = int(np.max(gen[3:6]))
        deltaY = int(maxY - minY)
        
        if (deltaX == 0) and (deltaY != 0):
            deltaX = 1
        elif (deltaX != 0) and (deltaY == 0):
            deltaY = 1
        elif (deltaX == 0) and (deltaY == 0):
            return prevFit
        
        # make a blank image for the text, initialized to transparent color
        draw = Image.new('RGBA', (deltaY, deltaX), (255,255,255,0))
        pdraw = ImageDraw.Draw(draw)
        p = ((gen[3]-minY, gen[0]-minX),(gen[4]-minY, gen[1]-minX),(gen[5]-minY, gen[2]-minX))
        # create new image ("1-bit pixels, black and white", (width, height), "default color")
        mask_img = Image.new('1', (draw.size[0], draw.size[1]), 0)
        ImageDraw.Draw(mask_img).polygon(p, outline=1, fill=1)
        mask = np.array(mask_img)
        
        tred = orimg[minX:minX+deltaX, minY:minY+deltaY, 0] * mask
        tred = tred[tred != 0]
        tgreen = orimg[minX:minX+deltaX, minY:minY+deltaY, 1] * mask
        tgreen = tgreen[tgreen != 0]
        tblue = orimg[minX:minX+deltaX, minY:minY+deltaY, 2] * mask
        tblue = tblue[tblue != 0]
        talpha = orimg[minX:minX+deltaX, minY:minY+deltaY, 3] * mask
        talpha = talpha[talpha != 0]

        # if mask is empty array
        if (tred.size == 0) or (tgreen.size == 0) or (tblue.size == 0) or (talpha.size == 0):
            return prevFit
    
        mred = int(np.mean(tred))
        mgreen = int(np.mean(tgreen))
        mblue = int(np.mean(tblue))
        malpha = int(np.mean(talpha))
        c = (mred, mgreen, mblue, malpha)

        # vytvorime novy polygon 
        pdraw.polygon(p, fill=c, outline=c)
        # partgenImg = geimg.crop((minX, minY, minX + deltaX, minY + deltaY))
        partgenImg = geimg.crop((minY, minX, minY + deltaY, minX + deltaX))
        out = Image.alpha_composite(partgenImg, draw)
        
        # substract similarity between previous generated image and target, add newly computed part
        newSum = tSum - np.sum((orimg[minX:minX+deltaX, minY:minY+deltaY,:] - np.asarray(partgenImg, dtype=int))**2.0) + np.sum((orimg[minX:minX+deltaX, minY:minY+deltaY,:] - np.asarray(out, dtype=int))**2.0)
        if (newSum < 0):
            return prevFit
        else:
            return np.sqrt(newSum/(geimg.size[0]*geimg.size[1]*4.0)) / 255.0

''' Evaluate partial similarity between original and generated image '''
def lines_evalPartialSimilarity(imglist):
    from PIL import Image, ImageDraw
    import numpy as np
    
    orimg = imglist[0]
    geimg = imglist[1]
    gen = imglist[2]
    prevFit = imglist[3]
    LINE_WIDTH = imglist[4]
    
    # first evaluation
    if prevFit is None:
        # evaluate full image

        # make a blank image for the text, initialized to transparent color
        draw = Image.new('RGBA', geimg.size, (255,255,255,0))
        pdraw = ImageDraw.Draw(draw)
        
        line = ((gen[2], gen[0]),(gen[3], gen[1]))
        mred = int(np.mean((orimg[int(line[0][1]),int(line[0][0]),0], orimg[int(line[1][1]),int(line[1][0]),0])))
        mgreen = int(np.mean((orimg[int(line[0][1]),int(line[0][0]),1], orimg[int(line[1][1]),int(line[1][0]),1])))
        mblue = int(np.mean((orimg[int(line[0][1]),int(line[0][0]),2], orimg[int(line[1][1]),int(line[1][0]),2])))
        malpha = int(np.mean((orimg[int(line[0][1]),int(line[0][0]),3], orimg[int(line[1][1]),int(line[1][0]),3])))

        c = (mred, mgreen, mblue, malpha)

        # generate one line
        pdraw.line(line, fill=c, width=LINE_WIDTH)
        out = Image.alpha_composite(geimg, draw)
        # compute similarity between original and generated image
        return np.sqrt(np.sum((orimg - np.asarray(out, dtype=int))**2.0)/(out.size[0]*out.size[1]*4.0)) / 255.0 
            
    else:
        # reconstruct previous similarity 
        tSum = ((prevFit*255.0)**2.0) * geimg.size[1] * geimg.size[0] * 4  
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
        
        # make a blank image for the text, initialized to transparent color
        draw = Image.new('RGBA', (deltaY, deltaX), (255,255,255,0))
        pdraw = ImageDraw.Draw(draw)
        line = ((gen[2]-minY, gen[0]-minX),(gen[3]-minY, gen[1]-minX))
    
        mred = int(np.mean((orimg[int(line[0][1]),int(line[0][0]),0], orimg[int(line[1][1]),int(line[1][0]),0])))
        mgreen = int(np.mean((orimg[int(line[0][1]),int(line[0][0]),1], orimg[int(line[1][1]),int(line[1][0]),1])))
        mblue = int(np.mean((orimg[int(line[0][1]),int(line[0][0]),2], orimg[int(line[1][1]),int(line[1][0]),2])))
        malpha = int(np.mean((orimg[int(line[0][1]),int(line[0][0]),3], orimg[int(line[1][1]),int(line[1][0]),3])))

        c = (mred, mgreen, mblue, malpha)

        # create new line
        pdraw.line(line, fill=c, width=LINE_WIDTH)
        partgenImg = geimg.crop((minY, minX, minY + deltaY, minX + deltaX))
        out = Image.alpha_composite(partgenImg, draw)
        
        # substract similarity between previous generated image and target, add newly computed part
        newSum = tSum - np.sum((orimg[minX:minX+deltaX, minY:minY+deltaY,:] - np.asarray(partgenImg, dtype=int))**2.0) + np.sum((orimg[minX:minX+deltaX, minY:minY+deltaY,:] - np.asarray(out, dtype=int))**2.0)
        if (newSum < 0):
            return prevFit
        else:
            return np.sqrt(newSum/(geimg.size[0]*geimg.size[1]*4.0)) / 255.0

''' Evaluate partial similarity between original and generated image '''
def gslines_evalPartialSimilarity(imglist):
    from PIL import Image, ImageDraw
    import numpy as np
    
    orimg = imglist[0]
    geimg = imglist[1]
    gen = imglist[2]
    prevFit = imglist[3]
    LINE_WIDTH = imglist[4]
    STROKE_INTENSITY = imglist[5]
    BLEND_MODE = imglist[6]
    
    # first evaluation
    if prevFit is None:
        # evaluate full image

        # make a blank image for the text, initialized to transparent color
        draw = Image.new('RGBA', geimg.size, COLOUR_WHITE) # canvas
        draw = draw.convert('L')
        pdraw = ImageDraw.Draw(draw)
        
        line = ((gen[2], gen[0]),(gen[3], gen[1]))
        # compute color from intesity of stroke
        c = int((1.0-STROKE_INTENSITY)*255)
        
        # draw one line
        pdraw.line(line, fill=(c), width=LINE_WIDTH)
        
        # call blending mode function by name
        out = eval('Image4Layer.' + BLEND_MODE)(draw, geimg)

        # compute similarity between original and generated image
        return np.sqrt(np.sum((orimg - np.asarray(out, dtype=int))**2.0)/(out.size[0]*out.size[1])) / 255.0 
            
    else:
        # reconstruct previous similarity 
        tSum = ((prevFit*255.0)**2.0) * geimg.size[1] * geimg.size[0]  
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
        
        # make a blank image for the text, initialized to transparent color
        draw = Image.new('RGBA', (deltaY, deltaX), (255,255,255,0))
        draw = draw.convert('L')
        pdraw = ImageDraw.Draw(draw)
        line = ((gen[2]-minY, gen[0]-minX),(gen[3]-minY, gen[1]-minX))
    
        # compute color from intesity of stroke
        c = int((1.0-STROKE_INTENSITY)*255)

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

''' Evaluate partial similarity between original and generated image '''
def gslines_evalPartialSimilarity_v2(imglist):
    from PIL import Image, ImageDraw
    import numpy as np
    
    orimg = imglist[0]
    geimg = imglist[1]
    gen = imglist[2]
    prevFit = imglist[3]
    LINE_WIDTH = imglist[4]
    LINE_MIN_INTENSITY = imglist[5]
    LINE_MAX_INTENSITY = imglist[6]
    BLEND_MODE = imglist[7]
    
    # first evaluation
    if prevFit is None:
        # evaluate full image

        # make a blank image for the text, initialized to transparent color
        draw = Image.new('RGBA', geimg.size, COLOUR_WHITE) # canvas
        draw = draw.convert('L')
        pdraw = ImageDraw.Draw(draw)
        
        line = ((gen[2], gen[0]),(gen[3], gen[1]))
        cc = int(np.min((orimg[int(line[0][1]),int(line[0][0])], orimg[int(line[1][1]),int(line[1][0])])))
        
        # convert to specified range
        cr = np.interp(cc, [0,255],[LINE_MIN_INTENSITY, LINE_MAX_INTENSITY])
        
        # convert back to 0-255 value, with inverse model
        c = int((1.0-cr)*255)
        
        # draw one line
        pdraw.line(line, fill=(c), width=LINE_WIDTH)
        
        # call blending mode function by name
        out = eval('Image4Layer.' + BLEND_MODE)(draw, geimg)

        # compute similarity between original and generated image
        return np.sqrt(np.sum((orimg - np.asarray(out, dtype=int))**2.0)/(out.size[0]*out.size[1])) / 255.0 
            
    else:
        # reconstruct previous similarity 
        tSum = ((prevFit*255.0)**2.0) * geimg.size[1] * geimg.size[0]  
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
        
        # make a blank image for the text, initialized to transparent color
        draw = Image.new('RGBA', (deltaY, deltaX), (255,255,255,0))
        draw = draw.convert('L')
        pdraw = ImageDraw.Draw(draw)
        line = ((gen[2]-minY, gen[0]-minX),(gen[3]-minY, gen[1]-minX))
        cc = int(np.min((orimg[int(line[0][1]),int(line[0][0])], orimg[int(line[1][1]),int(line[1][0])])))
        # convert to specified range
        cr = np.interp(cc, [0,255],[LINE_MIN_INTENSITY, LINE_MAX_INTENSITY])
        
        # convert back to 0-255 value, with inverse model
        c = int((1.0-cr)*255)
        
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

''' Evaluate partial similarity between original and generated image '''
def gslines_evalPartialSimilarity_v3(imglist):
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
        
        #check most white colour on the line
        #x_list = list(range(minX, maxX + 1))
        #y_list = np.linspace(minY, maxY, deltaX + 1)
        ##y_list = y_list.astype(int)
        #cc_list = orimg[y_list,x_list]
        #c = int(max(cc_list))
        
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

''' Sequentially evaluate fitness function '''
def sequentialFitness(Pop, orimg, geimg, fitOld):
    if(len(Pop.shape)<=1):
        Pop = np.reshape(Pop, (1,len(Pop)))
    lpop, lstring = Pop.shape
    Fit = np.zeros((lpop,))
    
    for i in range(lpop):
        G = Pop[i,:]
        Fit[i] = evalPartialSimilarity(list([orimg, geimg, G, fitOld]))     
    return Fit

''' Sequentially evaluate fitness function '''
def lines_sequentialFitness(Pop, orimg, geimg, fitOld, LINE_WIDTH):
    if(len(Pop.shape)<=1):
        Pop = np.reshape(Pop, (1,len(Pop)))
    lpop, lstring = Pop.shape
    Fit = np.zeros((lpop,))
    
    for i in range(lpop):
        G = Pop[i,:]
        Fit[i] = lines_evalPartialSimilarity(list([orimg, geimg, G, fitOld, LINE_WIDTH]))     
    return Fit

''' Sequentially evaluate fitness function '''
def gslines_sequentialFitness(Pop, orimg, geimg, fitOld, LINE_WIDTH, STROKE_INTENSITY, BLEND_MODE):
    if(len(Pop.shape)<=1):
        Pop = np.reshape(Pop, (1,len(Pop)))
    lpop, lstring = Pop.shape
    Fit = np.zeros((lpop,))
    
    for i in range(lpop):
        G = Pop[i,:]
        Fit[i] = gslines_evalPartialSimilarity(list([orimg, geimg, G, fitOld, LINE_WIDTH, STROKE_INTENSITY, BLEND_MODE]))     
    return Fit

''' Sequentially evaluate fitness function '''
def gslines_sequentialFitness_v2(Pop, orimg, geimg, fitOld, LINE_WIDTH, LINE_MIN_INTENSITY, LINE_MAX_INTENSITY, BLEND_MODE):
    if(len(Pop.shape)<=1):
        Pop = np.reshape(Pop, (1,len(Pop)))
    lpop, lstring = Pop.shape
    Fit = np.zeros((lpop,))
    
    for i in range(lpop):
        G = Pop[i,:]
        Fit[i] = gslines_evalPartialSimilarity_v2(list([orimg, geimg, G, fitOld, LINE_WIDTH, LINE_MIN_INTENSITY, LINE_MAX_INTENSITY, BLEND_MODE]))     
    return Fit


''' Sequentially evaluate fitness function '''
def gslines_sequentialFitness_v3(Pop, orimg, geimg, fitOld, LINE_WIDTH, BLEND_MODE):
    if(len(Pop.shape)<=1):
        Pop = np.reshape(Pop, (1,len(Pop)))
    lpop, lstring = Pop.shape
    Fit = np.zeros((lpop,))
    
    for i in range(lpop):
        G = Pop[i,:]
        Fit[i] = gslines_evalPartialSimilarity_v3(list([orimg, geimg, G, fitOld, LINE_WIDTH, BLEND_MODE]))     
    return Fit

''' Evaluate the fitness function in parallel '''
def parallelFitness(Pop, pool, orimg, geimg, fitOld):
    # for all individuals in population
    err = [pool.apply_async(*make_applicable(evalPartialSimilarity, [orimg, geimg, gen, fitOld])) for gen in Pop] # paralelne vyhodnotenie
    return np.squeeze(np.array([result.get(timeout=10) for result in err]))

'''Functions for parallelization (Windows OS only)'''
def _applicable(*args, **kwargs):
  name = kwargs['__pw_name']
  code = marshal.loads(kwargs['__pw_code'])
  gbls = globals()
  defs = marshal.loads(kwargs['__pw_defs'])
  clsr = marshal.loads(kwargs['__pw_clsr'])
  fdct = marshal.loads(kwargs['__pw_fdct'])
  func = FunctionType(code, gbls, name, defs, clsr)
  func.fdct = fdct
  del kwargs['__pw_name']
  del kwargs['__pw_code']
  del kwargs['__pw_defs']
  del kwargs['__pw_clsr']
  del kwargs['__pw_fdct']
  return func(*args, **kwargs)


def make_applicable(f, *args, **kwargs):
  if not isinstance(f, FunctionType): raise ValueError('Argument must be a function')
  kwargs['__pw_name'] = f.__name__  
  kwargs['__pw_code'] = marshal.dumps(f.__code__)   
  kwargs['__pw_defs'] = marshal.dumps(f.__defaults__)  
  kwargs['__pw_clsr'] = marshal.dumps(f.__closure__)  
  kwargs['__pw_fdct'] = marshal.dumps(f.__dict__)   
  return _applicable, args, kwargs

def _mappable(x):
  x,name,code,defs,clsr,fdct = x
  code = marshal.loads(code)
  gbls = globals()
  defs = marshal.loads(defs)
  clsr = marshal.loads(clsr)
  fdct = marshal.loads(fdct)
  func = FunctionType(code, gbls, name, defs, clsr)
  func.fdct = fdct
  return func(x)

def make_mappable(f, iterable):
  if not isinstance(f, FunctionType): raise ValueError('Argument must be a function')
  name = f.__name__    
  code = marshal.dumps(f.__code__)  
  defs = marshal.dumps(f.__defaults__) 
  clsr = marshal.dumps(f.__closure__) 
  fdct = marshal.dumps(f.__dict__) 
  return _mappable, ((i,name,code,defs,clsr,fdct) for i in iterable)





