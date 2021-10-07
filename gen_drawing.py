# import libraries
import time
import datetime
import sys, os
import numpy as np
import matplotlib.pyplot as plt
# add a folder with a library to the path
sys.path.append(".")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "./geneticlib")
# import functions from the genetic library
from genetic.utils import *
from PIL import Image, ImageDraw, ImageOps
from image4layer import Image4Layer

# ------------------------------ USER PARAMETERS --------------------------
img_str = "kingfisher.jpg"  # image to load (from ./images folder)
basewidth = 256 # output size of generated image 
deterministic_mode = True  # reproducible results [True, False]
generate_gif = True # generate animation [True, False]
deterministic_seed = 42 # seed for pseudo-random generator
N = 6000 # number of objects in created image
# genetic optimization
NEvo = 10 # number of evolution steps per one optimized object 
MAX_BUFF = 5  # stopping evolution if there are no changes (MAX_BUFF consecutive evolution steps)
MAX_ADDMUT = 5 # [%] - maximum aditive mutation range
LINE_WIDTH = 2 # [px] line width
MLTPL_EVO_PARAMS = 1 # parameter multiplier
BLEND_MODE = "darken" # available options: ["normal", "multiply", "screen", "overlay", "darken", "lighten", "color_dodge", "color_burn", "hard_light", "soft_light", "difference", "exclusion", "hue", "saturation", "color", "luminosity", "vivid_light", "pin_light", "linear_dodge", "subtract"]
# -------------------------------------------------------------------------

"""
 Optimized parameters: x1,x2,y1,y2 (for one line)

   -------> y
   | °°°°°°°°°°°°°°°
   | °             °
   | °             °
   v °             °
   x °             °
     °°°°°°°°°°°°°°°

 x1,x2 <0, image_height> - vector of x positions
 y1,y2 <0, image_width> - vector of y positions
"""
'''
NOTE:
    - numpy -> works with the image as a dimensional tensor (H, W, D)
      [ the higher axis designation and the tensor correspond to this (x,y,d) ]
    - Pillow -> works with the image as a dimensional tensor (W, H, D)
      [ this results in a discrepancy and a change of labeling, x and y in code according to object type ]
'''


# if deterministic mode, use specified seed for reproducible results
if (deterministic_mode):
    np.random.seed(deterministic_seed)
    
# rendering settings (font and style)
plt.style.use('seaborn-paper')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.linewidth'] = 0.1 # frame boundaries in graphs

# load image and convert it to greyscale
orig_img = Image.open("./images/" + img_str).convert('L')
# resize image to specified size with aspect ratio preserved
wpercent = (basewidth/float(orig_img.size[0]))
hsize = int((float(orig_img.size[1])*float(wpercent)))
orig_img = orig_img.resize((basewidth,hsize), Image.ANTIALIAS)

# convert to numpy array
orig_img = np.asarray(orig_img, dtype=int)
# start the timer
start_time = time.time()
# --------------------------------------------------------

# generate empty image with background colour
gen_img = Image.new('RGBA', (orig_img.shape[1], orig_img.shape[0]), COLOUR_WHITE) # canvas
gen_img = gen_img.convert('L') # canvas

# definition of search space limitations (for one polygon only)
OneSpace = np.concatenate((np.zeros((1,4)),           # mininum
                        np.array([[orig_img.shape[0]-1, orig_img.shape[0]-1, orig_img.shape[1]-1, orig_img.shape[1]-1]])), axis=0)  # maximum
# range of changes for the additive mutation
Amp = OneSpace[1,:]*(MAX_ADDMUT/100.0) 
# results to be saved
lpoly = np.zeros((N,6)) # (x1,x2,y1,y2,stroke,fitness)
data = list() # list of fitness values
# we start from the white canvas to which we add line segments
rfit = None # initial fitness value
buffer = 0 # auxiliary variable to stop evolution if no changes occur
count = 1 # number of objects in final image
images = [] # list of image used for animation process
if generate_gif:
    images.append(gen_img)

# repeat, until we reached specified number of line segments
while(count<=N):
    # initial population generation
    NewPop = genLinespop(24, Amp, OneSpace)
    # first fitness evaluation
    fitness = evalFitness(NewPop, orig_img, gen_img, rfit, LINE_WIDTH, BLEND_MODE)
    
    # start of genetic optimization process
    for i in range(NEvo):  # high enough value (we expect an early stop)
        OldPop = np.copy(NewPop)    # save population and fitness from previous generation
        fitnessOld = np.copy(fitness) 
        PartNewPop1, PartNewFit1 = selbest(OldPop, fitness, [3*MLTPL_EVO_PARAMS,2*MLTPL_EVO_PARAMS,1*MLTPL_EVO_PARAMS])    # select best polygons
        PartNewPop2, PartNewFit2 = selsus(OldPop, fitness, 18*MLTPL_EVO_PARAMS)
        PartNewPop2 = mutLine(PartNewPop2, 0.2, Amp, OneSpace)   # additive mutation
        NewPop = np.concatenate((PartNewPop1, PartNewPop2), axis=0) # create new population
        fitness = evalFitness(NewPop, orig_img, gen_img, rfit, LINE_WIDTH, BLEND_MODE)
        if (np.min(fitness) == np.min(fitnessOld)):
            buffer += 1 # if we stagnate start with counting
        else: 
            buffer = 0  # if the solution has improved, continue evolution
        # if we have exceeded the maximum limit, we will stop evolution
        if (buffer >= MAX_BUFF):
            break
    
    # add the best line segment in the image and continue evolution
    priesenie, rfitnew = selbest(NewPop, fitness, [1])
    
    if(rfit is None):
        rfit = 1e6 # safe big value
    # draw line segment only if it improves fitness
    if(rfitnew < rfit):
        rfit = rfitnew
        data.append(rfit) # save line segment info
        draw = Image.new('RGBA', gen_img.size, (255,255,255,0))
        draw = draw.convert('L')
        pdraw = ImageDraw.Draw(draw)
        p = ((priesenie[0,2], priesenie[0,0]),(priesenie[0,3], priesenie[0,1]))
        c = computeLineShade(orig_img, p, LINE_WIDTH)
        
        pdraw.line(p, fill=(c), width=LINE_WIDTH)
        gen_img = eval('Image4Layer.' + BLEND_MODE)(draw, gen_img)
        print("# " + str(count) + " Fitness: " + str(rfit))
        lpoly[count-1,:] = np.concatenate(np.array((priesenie[0,2], priesenie[0,3], priesenie[0,0], priesenie[0,1], 255-c, rfit[0])).reshape(6,1))
        count += 1 # increment number of objects
        if generate_gif:
            images.append(gen_img.convert('P'))
        
# create new graph
fig, ax = plt.subplots()
plt.plot(data, 'b', linewidth=0.5)
plt.title('Image vectorization via genetic evolution')
plt.xlabel('Number of generations')
plt.ylabel('Fitness')
plt.xlim(left=0)
# grid and display settings
plt.box(True)
ax.set_axisbelow(True)
ax.minorticks_on()
ax.grid(which='major', linestyle='--', linewidth='0.5')
ax.grid(which='minor', linestyle='-.', linewidth='0.05', alpha=0.1)
# display the resulting graph and list the solution found
plt.show()

# find out the final solution
riesenie, rfit = selbest(NewPop, fitness, [1])
print("Final fitness value: " + str(rfit[0]))
print("--- Evolution lasted: %s seconds ---" % (time.time() - start_time))

# save generated image to the file
uniq_filename = str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.')
out_path = u"./results/{}.png".format(img_str.rsplit('.', 1)[0] + '_' + uniq_filename)
gen_img.save(out_path, dpi=(600,600))
gen2 = gen_img.convert('RGB')
gen2.save(u"./results/{}.pdf".format(img_str.rsplit('.', 1)[0] + '_' + uniq_filename))
# save solution to csv file
np.savetxt("./results/" + img_str.rsplit('.', 1)[0] + '_' + uniq_filename + ".csv", lpoly, delimiter=";")
# save animation to the file
if generate_gif:
    images[0].save(u"./results/{}.gif".format(img_str.rsplit('.', 1)[0] + '_' + uniq_filename), save_all=True, append_images=images[1::2], optimize=False, duration=2, loop=0)






