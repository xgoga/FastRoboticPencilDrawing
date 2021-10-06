"""
Genetic Algorithm Toolbox for Python3 ver.1.2
based on Genetic Algorithm Toolbox for Matlab(R)
----------------------------------------
I.Sekaj, 10/2002 (re-implemented by J.Goga 6/2020)
Department of Artificial Intelligence
Institute of Robotics and Cybernetics (IRC)
Faculty of Electrical Engineering and Information Technology
Slovak University of Technology
Ilkovicova 3, 812 19 Bratislava, Slovak Republic
E-mail: ivan.sekaj@stuba.sk, jozef.goga@stuba.sk

Description:
------------
The Toolbox can be used for solving of real-coded search and optimizing problems.
The Toolbox functions minimizes the objective function, maximizing problems can be 
solved as complementary tasks.


List of functions:
------------------
[*] around - intermediate crossover
[*] change - elimination of duplicite strings in the population
[*] crosgrp - crossover between more parents
[*] crosord - crossover of permutation type string
[*] crossov - multipoint crossover
[*] genrpop - generating of a random real-coded population
[*] invfit - inversion of the objective (fitness) function
[*] invord - inversion of order of a substring 
[*] muta - aditive mutation with uniform probability distribution
[*] mutm - multiplicative mutation
[*] mutn - aditive mutation with normal probability distribution
[*] mutx - simple mutation
[*] selbest - selection of best strings
[*] seldiv - selection based on the maximum diversity measure
[*] selfxd - selection based on the diversity measure and fitness
[*] selrand - random selection 
[*] selwrul - rulette wheel selection
[*] selsort - selection and sorting of best string of a population according fitness
[*] selsus - stochastic universal selection
[*] seltourn - tournament selection
[*] shake - random shaking of the string order in the population
[*] swapgen - mutation of the gene-order in strings
[*] swappart- exchange of the order of two substrings in the strings

Fitness functions:
------------------
[*] eggholder - Egg holder objective function
[*] testfn1 - Quadratic objective function, unimodal optimisation problem
[*] testfn2 - Rastigin's objective function
[*] testfn3 - Schwefel's objective function
[*] testfn6 - Search for the in string H defined combination of 8 integer numbers
[*] testfn8 - Rosenbrock objective function
[*] testfn9 - Griewank objective function
"""

# necessary imports
import numpy as np

'''
 genrpop - generating of a random real-coded population

	Description: 
	The function generates a population of random real-coded strings
	which items (genes) are limited by a two-row matrix Space. The first
	row of the matrix Space consists of the lower limits and the second row 
	consists of the upper limits of the possible values of genes. 
	The length of the string is equal to the length of rows of the matrix Space.
	

	Syntax: 

	Newpop=genrpop(popsize, Space)
           
	   Newpop - random generated population
	   popsize - required number of strings in the population
	   Space - 2-row matrix, which 1-st row is the vector of the lower limits
		   and the 2-nd row is the vector of the upper limits of the
		   gene values.


 I.Sekaj, 5/2000
'''
def genrpop(popsize, Space):
    lpop, lstring = Space.shape
    Newpop = np.zeros((int(popsize),int(lstring)))
    
    for r in range(int(popsize)):
        for s in range(int(lstring)):
            d = Space[1,s] - Space[0,s]
            Newpop[r,s] = np.random.uniform()*d + Space[0,s]
            
            if(Newpop[r,s]<Space[0,s]):
                Newpop[r,s] = Space[0,s]
            if(Newpop[r,s]>Space[1,s]):
                Newpop[r,s] = Space[1,s]
    
    return Newpop

'''
 invfit - inversion of the objective (fitness) function

	Description:
	The function calculates the complement function to the objective function
	as follows:

	Newobj=(max(Oldobj)-Oldobj)+min(Oldobj)

	In this way it is possible to convert a maximization problem to a minimization
	one for the need of the GA-toolbox.


	Syntax:

	Newobj=invfit(Oldobj)

	Newobj - vector of the complementary objective function
	Oldobj - vector of the old objective function


 I.Sekaj, 5/2000
'''
def invfit(Oldfv):
    return (np.max(Oldfv)-Oldfv) + np.min(Oldfv)


'''
 INVORD - inversion of order of a substring

 V nahodne vybranych retazcoch populacie invertuje poradie genov
 vzdy jedneho nahodne vybraneho subretazca 

 Pouzitie:  Newpop=invord(Oldpop,rate)

            Oldpop - stara populacia
            rate - pocetnost vyskytu modifikovanych retazcov v populacii (0;1) 

 I.Sekaj, 8/2001
'''
def invord(Oldpop, rate):
    lpop, lstring = Oldpop.shape
    
    if rate>1:
        rate = 1
    if rate<0:
        rate = 0
        
    n = np.ceil(lpop*rate*np.random.uniform())
    Newpop = np.copy(Oldpop)
    
    for i in range(int(n)):
        r = int(np.ceil(np.random.uniform()*lpop)) # vybrany retazec
        p1 = int(np.ceil(0.001+np.random.uniform()*(lstring-1)))
        p2 = int(np.ceil(0.001+np.random.uniform()*(lstring-p1)) + p1)
        if (p1==lstring):
            p1 = lstring-1
        if (p2>lstring):
            p2 = lstring
        
        for j in range(int(p1),int(p2+1)):
            k = int(p2 - (j-p1))
            Newpop[r-1,j-1] = Oldpop[r-1,k-1]

        Oldpop = np.copy(Newpop)
        
    return Newpop

'''
 muta - aditive mutation

	Description:
	The function mutates the population of strings with the intensity
	proportional to the parameter rate from interval <0;1>. Only a few genes  
	from a few strings are mutated in the population. The mutations are realized
	by addition or substraction of random real-numbers to the mutated genes. The 
	absolute values of the added constants are limited by the vector Amp. 
	Next the mutated strings are limited using boundaries defined in 
	a two-row matrix Space. The first row of the matrix represents the lower 
	boundaries and the second row represents the upper boundaries of corresponding 
	genes.


	Syntax: 

	Newpop=muta(Oldpop,rate,Amp,Space)

	       Newpop - new, mutated population
	       Oldpop - old population
	       Amp   -  vector of absolute values of real-number boundaries
	       Space  - matrix of gene boundaries in the form: 
	                [real-number vector of lower limits of genes
                        real-number vector of upper limits of genes];
	       rate   - mutation intensity, 0 =< rate =< 1

 I.Sekaj, 5/2000
'''
def muta(Oldpop,factor,Amps,Space):
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
        r = int(np.ceil(np.random.uniform()*lpop))-1
        s = int(np.ceil(np.random.uniform()*lstring))-1
        Newpop[r,s] = Oldpop[r,s] + (2.0*np.random.uniform()-1)*Amps[0,s];
        if (Newpop[r,s]<Space[0,s]):
            Newpop[r,s] = Space[0,s]
        if (Newpop[r,s]>Space[1,s]):
            Newpop[r,s] = Space[1,s]

    return Newpop


'''
 mutm - multiplicative mutation

	Description:
	The function mutates the population of strings with the intensity
	proportional to the parameter rate from interval <0;1>. Only a few genes  
	from a few strings are mutated in the population. The mutations are realized
	by multiplication of the mutated genes with real numbers from bounded intervals.
	The intervals are defined in the two-row matrix Amps. The first row of the 
	matrix represents the lower boundaries and the second row represents the upper 
	boundaries of the multiplication constants. Next the mutated strings
	are limited using boundaries defined in a similar two-row matrix Space. 


	Syntax: 

	Newpop=mutm(Oldpop,rate,Amps,Space)

	       Newpop - new mutated population
	       Oldpop - old population
	       Amps   - matrix of multiplicative constant boundaries in the form:
			[real-number vector of lower limits;
                        real-number vector of upper limits];
	       Space  - matrix of gene boundaries in the form: 
	                [real-number vector of lower limits of genes;
                        real-number vector of upper limits of genes];
	       rate   - mutation intensity, 0 =< rate =< 1


 I.Sekaj, 5/2000
'''
def mutm(Oldpop,factor,Amps,Space):
    lpop, lstring = Oldpop.shape
    
    if (factor>1):
        factor = 1
    if (factor<0):
        factor = 0
    
    n = int(np.ceil(lpop*lstring*factor*np.random.uniform()))
    Newpop = np.copy(Oldpop)
    
    for i in range(n):
        r = int(np.ceil(np.random.uniform()*lpop))-1
        s = int(np.ceil(np.random.uniform()*lstring))-1
        d = Amps[1,s] - Amps[0,s]
        Newpop[r,s] = Oldpop[r,s]*(np.random.uniform()*d + Amps[0,s]);
        if (Newpop[r,s]<Space[0,s]):
            Newpop[r,s] = Space[0,s]
        if (Newpop[r,s]>Space[1,s]):
            Newpop[r,s] = Space[1,s]

    return Newpop

'''
 mutn                - aditívna mutácia s normálnym rozdelením pravdepodobnosti

	Charakteristika:
	Funkcia zmutuje populáciu reťazcov s intenzitou úmernou parametru 
	rate (z rozsahu od 0 do 1). Mutovaných je len niekoľko génov v rámci 
	celej populácie. Mutácie vzniknú pripočítaním alebo odpočítaním 
	náhodných čísel ohraničených veľkostí k pôvodným hodnotám náhodne 
	vybraných génov celej populácie. Absolútne hodnoty prípustných veľkostí 
	aditívnych mutácií sú ohraničené hodnotami vektora Amp. Po tejto operácii 
	sú ešte výsledné hodnoty génov ohraničené (saturované) na hodnoty prvkov 
	matice Space. Prvý riadok matíce určuje dolné ohraničenia a druhý riadok 
	horné ohraničenia jednotlivých génov. 


	Syntax: 

	Newpop=mutn(Oldpop,rate,Amp,Space)

	       Newpop - nová, zmutovaná populácia
	       Oldpop - stará populácia
	       Amp    - vektor ohraničení prípustných aditívnych hodnôt mutácií
	       Space  - matica obmedzení, ktorej 1.riadok je vektor  minimálnych a 2.  
	                riadok je vektor maximálnych prípustných mutovaných hodnôt
	       rate   - miera početnosti mutovania génov v populácii (od 0 do 1)

 I.Sekaj, 5/2000
'''
def mutn(Oldpop,factor,Amps,Space):
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
        r = int(np.ceil(np.random.uniform()*lpop))-1
        s = int(np.ceil(np.random.uniform()*lstring))-1
        Newpop[r,s] = Oldpop[r,s] + (np.random.randn()/4.0)*Amps[0,s];
        if (Newpop[r,s]<Space[0,s]):
            Newpop[r,s] = Space[0,s]
        if (Newpop[r,s]>Space[1,s]):
            Newpop[r,s] = Space[1,s]

    return Newpop

'''
 mutx - simple mutation

	Description:
	The function mutates the population of strings with the intensity
	proportional to the parameter rate from interval <0;1>. Only a few genes  
	from a few strings are mutated in the population. The mutated values are
	selected from the bounded real-number space, which is defined by the two-row 
	matrix Space. The first row of the matrix represents the lower boundaries and the 
	second row represents the upper boundaries of corresponding genes in the strings. 


	Syntax: 

	Newpop=mutx(Oldpop,rate,Space)

	       Newpop - new mutated population
	       Oldpop - old population
	       Space  - matrix of boundaries in the form: [vector of lower limits of genes;
                                                          vector of upper limits of genes];
	       rate   - mutation intensity, 0 =< rate =< 1


 I.Sekaj, 5/2000
'''
def mutx(Oldpop,factor,Space):
    lpop, lstring = Oldpop.shape
    
    if (factor>1):
        factor = 1
    if (factor<0):
        factor = 0
    
    n = int(np.ceil(lpop*lstring*factor*np.random.uniform()))
    Newpop = np.copy(Oldpop)
    
    for i in range(n):
        r = int(np.ceil(np.random.uniform()*lpop))-1
        s = int(np.ceil(np.random.uniform()*lstring))-1
        d = Space[1,s] - Space[0,s]
        Newpop[r,s] = np.random.uniform()*d + Space[0,s]
        if (Newpop[r,s]<Space[0,s]):
            Newpop[r,s] = Space[0,s]
        if (Newpop[r,s]>Space[1,s]):
            Newpop[r,s] = Space[1,s]

    return Newpop

'''
 crossov - multipoint crossover

	Description:
	The function creates a new population of strings, which rises after
	1- to 4-point crossover operation of all (couples of) strings of the old
	population. The selection of strings into couples is either random or
	the neighbouring strings are selected, depending on the parameter sel.


	Syntax: 

	Newpop=crossov(Oldpop,num,sel)

	Newpop - new population 
	Oldpop - old population
	         num - the number of crossover points from 1 to 4
	         sel - type of the string-couple selection:
	               0 - random 
	               1 - neighbouring strings in the population 


 I.Sekaj, 2/2001
'''
def crossov(Oldpop,pts,sel):
    Newpop = np.copy(Oldpop)
    lpop, lstring = Oldpop.shape
    flag = np.zeros((lpop,))
    num = int(np.floor(lpop/2))
    i = 1

    for cyk in range(1,num+1):
        if (sel==0):
            while(flag[i-1]!=0):
                i += 1
            flag[i-1] = 1
            j = int(np.ceil(lpop*np.random.uniform()))
            while (flag[j-1]!=0):
                j = int(np.ceil(lpop*np.random.uniform()))
            flag[j-1] = 2
        elif (sel==1):
            i = int(2*cyk - 1)
            j = int(i + 1)

        if (pts>4):
            pts = 4
        n = lstring*(1-(pts-1)*0.15)
        p = int(np.ceil(np.random.rand()*n))
        
        if(p==lstring):
            p = int(lstring-1)
        v = list()
        v.append(p)
        
        for k in range(1,pts):
            h = int(np.ceil(np.random.rand()*n))
            if (h==1):
                h = 2
            p = p + h
            if (p>=lstring):
                break
            v.append(p)

        lv = len(v)
        if (lv==4):
            Newpop[i-1,:] = np.concatenate( (np.concatenate( (np.concatenate( (np.concatenate((Oldpop[i-1,0:(v[0]+1)], Oldpop[j-1,(v[0]+1):(v[1]+1)]), axis=0), Oldpop[i-1,(v[1]+1):(v[2]+1)]), axis=0), Oldpop[j-1,(v[2]+1):(v[3]+1)]), axis=0), Oldpop[i-1,(v[3]+1):lstring]), axis=0)
            Newpop[j-1,:] = np.concatenate( (np.concatenate( (np.concatenate( (np.concatenate((Oldpop[j-1,0:(v[0]+1)], Oldpop[i-1,(v[0]+1):(v[1]+1)]), axis=0), Oldpop[j-1,(v[1]+1):(v[2]+1)]), axis=0), Oldpop[i-1,(v[2]+1):(v[3]+1)]), axis=0), Oldpop[j-1,(v[3]+1):lstring]), axis=0)
        elif (lv==3):
            Newpop[i-1,:] = np.concatenate( (np.concatenate( (np.concatenate((Oldpop[i-1,0:(v[0]+1)], Oldpop[j-1,(v[0]+1):(v[1]+1)]), axis=0), Oldpop[i-1,(v[1]+1):(v[2]+1)]), axis=0),  Oldpop[j-1,(v[2]+1):lstring]), axis=0)
            Newpop[j-1,:] = np.concatenate( (np.concatenate( (np.concatenate((Oldpop[j-1,0:(v[0]+1)], Oldpop[i-1,(v[0]+1):(v[1]+1)]), axis=0), Oldpop[j-1,(v[1]+1):(v[2]+1)]), axis=0),  Oldpop[i-1,(v[2]+1):lstring]), axis=0)
        elif (lv==2):
            Newpop[i-1,:] = np.concatenate( (np.concatenate((Oldpop[i-1,0:(v[0]+1)], Oldpop[j-1,(v[0]+1):(v[1]+1)]), axis=0), Oldpop[i-1,(v[1]+1):lstring]), axis=0)
            Newpop[j-1,:] = np.concatenate( (np.concatenate((Oldpop[j-1,0:(v[0]+1)], Oldpop[i-1,(v[0]+1):(v[1]+1)]), axis=0), Oldpop[j-1,(v[1]+1):lstring]), axis=0)
        else:
            Newpop[i-1,:] = np.concatenate((Oldpop[i-1,0:(v[0]+1)], Oldpop[j-1,(v[0]+1):lstring]), axis=0)
            Newpop[j-1,:] = np.concatenate((Oldpop[j-1,0:(v[0]+1)], Oldpop[i-1,(v[0]+1):lstring]), axis=0)
    
    return Newpop

'''
 CROSORD - crossover of permutation type string

 Vytvori novu populaciu retazcov ktora vznikne skrizenim vsetkych 
 retazcov starej populacie 2-bodovym krizenim permutacneho typu. 
 Krizene su vsetky retazce (ak je ich parny pocet).

 Pouzitie:  Newpop=crosord(Oldpop,sel)

            Oldpop - stara populacia
            sel - sposob vyberu dvojic: 0 - nahodny, 1 - susedne dvojice v populacii 

 I.Sekaj, 8/2001
'''
def crosord(Oldpop,sel):
    lpop, lstring = Oldpop.shape
    Newpop = np.copy(Oldpop)
    flag = np.zeros((lpop,))
    num = int(np.floor(lpop/2.0))
    i = 1
    
    for cyk in range(1,num+1): # vytvaranie dvojic retazcov s indexami i a j
        if (sel==0):    # nahodne parovanie retazcov
            while (flag[i-1]!=0):
                i += 1
            flag[i-1] = 1
            j = int(np.ceil(lpop*np.random.uniform()))
            while (flag[j-1]!=0):
                j = int(np.ceil(lpop*np.random.uniform()))
            flag[j-1] = 2
        elif (sel==1):
            i = int(2*cyk-1)
            j = i + 1
        
        p1 = int(np.ceil(0.0001+np.random.uniform()*(lstring-2))) # pozicie delenia retazca p1, p2
        p2 = int(np.ceil(0.0001+np.random.uniform()*(lstring-p1)) + p1)
        
        if (p2>lstring):
            p2 = lstring
        if (p1==1 and p2>=(lstring-1)):
            p2 = lstring - 2
        if (p1==2 and p2>=lstring):
            p2 = lstring - 1
        
        nxch = lstring - (p2-p1+1)

        try:
            Newpop[i-1,(p1-1):p2] = Oldpop[i-1,(p1-1):p2]
            Newpop[j-1,(p1-1):p2] = Oldpop[j-1,(p1-1):p2]
        except IndexError:
            Newpop = np.concatenate((Newpop, np.zeros((lpop,p2-Newpop.shape[1]))), axis=1)
            Newpop[i-1,(p1-1):p2] = Oldpop[i-1,(p1-1):p2]
            Newpop[j-1,(p1-1):p2] = Oldpop[j-1,(p1-1):p2]
        
        pos = 1
        pall = 0
        while (pall==0):
            for k2 in range(1,lstring+1):
                if (pos==p1):
                    pos = p2 + 1
                nasiel = 0
                for k1 in range(p1,p2+1):
                    if (Oldpop[i-1,k1-1]==Oldpop[j-1,k2-1]):
                        nasiel = 1
                if(nasiel==0):
                    try:
                        Newpop[i-1,pos-1] = Oldpop[j-1,k2-1]
                    except IndexError:
                        Newpop = np.concatenate((Newpop, np.zeros((lpop,pos-Newpop.shape[1]))), axis=1)
                        Newpop[i-1,pos-1] = Oldpop[j-1,k2-1]
                    pos += 1
                    if (pos>=(nxch+1)):
                        pall=1

        pos = 1
        pall = 0
        while (pall==0):
            for k2 in range(1,lstring+1):
                if (pos==p1):
                    pos = p2 + 1
                nasiel = 0
                for k1 in range(p1,p2+1):
                    if (Oldpop[j-1,k1-1]==Oldpop[i-1,k2-1]):
                        nasiel = 1
                if(nasiel==0):
                    try:
                        Newpop[j-1,pos-1] = Oldpop[i-1,k2-1]
                    except IndexError:
                        Newpop = np.concatenate((Newpop, np.zeros((lpop,pos-Newpop.shape[1]))), axis=1)
                        Newpop[j-1,pos-1] = Oldpop[i-1,k2-1]
                    pos += 1
                    if (pos>=(nxch+1)):
                        pall=1                

    return Newpop


'''
 crosgrp - crossover between more parents

	Description:
	The function recombinates more (also more than 2) parents and creates a required 
	number of offsprings which genes are a random combination of genes of 
	all parents. The number of parents and offsprings need not be equal.


	Syntax:  

	Newgroup=crosgrp(Oldgroup,num)

	         Newgroup - new group of strings
	         Oldgroup - old group of strings
	         num -  number of required new strings
     
 I.Sekaj, 2000
'''
def crosgrp(Oldgrp,num):
    lgrp, lstring = Oldgrp.shape
    Newgrp = np.zeros((num,lstring))
    
    for r in range(num):
        for s in range(lstring):
            m = int(np.ceil(np.random.uniform()*lgrp))-1
            Newgrp[r,s] = Oldgrp[m,s]
    
    return Newgrp

'''
 change - elimination of duplicite strings in the population

	Description:
	The function is searching for and changing all duplicite strings
	in a population. Depending on the parameter option the duplicite strings
	are either modificated using the simple or multiplicative mutation
	or they are replaced by a new random real-number string.


	Syntax:

	Newpop=change(Oldpop,option,Space)

	Newpop - new modificated population 
	Oldpop - old population
	option - 0 - duplicite strings are mutated in one gene 
	             in the ranges defined by the matrix Space
	         1 - duplicite strings are multiplicated in one random gene by a random
	             constant from the range defined by the matrix Space
	         2 - duplicite strings are replaced by a new random real-number 
		     string, which items are limited by the matrix Space
	Space -  2-row matrix, which 1-st row is the vector of the lower limits
		   and the 2-nd row is the vector of the upper limits of the
		   genes.


 I.Sekaj 8/2000
'''
def change(Oldpop,option,Space):
    lpop, lstring = Oldpop.shape
    Newpop = np.copy(Oldpop)
    for s1 in range(1,lpop):
        for s2 in range(s1+1,lpop+1):
            ch = lstring
            for g in range(1,lstring+1):
                if(Newpop[s1-1,g-1]==Newpop[s2-1,g-1]):
                    ch = ch - 1
            if (ch==0):
                if(option==0):
                    s = int(np.ceil(np.random.uniform()*lstring))
                    d = Space[1,s-1] - Space[0,s-1]
                    Newpop[s2-1,s-1] = np.random.uniform()*d + Space[0,s-1]
                    if (Newpop[s2-1,s-1]<Space[0,s-1]):
                        Newpop[s2-1,s-1] = Space[0,s-1]
                    if (Newpop[s2-1,s-1]>Space[1,s-1]):
                        Newpop[s2-1,s-1] = Space[1,s-1]
                elif (option==1):
                    s = int(np.ceil(np.random.uniform()*lstring))
                    d = Space[1,s-1] - Space[0,s-1]
                    Newpop[s2-1,s-1] = Newpop[s2-1,s-1]*(np.random.uniform()*d + Space[0,s-1])
                    if (Newpop[s2-1,s-1]<Space[0,s-1]):
                        Newpop[s2-1,s-1] = Space[0,s-1]
                    if (Newpop[s2-1,s-1]>Space[1,s-1]):
                        Newpop[s2-1,s-1] = Space[1,s-1]
                elif (option==2):
                    rr = genrpop(1, Space)
                    Newpop[s2-1,:] = rr
    
    return Newpop

'''
 around - intermediate crossover with increasing/decreasing of the population living area

	Description:
	The function creates a new population of the strings, which rises after
	intermediate crossover operation of all (couples of) strings of the old
	population. The selection of strings into couples is either random or
	the neighbouring strings are selected, depending on the parameter sel.
	From each couple of parents will be calculated a new couple of offsprings
	as follows: 

	Offspring = (Parent1+Parent2)/2 +(-) alfa(Parent_distance)/2


	Syntax:  Newpop=around(Oldpop,sel,alfa,Space) 

	Newpop - new population
	Oldpop - old population
	sel - selection type of crossover couples:
		0-random couples
		1-neighbouring strings in the population 
       alfa - enlargement parameter,  0.1<alfa<10, (usually: alfa=1.25; or 0.75<alfa<2)
       Space  - matrix of boundaries in the form: [vector of lower limits of genes;
                                                   vector of upper limits of genes];


 I.Sekaj, 5/2000
'''
def around(Oldpop,sel,alfa,Space):
    Newpop = np.copy(Oldpop)
    lpop, lstring = Oldpop.shape
    flag = np.zeros((lpop,))
    num = int(np.floor(lpop/2.0))
    i = 1
    m = 1
    
    for cyk in range(1,num+1):
        if (sel==0):
            while (flag[i-1]!=0):
                i += 1
            flag[i-1] = 1
            j = int(np.ceil(lpop*np.random.uniform()))
            while (flag[j-1]!=0):
                j = int(np.ceil(lpop*np.random.uniform()))
            flag[j-1] = 2
        elif (sel==1):
            i = int(2*cyk - 1)
            j = i + 1
        b, d, c = np.zeros((lstring,)), np.zeros((lstring,)), np.zeros((lstring,))
        for k in range(1,lstring+1):
            b[k-1] = np.min([Oldpop[i-1,k-1], Oldpop[j-1,k-1]])
            d[k-1] = np.max([Oldpop[i-1,k-1], Oldpop[j-1,k-1]]) - b[k-1]
            c[k-1] = (Oldpop[i-1,k-1] + Oldpop[j-1,k-1])/2.0
        
        for k in range(1,lstring+1):
            Newpop[m-1,k-1] = c[k-1] + alfa*(2*np.random.uniform() - 1)*d[k-1]/2.0
            Newpop[m,k-1] = c[k-1] + alfa*(2*np.random.uniform() - 1)*d[k-1]/2.0
            if (Newpop[m-1,k-1]<Space[0,k-1]):
                Newpop[m-1,k-1] = Space[0,k-1]
            if (Newpop[m-1,k-1]>Space[1,k-1]):
                Newpop[m-1,k-1] = Space[1,k-1]
            if (Newpop[m,k-1]<Space[0,k-1]):
                Newpop[m,k-1] = Space[0,k-1]
            if (Newpop[m,k-1]>Space[1,k-1]):
                Newpop[m,k-1] = Space[1,k-1]
        m = m + 2

    return Newpop

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


 I.Sekaj, 12/1998
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
 seldiv - selection based on the maximum diversity measure

	Description:
	The function selects from the input population a required number of strings
	which has the maximal value of Euklidian distance between its genes and the reference
       string. The reference string can be either the most fit string (with the smallest value
	of the objective f. value) or a string which, contains mean values of all corresponding 
	strings of the population. The number of the selected strings depends on the 
	vector Nums in the form: Nums=[number of copies of the best string, ... ,
                                      number of copies of the i-th best string, ...]


	Syntax:

	Newpop=seldiv(Oldpop,Oldfit,Nums,sw)
	[Newpop,Newfit]=seldiv(Oldpop,Oldfit,Nums,sw)

	       Newpop - new selected population
	       Oldpop - old population
	       Oldfit - vector of the objective function values of Oldpop
              Newfit - vector of the objective function values of Newpop
	       Nums   - vector in the form: Nums=[number of copies of the best string, ... ,
                                                 number of copies of the i-th best string,
                                                  ...]
	       sw     - switch: 0 - the reference string consists from genes which are 
                                   mean values of corresponding genes of the Oldpop 
                               1 - the reference string is the best string forom Oldpop


 I.Sekaj, 5/2000
'''
def seldiv(Oldpop,Fvpop,Nums,sw):
    if(len(Fvpop.shape)<=1):
        Fvpop = np.reshape(Fvpop, (1,len(Fvpop)))
    lpop, lstring = Oldpop.shape
    Newpop0 = np.zeros((0, Oldpop.shape[1]))
    Newfit0 = list()
    Newpop = np.zeros((int(np.sum(Nums)), Oldpop.shape[1]))
    Newfit = np.zeros((int(np.sum(Nums),)))
    
    if (sw==1):
        mini = 10e10
        for i in range(lpop):
            if (Fvpop[0,i]<mini):
                mini = Fvpop[0,i]
                ix = i
                
        pstr = Oldpop[ix,:]
    elif (sw==0):
        pstr = np.zeros((lstring,))
        for j in range(lstring):
            pstr[j] = np.mean(Oldpop[:,j])
    
    div = np.zeros((lpop,))
    for i in range(lpop):
        div[i] = 0
        for j in range(lstring):
            div[i] = div[i] + np.abs(pstr[j] - Oldpop[i,j])
    
    N = len(Nums)
    if (N>lpop):
        N = lpop
        
    for j in range(N):
        maxi = 0
        for i in range(lpop):
            if(div[i]>maxi):
                maxi = div[i]
                ix = i
        div[ix] = 0
        Newpop0 = np.concatenate((Newpop0,Oldpop[ix,:].reshape((1,lstring))), axis=0)
        Newfit0.append(Fvpop[0,ix])
    
    Newfit0 = np.array(Newfit0)
    r = 0
    for i in range(N):
        for j in range(Nums[i]):
            Newpop[r,:] = Newpop0[i,:]
            Newfit[r] = Newfit0[i]
            r += 1
    
    return [Newpop, Newfit]

'''
 selfxd - selection based on the diversity measure and fitness

	Description:
	The function selects from the input population a required number of strings
	which combines the maximal distance from the reference string (or their affinity)  and fitness.


	Syntax:

	Newpop=selfxd(Oldpop,Oldfit,Nums,sw)
   [Newpop,Newfit]=selfxd(Oldpop,Oldfit,Nums,sw)

	       Newpop - new selected population
	       Pop - old population
	       Fit - vector of the objective function values of Oldpop
          Newfit - vector of the objective function values of Newpop
	       Nums   - number of selected individuals
	       sw     - 0 = the reference string consists from genes which are 
                       mean values of corresponding genes of the Oldpop 
                   1 = the reference string is the best string forom Oldpop


 I.Sekaj, 1/2012
'''
def selfxd(Pop,Fit,Nums,sw):
    if(len(Fit.shape)<=1):
        Fit = np.reshape(Fit, (1,len(Fit)))
    lpop, lstring = Pop.shape
    if(Nums>lpop):
        Nums = lpop
        
    minfit = np.min(Fit)
    maxfit = np.max(Fit)
    if (sw==1):
        pstr = selbest(Pop,Fit,[1])
    elif (sw==0):
        pstr = np.mean(Pop, axis=0)
    div = np.zeros((1,lpop))

    for i in range(lpop): # Euclid. distance to reference string
        xx = 0
        for j in range(lstring):
            try:
                xx = xx + (pstr[0][0][j] - Pop[i,j])**2
            except IndexError:
                xx = xx + (pstr[j] - Pop[i,j])**2
        div[0,i] = np.sqrt(xx)
        
    
    mindiv = np.min(div)
    maxdiv = np.max(div)
    
    Nfit = np.zeros((lpop,))
    Ndiv = np.zeros((lpop,))
    fxd = np.zeros((lpop,))
    
    for i in range(lpop):
        Nfit[i] = (Fit[0,i]-minfit)/(maxfit-minfit) # normed fitness  <0,1>
        Ndiv[i] = (div[0,i]-mindiv)/(maxdiv-mindiv) # normed distance <0,1>
        fxd[i] = np.sqrt(Nfit[i]**2+(1-Ndiv[i])**2) # fitness x distance
        
    nn = np.ones((1, Nums), dtype=np.dtype(int))
    Newpop, Newfit = selbest(Pop, fxd, nn.tolist()[0])
    
    return [Newpop,Newfit]

'''
 selrand - random selection 


	Description:
	The function selects randomly from the old population a required number
	of strings. 


	Syntax:

	Newpop=selrand(Oldpop,Oldfit,Num);
	[Newpop,Newfit]=selrand(Oldpop,Oldfit,Num);

	       Newpop - new selected population
	       Newfit - fitness vector of Newpop
	       Oldpop - old population
	       Oldfit - fitness vector of Oldpop
	       Num    - number of selected strings


 I.Sekaj, 5/2000
'''
def selrand(Oldpop,Oldfit,n):
    if(len(Oldfit.shape)<=1):
        Oldfit = np.reshape(Oldfit, (1,len(Oldfit)))
    lpop, lstring = Oldpop.shape
    Newpop = np.zeros((n, Oldpop.shape[1]))
    Newfit = np.zeros((n,))
    
    for i in range(n):
        j = int(np.ceil(lpop*np.random.uniform()))-1
        Newpop[i,:] = Oldpop[j,:]
        Newfit[i] = Oldfit[0,j]
    
    return [Newpop,Newfit]

'''
 selsort - selection and sorting of best string of a population

	Descrption:
	The function selects from the old population into the new population
	the required number of best strings and also sorts this strings according
 	their fitness from the most fit to the least fit. The most fit is the string 
	with the lowest value of the objective function and vice-versa.


	Syntax:

	Newpop=selsort(Oldpop,Oldfit,Num);
	[Newpop,Newfit]=selsort(Oldpop,Oldfit,Num);

	       Newpop - new selected population
	       Newfit - fitness vector of Newpop
	       Oldpop - old population
	       Oldfit - fitness vector of Oldpop
	       Num    - number of the selected strings 


 I.Sekaj, 12/1998
'''
def selsort(Pop,Fvpop,N):
    if(len(Fvpop.shape)<=1):
        Fvpop = np.reshape(Fvpop, (1,len(Fvpop)))
    Newpop = np.zeros((N, Pop.shape[1]))
    Newfit = np.zeros((N,))
    
    fit = np.sort(Fvpop)
    nix = np.argsort(Fvpop)
    for j in range(N):
        Newpop[j,:] = Pop[nix[0][j],:]
        Newfit[j] = fit[0][j]

    return [Newpop,Newfit]

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


 I.Sekaj, 5/2000
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

'''
 seltourn - tounament selection 


	Description:
	The function selects using tournament selection from the old population 
       a required number of strings. 


	Syntax:

	Newpop=seltourn(Oldpop,Oldfit,Num);
	[Newpop,Newfit]=seltourn(Oldpop,Oldfit,Num);

	       Newpop - new selected population
              Newfit - fitness vector of Newpop
	       Oldpop - old population
              Oldfit - fitness vector of Oldpop
	       Num    - number of selected strings


 I.Sekaj, 8/2002
'''
def seltourn(Oldpop,Fit,n):
    if(len(Fit.shape)<=1):
        Fit = np.reshape(Fit, (1,len(Fit)))
    lpop, lstring = Oldpop.shape
    Newpop = np.zeros((n, Oldpop.shape[1]))
    Newfit = np.zeros((n,))
    
    for i in range(int(n)):
        j = int(np.ceil(lpop*np.random.uniform()))-1
        k = int(np.ceil(lpop*np.random.uniform()))-1
        if (j==k):
            Newpop[i,:] = Oldpop[j,:]
            Newfit[i] = Fit[0,j]
        elif (Fit[0,j]<=Fit[0,k]):
            Newpop[i,:] = Oldpop[j,:]
            Newfit[i] = Fit[0,j]
        else:
            Newpop[i,:] = Oldpop[k,:]
            Newfit[i] = Fit[0,k]
            
    return [Newpop,Newfit]

'''
 selwrul - rulette wheel selection


	Description:
	The function selects from the old population a required number of strings using
	the "weighted roulette wheel selection". Under this selection method the individuals
	have a direct-proportional probability to their fitness to be selected into the 
	new population. 
	
 
	Syntax:

	Newpop=selwrul(Oldpop,Oldfit,Num);
	[Newpop,Newfit]=selwrul(Oldpop,Oldfit,Num);

	       Newpop - new selected population
	       Newfit - fitness vector of Newpop
	       Oldpop - old population
	       Oldfit - fitness vector of Oldpop
	       Num    - required number of selected strings


 I.Sekaj, 5/2000
'''
def selwrul(Oldpop,Fvpop,n):
    if(len(Fvpop.shape)<=1):
        Fvpop = np.reshape(Fvpop, (1,len(Fvpop)))
    lpop, lstring = Oldpop.shape
    Newpop = np.zeros((n, Oldpop.shape[1]))
    Newfit = np.zeros((n,))
    sumfv = np.sum(Fvpop)
    
    w0 = np.zeros((1,lpop+1))
    for i in range(lpop):
        if(Fvpop[0,i]==0):
            Fvpop[0,i] = 0.001
        men = Fvpop[0,i]*sumfv
        if (men==0):
            men = 0.0000001
        w0[0,i] = 1.0/men # tvorba inverznych vah
    
    #w0[0,i+1] = 0
    w = np.zeros((1,lpop+1))
    
    for i in np.arange(lpop-1,-1,-1):
        w[0,i] = w[0,i+1] + w0[0,i]
    
    maxw = np.max(w)
    if (maxw==0):
        maxw = 0.00001
    w = (w/maxw)*100 # vahovaci vektor
    
    for i in range(n):  # tocenie ruletou
        q = np.random.uniform()*100
        for j in range(lpop):
            if (q<w[0,j] and q>w[0,j+1]):
                break
        Newpop[i,:] = Oldpop[j,:]
        Newfit[i] = Fvpop[0,j]
    
    return [Newpop,Newfit]

'''
 shake - random shaking of the string order in the population

	Description: 
	The function returns a population with random changed order of strings. 
	The strings are without any changes. The intensity of shaking depends on the 
	parameter rate.


	Syntax: 

	Newpop=shake(Oldpop,rate);

	       Newpop - new population with changed string order
	       Oldpop - old population
	       rate   - shaking intensity from 0 to 1


 I.Sekaj, 12/1998
'''
def shake(Oldpop,MR):
    lpop, lstring = Oldpop.shape
    if (MR<0):
        MR = 0
    if (MR>1):
        MR = 1
    
    Newpop = np.copy(Oldpop)
    Hlp = np.copy(Oldpop)
    n = int(lpop*MR)
    
    for i in range(n):
        ch1, ch2 = 0, 0
        while(ch1==ch2):
            ch1 = int(np.ceil(np.random.uniform()*lpop))-1
            ch2 = int(np.ceil(np.random.uniform()*lpop))-1
        Newpop[ch1,:] = Hlp[ch2,:]
        Newpop[ch2,:] = Hlp[ch1,:]
        Hlp[ch1,:] = Newpop[ch1,:]
        Hlp[ch2,:] = Newpop[ch2,:]
    
    return Newpop

'''
 swapgen - mutation of the gene-order in strings


	Description:
	The function exchanges (mutates) the order of some random selected genes
	in random selected strings in the population. The mutation intensity depends
	on the parameter rate.


	Syntax: 

	Newpop=swapgen(Oldpop,rate)

	       Newpop - new mutated population
	       Oldpop - old population
	       rate   - mutation intensity, 0 =< rate =< 1


 I.Sekaj, 2/2001
'''
def swapgen(Oldpop,factor):
    lpop, lstring = Oldpop.shape
    
    if(factor>1):
        factor = 1
    if (factor<0):
        factor = 0
    
    n = int(np.ceil(lpop*lstring*factor*np.random.uniform()))
    Newpop = np.copy(Oldpop)
    
    for i in range(n):
        r = int(np.ceil(np.random.uniform()*lpop))-1
        s1 = int(np.ceil(np.random.uniform()*lstring))-1
        s2 = int(np.ceil(np.random.uniform()*lstring))-1
        if (s1==s2):
            s2 = int(np.ceil(np.random.uniform()*lstring))-1
        pstr = Newpop[r,s1]
        Newpop[r,s1] = Newpop[r,s2]
        Newpop[r,s2] = pstr
        
    return Newpop

'''
 swappart- exchange of the order of two substrings in the strings

	Description:
	The function exchanges the order of two substrings, which will arise after 
	spliting a string in two parts. The number of such modificated strings in
	the population depends on the parameter rate.


	Syntax: 

	Newpop=swappart(Oldpop,rate)

	       Newpop - new modificated population
	       Oldpop - old population
	       rate   - mutation intensity, 0 =< rate =< 1


 I.Sekaj, 2/2001
'''
def swappart(Oldpop,factor):
    lpop, lstring = Oldpop.shape
    
    if(factor>1):
        factor = 1
    if (factor<0):
        factor = 0

    n = int(np.ceil(lpop*factor*np.random.uniform()))
    Newpop = np.copy(Oldpop)
    
    for i in range(n):
        r = int(np.ceil(np.random.uniform()*lpop))-1
        s = int(np.ceil(np.random.uniform()*lstring))-1
        Newpop[r,:] = np.concatenate((Oldpop[r,s:lstring], Oldpop[r,0:s]), axis=0)
    
    return Newpop

'''
 Quadratic objective function, unimodal optimisation problem
 X(opt)=[0 0 0 ... 0]; Fit(opt)=0;
 -10<x<10; 
'''
def testfn1(Pop):
    if(len(Pop.shape)<=1):
        Pop = np.reshape(Pop, (1,len(Pop)))
    lpop, lstring = Pop.shape
    Fit = np.zeros((lpop,))
    for i in range(lpop):
        G = Pop[i,:]
        Fit[i] = 0
        for j in range(lstring):
            Fit[i] = Fit[i] + G[j]*G[j]
            
    return Fit

'''
 Test function 2 (Rastigin's objective function)
 It is a multimodal function with optional number of input variables.
 The global optimum is:  x(i)=0; i=1...n ;  Fit(x)=0;
 -5 < x(i) < 5
 Other local minimas are located in a grid with the step=1
'''
def testfn2(Pop):
    if(len(Pop.shape)<=1):
        Pop = np.reshape(Pop, (1,len(Pop)))
    lpop, lstring = Pop.shape
    Fit = np.zeros((lpop,))
    
    for i in range(lpop):
        G = Pop[i,:]
        Fit[i] = lstring*10
        for j in range(lstring):
            Fit[i] = Fit[i] +  (G[j]**2 - 10*np.cos(2*np.pi*G[j]))
    
    return Fit

'''
 Test function 3 (Schwefel's objective function)
 global optimum: x(i)=420.9687 ;  Fit(x)=-n*418.9829 , n-number of variables
 -500 < x(i) < 500
'''
def testfn3(Pop):
    if(len(Pop.shape)<=1):
        Pop = np.reshape(Pop, (1,len(Pop)))
    lpop, lstring = Pop.shape
    Fit = np.zeros((lpop,))
    
    for i in range(lpop):
        G = Pop[i,:]
        Fit[i] = 0
        for j in range(lstring):
            Fit[i] = Fit[i] - G[j]*np.sin(np.sqrt(np.abs(G[j])))
    
    return Fit

'''
 Test function 6 (Search for the in string H defined combination of 8 integer numbers)
 0<x<10, Fit(opt)=0, (Fit -> number of incorrect integers)
'''
def testfn6(Pop):
    if(len(Pop.shape)<=1):
        Pop = np.reshape(Pop, (1,len(Pop)))
    lpop, lstring = Pop.shape
    eps = 0.05 # tolerance
    H=[0, 1, 2, 3, 4, 5, 6, 7]; # solution
    Fit = np.zeros((lpop,))
    
    for i in range(lpop):
        G = Pop[i,:]
        Fit[i] = lstring
        for j in range(lstring):
            if (np.abs(G[j]-H[j])<eps):
                Fit[i] = Fit[i] - 1
                
    return Fit

'''
 Test function 8 (Rosenbrock fn.)
 The global optimum is:  x(i)=1 ; i=1...n ;  Fit(x)=0 ;
 -2 < x(i) < 2
'''
def testfn8(Pop):
    if(len(Pop.shape)<=1):
        Pop = np.reshape(Pop, (1,len(Pop)))
    lpop, lstring = Pop.shape
    Fit = np.zeros((lpop,))
    
    for i in range(lpop):
        G = Pop[i,:]
        Fit[i] = 0
        for j in range(lstring):
            Fit[i] = Fit[i] + (100*(G[j] - G[j]**2)**2 + (G[j] - 1)**2)

    return Fit

'''
 Test function 9 (Griewank fun.)
 The global optimum is:  x(i)=0 ; i=1...n ;  Fit(x)=0 ;
 -600 < x(i) < 600
'''
def testfn9(Pop):
    if(len(Pop.shape)<=1):
        Pop = np.reshape(Pop, (1,len(Pop)))
    lpop, lstring = Pop.shape
    Fit = np.zeros((lpop,))
    
    for i in range(lpop):
        G = Pop[i,:]
        ff1 = (1.0/4000)*np.sum(np.power(G,2))
        ff2 = 1
        for j in range(lstring):
            ff2 = ff2*(np.cos(G[j]/np.sqrt(j+1)))
        Fit[i] = ff1 - ff2 + 1

    return Fit

'''
 Egg holder
 Number of variables : n
 The global optimum is: n=5  f(x)=-3719.7  x=[481.3291  436.7954  451.5467 466.7904  422.3174]
 -500 < x(i) < 500
'''
def eggholder(Pop):
    if(len(Pop.shape)<=1):
        Pop = np.reshape(Pop, (1,len(Pop)))
    lpop, lstring = Pop.shape
    Fit = np.zeros((lpop,))
    for i in range(lpop):
        x = Pop[i,:]
        Fit[i] = 0
        for j in range(lstring-1):
            Fit[i] = Fit[i] - x[j]*np.sin(np.sqrt(np.abs(x[j]-(x[j+1]+47)))) - (x[j+1]+47)*np.sin(np.sqrt(np.abs(x[j+1]+47+x[j]/2.0)))
    
    return Fit





