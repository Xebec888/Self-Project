import numpy as np
import math
import matplotlib.collections as mc
import matplotlib.pylab as pl

#Psedorandom number generator using number as a Seed.
random_seed = 1729
np.random.seed(random_seed)
N = 40 #number of Cities
x = np.random.rand(N)
y = np.random.rand(N)

#Zipping the X & Y random value points of the city
points = zip(x,y)
cities = list(points)

#First proposed solution will be to simply visit all the cities in the order in which they appear
itinerary = list(range(0,N))
print(itinerary)

#In this algorithm , we will minimize the cost which is same as minimizing distance travelled
"""To determine the distance required by a particular itinerary, we need to define two Functions.
 First, we need a function will generate a collection of lines that connect all of our points.
 Second, We need to sum up the distances represented by those lines."""

lines = [] #empty list to store information about our lines
for j in range(0, len(itinerary)-1):
    lines.append([cities[itinerary[j]], cities[itinerary[j+1]]])
# iterate over every city in our itinerary, at each step adding a new line to our collection of lines.

print(lines) #to see the rand(lines x , y )

#Genlines function put these two elements together in one function (cities, itinerary) returning collection of lines connecting each city in our list of cities
def genlines(cities, itinerary):
    lines = []
    for j in range(0, len(itinerary)-1):
        lines.append([cities[itinerary[j]], cities[itinerary[j+1]]])
    return lines

# To create a function that measures the total distances along these lines generated
# we will be using Pythagorean Theorem to calculate these lines length
#P.s.: In real world earth is curve so straight line like this , can't generate long distance very well. Need to use different math element. For short distance Pythagorean provides a very good approximation to the true distance

def howfar(lines):
    distance = 0
    for j in range(0, len(lines)):
        distance += math.sqrt(abs(lines[j][1][0] - lines[j][0][0])**2 + \
                              abs(lines[j][1][1] - lines[j][0][1])**2)

    return distance
# this function takes a list of lines as its input outputs the sum of the lengths of every line, we use this to with our itinerary to determine total distance our salesman has to travel

total_distance = howfar(genlines(cities, itinerary))
print(total_distance) #outputs the total distance has to travel

#To plot our itinerary for visualization purpose
#the plotitinerary function takes the arguments where cities=cities , itin=itinerary,  plottitle=title and thename=thename for png plot output
def plotitinerary(cities, itin, plottitle, thename):
    lc = mc.LineCollection(genlines(cities, itin), linewidths = 2)
    fig , ax = pl.subplots() #we are working with Figure & Axes directly to exert more control over our visualization
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)
    pl.scatter(x,y)
    pl.title(plottitle)
    pl.xlabel('X Coordinate')
    pl.ylabel('Y Coordinate')
    pl.savefig(str(thename)+'.png')
    pl.show()
    pl.close()

plotitinerary(cities,itinerary,'TSP - Random Itinerary', 'figure2')

# We can see from the visuals we haven't yet found the best solution. Our next steps is to use algorithms to find an itinerary with minimum traveling distance
#first we will use nearest neighbour search algorithm to further tune our TSP travelling distance
#here instead of going to each city one by one , we instead visit first city then find the closest unvisited city and visit that city second


#creating a function to check the closest city to our point by iterating over every element in the cities and check the distance between point and every city
def findnearest(cities, idx, nnitinerary):
    point = cities[idx]
    mindistance = float('inf')
    minidx = -1
    for j in range(0, len(cities)):
        distance = math.sqrt((point[0] - cities[j][0])**2 + (point[1] - cities[j][1])**2) #pythagorean style formula
        if distance < mindistance and distance > 0 and j not in nnitinerary:
            mindistance = distance
            minidx = j
    return(minidx)


#implementing Nearest neighbour algorithm
def donn(cities, N):
    nnitinerary = [0]
    for j in range(0,N-1):
        next = findnearest(cities, nnitinerary[len(nnitinerary)-1], nnitinerary)
        nnitinerary.append(next)
    return(nnitinerary)
#this function start with first city and at every steps adds the closest city to the most recently to the most recently added city until every city has been added to the itinerary


#visulizing the performance of nearest neighbour algorithm
plotitinerary(cities, donn(cities,N),'TSP - Nearest Neighbour', 'figure3')
print(howfar(genlines(cities, donn(cities,N)))) #output total distance needs to travel


#To further improve our search for TSP , we can switch the places of index in the itinerary to see if that changes further improve our travelling distance
# we will create a perturb function/ search algorithm that will do the above mention idea for us
def perturb(cities, itinerary):
    neighborsids1 = math.floor(np.random.rand() * (len(itinerary)))
    neighborsids2 = math.floor(np.random.rand() * (len(itinerary)))

    itinerary2 = itinerary.copy()
    itinerary2[neighborsids1] = itinerary[neighborsids2]
    itinerary2[neighborsids2] = itinerary[neighborsids1]

    distance1 = howfar(genlines(cities, itinerary))
    distance2 = howfar(genlines(cities, itinerary2))

    itinerarytoreturn = itinerary.copy()

    if (distance2 < distance1):
        itinerarytoreturn = itinerary2.copy()

    return(itinerarytoreturn.copy())




#now we call the perturb() repeatedly on a random itinerary to get the lowest traveling distance possible
np.random.seed(random_seed)
itinerary_ps = itinerary.copy()
for n in range(0, len(itinerary) * 10000):
    itinerary_ps = perturb(cities, itinerary_ps)
print(howfar(genlines(cities, itinerary_ps)))

#the nearest neighbor algorithm and perturb search algorithm are family of the greedy search algorithm . Greedy search algorithms proceed in steps, and they make choices based of local optimal at each step but may not be globally optimal once all the steps considered
#since greedy algorithms search for only local improvements , they will never allow us to go down and can get us stuck on local extrema.
#To resolve this local optimization problem caused by greedy algorithms , we need to stop the commitment of always climbing up and sometimes go down to climb even more higher point.
#This is done by simulated annealing which forces the path to not always climb up but sometimes climb down to reach higher grounds
#In simulated Annealing , we are sometimes willing to accepts itinerary changes that increase the distance traveled , because this enables us to avoid the problem of local optimization. Our willingness to accept worse itineraries depends on the current temperature.

def perturb_sa1(cities, itinerary, time):
    neighborsids1 = math.floor(np.random.rand() * (len(itinerary)))
    neighborsids2 = math.floor(np.random.rand() * (len(itinerary)))

    itinerary2 = itinerary.copy()
    itinerary2[neighborsids1] = itinerary[neighborsids2]
    itinerary2[neighborsids2] = itinerary[neighborsids1]

    distance1 = howfar(genlines(cities, itinerary))
    distance2 = howfar(genlines(cities, itinerary2))

    itinerarytoreturn = itinerary.copy()

    #main changes from purturb to SA:

    randomraw = np.random.rand()
    temperature = 1 / ((time/1000)+1)

    if((distance2 > distance1 and (randomraw) < (temperature)) or (distance1 > distance2)):
        itinerarytoreturn = itinerary.copy()

    return (itinerarytoreturn.copy())

# restarting the search problem and comparing with each created search algorithm to find lowest distance found by this algorithms.

np.random.seed(random_seed)
itinerary_sa = itinerary.copy()
for n in range(0, len(itinerary) * 10000):
    itinerary_sa = perturb_sa1(cities, itinerary_sa,n)

print(howfar(genlines(cities, itinerary))) # random itinerary
print(howfar(genlines(cities, itinerary_ps))) #perturb search
print(howfar(genlines(cities, itinerary_sa))) # simulated annealing
print(howfar(genlines(cities, donn(cities,N))))  # nearest neighbour

""" Tuning our simulated annealing by choosing different perturbing methods reverse way of choosing subset of cities, switching the places of cities and lifting and moving subset of cities.
we will choose between these methods based of the random number of selection between 0 to 1. also, we added a major setbacks method to not to make changes by going down but not leaving us a 
 worse situation than it was. so we added a reset method to choose the last known less bad situation."""

def perturb_saupgrade(cities, itinerary, time, maxitin):
    neighborsids1 = math.floor(np.random.rand() * (len(itinerary)))
    neighborsids2 = math.floor(np.random.rand() * (len(itinerary)))
    global mindistance
    global minitinerary
    global minidx
    itinerary2 = itinerary.copy()
    randomraw = np.random.rand()
    randomraw2 = np.random.rand()

    small = min(neighborsids1, neighborsids2)
    big = max(neighborsids1, neighborsids2)
    if(randomraw2 >= 0.55):
        itinerary2[small:big] = itinerary[small:big] [::-1]
    elif(randomraw2 < 0.45):
        tempitin = itinerary[small:big]
        del(itinerary2[small:big])
        neighborsids3 = math.floor(np.random.rand() * (len(itinerary)))
        for j in range (0, len(tempitin)):
            itinerary2.insert(neighborsids3 + j, tempitin[j])

    else:
        itinerary2[neighborsids1] = itinerary[neighborsids2]
        itinerary2[neighborsids2] = itinerary[neighborsids1] #choosing the method to perturb based on randomraw2


    temperature = 1/(time/(maxitin/10)+1)
    distance1 = howfar(genlines(cities, itinerary))
    distance2 = howfar(genlines(cities, itinerary2))

    itinerarytoreturn = itinerary.copy()

    scale = 3.5
    if((distance2 > distance1 and (randomraw) < (math.exp(scale*(distance1 - distance2)) * temperature)) or (distance1 > distance2)):
        itinerarytoreturn = itinerary2.copy()

    #reset section to avoid worse decisions
    """we define the global variables for the minimum distance achieved so far and the itinerary that achieved it, the time it was achieved. if time progress very far without
     finding anything better than the itinerary that achieved our minimum distance, wee conclude that the changes made after point was mistakes and we allow reseting to our best itinerary.
     resetthresh will determing how long we should wait before resetting. Maxitin normally tells the function how many total times we intend to call this function. we use maxitin in our
     temperature function as well so the temperature curve can adjust flexibly to however many perturbations we intend to perform. when our time is up we return the itinerary that gave us the best result"""
    reset = True
    resetthresh = 0.04
    if(reset and (time - minidx) > (maxitin * resetthresh)):
        itinerarytoreturn = minitinerary
        minidx = time


    if(howfar(genlines(cities, itinerarytoreturn)) < mindistance):
        mindistance = howfar(genlines(cities, itinerary2))
        minitinerary = itinerarytoreturn
        minidx = time

    if(abs(time - maxitin) <= 1):
        itinerarytoreturn = minitinerary.copy()

    return(itinerarytoreturn.copy())


#we create fuction with our global variables and then call our newest perturb tuned version
def siman(itinerary, cities):
    newitinerary = itinerary.copy()
    global mindistance
    global minitinerary
    global minidx
    mindistance = howfar(genlines(cities, itinerary))
    minitinerary = itinerary
    minidx = 0

    maxitin = len(itinerary) * 10000
    for t in range(0 , maxitin):
        newitinerary = perturb_saupgrade(cities, newitinerary, t, maxitin)

    return(newitinerary.copy())

np.random.seed(random_seed)
itinerary = list(range(N))
nnitin = donn(cities, N)
nnresult = howfar(genlines(cities, nnitin))
simanitinerary = siman(itinerary, cities)
simanresult = howfar(genlines(cities, simanitinerary))
print(nnresult)
print(simanresult)
print(simanresult/nnresult)

plotitinerary(cities, simanitinerary , 'Traveling sales Itinrary - simulated annealing', 'figure5')

