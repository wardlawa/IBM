print "hello world"

import argparse
import sys
import random
import time
import numpy
import math
from collections import namedtuple
from scipy.stats import truncnorm
from random import shuffle

testFile = open('test.out','w')
sys.stdout = testFile

testFile2 = open('test.err','w')
sys.stderr = testFile2

random.seed(time.time())

#Individual = namedtuple("Individual", "sex y1 y2 x1 x2 inf exp")
#sex == 0 is females, sex == 1 is males, 
#inf (infected) --> individual is infected with an STD with exploitation rate inf
#exp (exposed) --> individual has been expposed to an STD with exploitation rate exp this generation, but cannot transmit the disease or suffer disease-induced mortality until next generation when they have become infected 

def main():
    print("#%s" % args)
    for rep in range(args.replicates):
        populations = generatePopulation()
        global listYTrait 
        global listXTrait 
        global listETrait 
        listYTrait = []
        listXTrait = []
        listETrait = []
        atEq = 0
        #popStats(populations,0)
        #####Think carefully about whether we need to report this initiated population
        for gen in range(args.generations):
            #print "Generation = ", gen
            newGenerations = pickParents(populations, gen)
            shuffle(newGenerations)
            populations = newGenerations
            finalStat = popStats(populations,gen,rep)
            if finalStat == 0:
                #print "population extinct!"
                break
            else:
                print finalStat
                slidingWindow(finalStat[2], finalStat[4], finalStat[6])
                #print "lists", listYTrait, listXTrait, listETrait
                avgYWindow = numpy.average(listYTrait)
                avgXWindow = numpy.average(listXTrait)
                avgEWindow = numpy.average(listETrait)
                stdYWindow = numpy.std(listYTrait)
                stdXWindow = numpy.std(listXTrait)
                stdEWindow = numpy.std(listETrait)
                #print "stds", stdYWindow, stdXWindow, stdEWindow
                if len(listYTrait) == args.window_length and stdYWindow < 0.001 and stdXWindow < 0.001 and stdEWindow < 0.001:
                    break
                else:
                    pass
                
#        #check if we need to split the population
#        if i == args.split_generation:
#            populations.append(splitPopulation(populations[0]))
#
#        for j, population in enumerate(populations):
#            selfing_rate = args.selfing_rate if j == 0 else args.new_selfing_rate
#            next_generation = []
#            #make all the offspring
#            for k in range(args.population_size):
#                next_generation.append(makeOffspring(population, selfing_rate))
#            #set this as the main generation and output data for it
#            populations[j] = next_generation
#            stats = calculateStats(next_generation)
#            outputData(stats, population_name=j, generation=i)

def parseArgs():
    parser = argparse.ArgumentParser(description="Simulates a population who's individuals' fitness depend on their persistence (male) or resistance (female) trait.") 
    parser.add_argument("-R", "--replicates", default=10, help="The number of replicate simulations to run.", type=int)
    parser.add_argument("-N", "--population_size", default=10, help="The population size of the starting population.", type=int)
    parser.add_argument("-K", "--carrying_capacity", default=100, help="The carrying capacity of the population.", type=float)
    parser.add_argument("-G", "--generations", default=10, help="The maximum number of generations to run the simulation.", type=int)
    parser.add_argument("-b", "--birth_rate", default=2, help="The intrinsic birth rate of females.", type=int)
    parser.add_argument("-r", "--recombination_rate", default=0.5, help="The recombination rate in males and females.", type=float)
    parser.add_argument("-c", "--cost_persistence", default=0.5, help="The cost to males of possessing persistence value y.", type=float)
    parser.add_argument("-d", "--cost_harassment", default=0.02, type=float, help="The cost to females of a certain mating rate.")
    parser.add_argument("-a", "--encounter_rate", default=1, type=float, help="The encounter rate between males and females")
    parser.add_argument("-dm", "--death_rate_males", default=0.2, type=float, help="The intrinsic mortality rate of males.")
    parser.add_argument("-df", "--death_rate_females", default=0.2, type=float, help="The intrinsic mortality rate of females.")
    parser.add_argument("-in", "--introduce_STD", default=1, type = int, help="The generation the STD should be introduced into an otherwise susceptible population")
    parser.add_argument("-e", "--exploitation", default=1, type=float, help="The exploitation rate of the parasite")
    parser.add_argument("-pm", "--male_resistance", default=0.0, type=float, help="The reduction in parasite exploitation due to male host resistance.")
    parser.add_argument("-pf", "--female_resistance", default=0.0, type=float, help="The reduction in parasite exploitation due to female host resistance.")
    parser.add_argument("-W", "--fitness_function", default="ALL", choices=["ALL", "HOMO", "HET"], help="The fitness function to be used. ")
    parser.add_argument("-w", "--tradeoff", default=1, type=int, help="The parameter governing the shape of the tradeoff between transmission and virulence.")
    parser.add_argument("-y", "--male_persistence_trait", default=5.0, type=float, help="The initial male persistence value in the population.")
    parser.add_argument("-x", "--female_resistance_trait", default=1.0, type=float, help="The initial female resistance value in the population.")
    parser.add_argument("-mhm", "--mutation_host_male", default = 0.0001, type=float, help="The mutation rate of male host traits.") 
    parser.add_argument("-mhf", "--mutation_host_female", default = 0.0001, type=float, help="The mutation rate of female host traits.")
    parser.add_argument("-mp", "--mutation_parasite", default = 0.0001, type=float, help="The mutation rate of parasite exploitation.")
    parser.add_argument("-wl", "--window_length", default = 50, type=int, help="The length of the sliding window over which standard deviation is measured to determine if simulation should stop") 
    parser.add_argument("-t", "--test", action='store_true', help="Use this option to run tests on all methods with doctests.")
    
    args = parser.parse_args()
    return args

'''
All the functions below here have an optional input called "rands".
This is used only for testing the modules so when writing tests
we dont need to worry about the random numbers that get drawn during any particular test. 
If you are writing a test "rands" should be a list of 'random' numbers 
that is at least as long as the number of random numbers the function would need. Otherwise the test will simply crash. 
'''

'''
Generates a random population based on the input values
'''
#def testFun(a, b):
#    '''
#    >>> testFun(2,3)
#    6
#    '''
#    return a*b

class Individual:
    def __init__(self, sex, y1, y2, x1, x2, infected, exposed):
        self.sex = sex
        self.y1 = y1
        self.y2 = y2
        self.x1 = x1
        self.x2 = x2
        self.infected = infected
        self.exposed = exposed

    def __str__(self):
        return "Individual(%s, %s, %s, %s, %s, %s, %s)" % (self.sex, self.y1, self.y2, self.x1, self.x2, self.infected, self.exposed)

    def __repr__(self):
        return "Individual(%s, %s, %s, %s, %s, %s, %s)" % (self.sex, self.y1, self.y2, self.x1, self.x2, self.infected, self.exposed)


def generatePopulation(rands = None):
    '''
    >>>print "testing 123"
    testing 123
    '''
    population = []
    sex = numpy.random.binomial(1,0.5,args.population_size)
    #0 = female, 1 = male
#    print sex
    std = 1
    if args.mutation_host_male == 0:
        persis1 = [args.male_persistence_trait]*args.population_size
        persis2 = [args.male_persistence_trait]*args.population_size
    else:
        persis1 = truncnorm.rvs(a = (0 - args.male_persistence_trait)/std, b = numpy.inf, loc = args.male_persistence_trait, scale = std, size = args.population_size)
        persis2 = truncnorm.rvs(a = (0 - args.male_persistence_trait)/std, b = numpy.inf, loc = args.male_persistence_trait, scale = std, size = args.population_size)
#    print "persis1", persis1
#    print "persis2", persis2
    if args.mutation_host_female == 0:
        resis1 = [args.female_resistance_trait]*args.population_size
        resis2 = [args.female_resistance_trait]*args.population_size
    else:
        resis1 = truncnorm.rvs(a = (0 - args.female_resistance_trait)/std, b = numpy.inf, loc = args.female_resistance_trait, scale = std, size = args.population_size)
        resis2 = truncnorm.rvs(a = (0 - args.female_resistance_trait)/std, b = numpy.inf, loc = args.female_resistance_trait, scale = std, size = args.population_size)
#    print "resis1", resis1
#    print "resis2", resis2
    for i in range(args.population_size):
        population.append(Individual(sex[i], persis1[i], persis2[i], resis1[i], resis2[i], 0, 0))
#        print population[i]
#        print "runningPop", population
#        #put half of the TEs on each haplotype
#        num_of_tes_haplotype1 = numpy.random.poisson(args.initial_te_number/2)
#        num_of_tes_haplotype2 = numpy.random.poisson(args.initial_te_number/2)
#        #assign each TE a position
#        haplotype1 = sorted(random.sample(range(0,args.genome_size), num_of_tes_haplotype1))
#        haplotype2 = sorted(random.sample(range(0,args.genome_size), num_of_tes_haplotype2))
#        #calculate the fitness of this individual and put it into the population
#        population.append(calculateFitness(Individual(0.0, haplotype1, haplotype2)))
    #print "population", population
    return population

def pickParents(population, generation):     
    if generation == args.introduce_STD:
        initInf = numpy.random.choice(len(population), 1)
        #print ("initInf %s" % initInf)
        std = 0.5
        population[initInf].exposed = truncnorm.rvs(a = (0 - args.exploitation)/std, b = numpy.inf, loc = args.exploitation, scale = std, size = 1)[0]
        #print "infected individual", population[initInf]
    females = [ind for ind in population if ind.sex == 0]
    males = [ind for ind in population if ind.sex == 1]
    #print "\t len(females)", len(females)
    #print "\t len(males)", len(males)
    newGeneration = []
    densityDependence = (1 - (float(len(females)) + float(len(males)))/args.carrying_capacity) 
    #print densityDependence
    numberOfEncounters = numpy.random.poisson(args.encounter_rate, len(females))
    if densityDependence < 0:
        fecundity = numpy.zeros((len(females)), dtype = numpy.int)
    else: 
        fecundity = numpy.random.poisson(args.birth_rate*densityDependence, len(females))
 #   print "numberOfEncounters", numberOfEncounters
    #print "density dep term", densityDependence
    #print "fecundity", fecundity
    #print ("len(females) %s" % len(females))
    for i in range(0, len(females)):
        #print "\t female ", i
        matedMales = []
        #matedMales keeps track of which males an individual female successfully mated with and needs to be cleared for each female
#        print "numberOfMates", numberOfMates
#        if numberOfEncounters[i] == 0:
#            #determine if she survives to the next generation
#            pickDeath = random.uniform(0,1)
#            print ("pickDeath %s" % pickDeath)
#            if (args.death_rate_females) <= pickDeath:
#                print "\t continue"
#                continue
#            else: 
#                newGeneration.append(females[i])
#                print "added to new gen!" 
        if len(males) == 0:
            encounteredMales = []
        else: 
            encounteredMales = numpy.random.choice(len(males),size = numberOfEncounters[i], replace = True)
        ### Need to deal with what happens when there are so few males in the population that this does not work anymore
#        print "encounteredMales", encounteredMales
        #print "length of testNum", len(testNum)
        for j in range(0, len(encounteredMales)): 
            test = matingProb(females[i], males[encounteredMales[j]])
            #print "test", test
            randy = random.uniform(0,1)
            #print "randy", randy
            if randy <= test:
#                print "mated!"
                #matedMales += 1
                #numberOfMates.append(1)
#                print males[testNum[j]]
                if males[encounteredMales[j]].infected != 0:
                    if females[i].infected == 0 and females[i].exposed == 0: 
                        pickTransmission = random.uniform(0,1)
                        transmissionProb = ((1 - args.male_resistance)*males[encounteredMales[j]].infected)/(args.tradeoff + ((1 - args.male_resistance)*males[encounteredMales[j]].infected))
                        #print "transmission probability from males to females", transmissionProb
                        if pickTransmission < transmissionProb:
                            females[i].exposed = males[encounteredMales[j]].infected
                            #print ("female exposed = %6.5f" % females[i].exposed)
                            #print ("male infected = %6.5f" % males[encounteredMales[j]].infected)
                        else: pass
                    else: pass
                if females[i].infected != 0:
                    if males[encounteredMales[j]].infected == 0 and males[encounteredMales[j]].exposed == 0:
                        pickTransmission = random.uniform(0,1)
                        transmissionProb = ((1 - args.female_resistance)*females[i].infected)/(args.tradeoff + ((1 - args.female_resistance)*females[i].infected))
                        #print "transmission probability from females to males", transmissionProb
                        if pickTransmission < transmissionProb:
                            males[encounteredMales[j]].exposed = females[i].infected
                        else: pass
                    else: pass
                matedMales.append(males[encounteredMales[j]])
        #print "num of mates", len(matedMales)
        #print "matedMales", matedMales 
        dads = []
        if len(matedMales) == 0:
            pass
        else:
            for test in range(fecundity[i]):
                #print ("test = %s" % test)
                #print("fecundity[%s] = %s" % (i, fecundity[i]))
                dads.append(numpy.random.randint(0, len(matedMales)))
            #print ("dads %s" % (dads))
            siringSuccess = [dads.count(m) for m in range(len(matedMales))]
            #print (" siringSuccess %s" % (siringSuccess))
       #siringSuccess prints out a list of how many babies each mated male sires given the fecundity of the female 
        #probs = [float(num)/len(numberOfMates) for num in numberOfMates]
        #siringSuccess = numpy.random.multinomial(len(matedMales), probs)
        #print "siringSuccess", siringSuccess
        for k in range(0, len(matedMales)):
            for babes in range(0, siringSuccess[k]):
                #print ("i in baby making loop %s" % i)
                #print ("k in baby making loop %s" % k)
                #print "Chosen Female", females[i]
                #print "Chosen Male", matedMales[k]
                #print "Siring Success of Chosen Male", siringSuccess[k]
                makeBabies(females[i], matedMales[k],newGeneration)
        pickDeath = random.uniform(0,1)
        #print ("pickDeathFemales %s" % pickDeath)
        #print ("double check number of mates %s" % len(matedMales))
        if females[i].infected != 0: 
            realizedDeathRateFemales = args.death_rate_females + args.cost_harassment*len(matedMales) + (1 - args.female_resistance)*females[i].infected
            #print "Infected female", realizedDeathRateFemales
        else: 
            realizedDeathRateFemales = args.death_rate_females + args.cost_harassment*len(matedMales)
            #print "Uninfected female", realizedDeathRateFemales
        if pickDeath <= (1 - numpy.exp(-realizedDeathRateFemales)):
            continue
        else: 
            newGeneration.append(females[i])
            #print "added to new generation!"
    for m in range(0,len(males)):
        pickDeath=random.uniform(0,1)
        #print("pickDeathMales %s" % pickDeath)
        if males[m].infected != 0:
            realizedDeathRateMales = args.death_rate_males + args.cost_persistence*(males[m].y1 + males[m].y2)/2 + (1 - args.male_resistance)*males[m].infected
            #print "Infected male", (1 - numpy.exp(-realizedDeathRateMales))
        else:
            realizedDeathRateMales = args.death_rate_males + args.cost_persistence*(males[m].y1 + males[m].y2)/2
            #print "Uninfected male", (1- numpy.exp(-realizedDeathRateMales))
        if pickDeath <= (1 - numpy.exp(-realizedDeathRateMales)):
            #print "Dies!"
            continue
        else:
           # print "Lives!"
            newGeneration.append(males[m])
    #print "newGeneration", newGeneration
    for ind in range(0, len(newGeneration)):
        if newGeneration[ind].exposed != 0:
            newGeneration[ind].infected = newGeneration[ind].exposed
            newGeneration[ind].exposed = 0
    #print "newGeneration Infected", newGeneration
    return newGeneration


def matingProb(female, male):
    resistanceTrait = (female.x1 + female.x2)/2
    persistenceTrait = (male.y1 + male.y2)/2
    phi = 1/(1 + math.exp(-(persistenceTrait-resistanceTrait)))
    return phi

def makeBabies(female, male, newPop):
    sex = numpy.random.binomial(1,0.5,1)
    #print ("sex %s" % (sex))
    chromosome1 = chooseAlleles(female)
    chromosome2 = chooseAlleles(male)
    genotype = chromosome1 + chromosome2
#    print "genotype", genotype
#    print "length", len(genotype)
    mut = numpy.random.uniform(low = 0, high = 1, size = 4)
#    print "mut", mut
    std = 0.3
        ###Maybe could use popStats to give current standard deviation and use that???$######
    for m in range (0,len(mut)):
        if m == 1 or m == 3:
            if mut[m] <= args.mutation_host_male:
                tempGenotype = genotype[m] + numpy.random.normal(0, std)
                genotype[m] = tempGenotype
                #####Ask Robert if we can just set genotype[m] = truncnorm.rvs etc#######
            else:
                pass
        elif m == 0 or m == 2:
            if mut[m] <= args.mutation_host_female:
                tempGenotype = genotype[m] + numpy.random.normal(0,std)
                genotype[m] = tempGenotype
            else:
                pass
    offspring = Individual(sex[0], genotype[1], genotype[3], genotype[0], genotype[2], 0, 0)
    #all offspring are born uninfected and unexposed
    #print "offspring", offspring
    newPop.append(offspring)
    return newPop        

def chooseAlleles(ind):
    pickX = random.uniform(0,1)
    if pickX <= 0.5:
        offspringX1 = ind.x1
        pickRecomb = random.uniform(0,1)
        if pickRecomb <= args.recombination_rate:
            offspringY1 = ind.y2
        else:
            offspringY1 = ind.y1
    else:
        offspringX1 = ind.x2
        pickRecomb = random.uniform(0,1)
        if pickRecomb <= args.recombination_rate:
            offspringY1 = ind.y1
        else:
            offspringY1 = ind.y2
    return [offspringX1, offspringY1]

def popStats(population, generation, replicate):
    yVals = []
    xVals = []
    parasite = []
    for i in range(0,len(population)):
        yVals.append(population[i].y1)
        yVals.append(population[i].y2)
        xVals.append(population[i].x1)
        xVals.append(population[i].x2)
        if population[i].infected != 0 or population[i].exposed != 0:            
            parasite.append(population[i].infected)
    avgY = numpy.average(yVals)
    stdY = numpy.std(yVals)
    avgX = numpy.average(xVals)
    stdX = numpy.std(xVals)
    if len(parasite) == 0:
        avgE = 0.0
        stdE = 0.0
    else:
        avgE = numpy.average(parasite)
        stdE = numpy.std(parasite)
    #print "length of parasite array", len(parasite)
    femaleSus = [ind for ind in population if ind.sex == 0 and ind.infected == 0]
    maleSus = [ind for ind in population if ind.sex == 1 and ind.infected == 0]
    #before calling popStats we make all exposed individuals infected
    femaleInf = [ind for ind in population if ind.sex == 0 and ind.infected != 0]
    maleInf = [ind for ind in population if ind.sex == 1 and ind.infected != 0]
####Possible way to save time by using this count from pickParents####
    popSize = (len(maleSus) + len(femaleSus) + len(maleInf) + len(femaleInf))        
    if popSize == 0:
        return 0
    else:
        return [replicate, generation, avgY, stdY, avgX, stdX, avgE, stdE, len(maleSus), len(femaleSus), len(maleInf), len(femaleInf)]

def slidingWindow(averageY, averageX, averageE):
    if len(listYTrait) < args.window_length:
        listYTrait.append(averageY)
        listXTrait.append(averageX)
        listETrait.append(averageE)
    else:
        listYTrait.pop(0)
        listXTrait.pop(0)
        listETrait.pop(0)
        listYTrait.append(averageY)
        listXTrait.append(averageX)
        listETrait.append(averageE)
    return listYTrait, listXTrait, listETrait

def pickParent(population, other_parent = None, rands = None): 
    #this scaling factor helps us deal with negative fitnesses by making all the numbers positive
    scaling_factor = abs(min([individual.fitness for individual in population])) + 0.001
    total_fitness = sum([individual.fitness+scaling_factor for individual in population]) 
    sorted_population = sorted(population)
    if other_parent: sorted_population.remove(other_parent)
    cummulative_fitness = 0.0
    chosen_fitness = rands.pop(0) if rands else random.random()

    for individual in sorted_population:
        cummulative_fitness += (individual.fitness+scaling_factor)/total_fitness
        if cummulative_fitness >= chosen_fitness:
            return individual
    return individual

if __name__ == "__main__":
    args = parseArgs()
    if not args.test:
        import time
        start = time.time()
        main()
        end = time.time()
        m, s = divmod(end-start, 60)
        h, m = divmod(m, 60)
        sys.stderr.write("TIME: %s:%s:%s\n" % (h,m,s))
    else:
        import doctest
        doctest.testmod()

testFile.close()
testFile2.close()
