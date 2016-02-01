print "hello world"

import argparse
import sys
import random
import time
import numpy
import math
from collections import namedtuple
from scipy.stats import truncnorm

testFile = open('test.out','w')
sys.stdout = testFile

testFile2 = open('test.err','w')
sys.stderr = testFile2

seeded = random.seed(time.time())

Individual = namedtuple("Individual", "sex y1 y2 x1 x2")

def main():
    print("#%s" % args)
    print("random seed = %s" % seeded)
    populations = generatePopulation()
#    popStats(populations)
    for i in range(args.generations):
        print "Generation = ", i
        newGenerations = pickParents(populations)
        populations = newGenerations
#        popStats(populations)
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
    parser.add_argument("-N", "--population_size", default=10, help="The population size of the starting population.", type=int)
    parser.add_argument("-K", "--carrying_capacity", default=10, help="The carrying capacity of the population.", type=int)
    parser.add_argument("-G", "--generations", default=10, help="The maximum number of generations to run the simulation.", type=int)
    parser.add_argument("-b", "--birth_rate", default=2, help="The intrinsic birth rate of females.", type=int)
    parser.add_argument("-r", "--recombination_rate", default=0.5, help="The recombination rate in males and females.", type=float)
    parser.add_argument("-c", "--cost_persistence", default=0.5, help="The cost to males of possessing persistence value y.", type=float)
    parser.add_argument("-d", "--cost_harassment", default=0.02, type=float, help="The cost to females of a certain mating rate.")
    parser.add_argument("-a", "--encounter_rate", default=1, type=float, help="The encounter rate between males and females")
    parser.add_argument("-dm", "--death_rate_males", default=0.2, type=float, help="The intrinsic mortality rate of males.")
    parser.add_argument("-df", "--death_rate_females", default=0.2, type=float, help="The intrinsic mortality rate of females.")
    parser.add_argument("-e", "--virulence", default=1, type=float, help="The exploitation rate of the parasite")
    parser.add_argument("-pm", "--male_resistance", default=0.0, type=float, help="The reduction in parasite exploitation due to male host resistance.")
    parser.add_argument("-pf", "--female_resistance", default=0.0, type=float, help="The reduction in parasite exploitation due to female host resistance.")
    parser.add_argument("-W", "--fitness_function", default="ALL", choices=["ALL", "HOMO", "HET"], help="The fitness function to be used. ")
    parser.add_argument("-w", "--tradeoff", default=1, type=int, help="The parameter governing the shape of the tradeoff between transmission and virulence.")
    parser.add_argument("-y", "--male_persistence_trait", default=5.0, type=float, help="The initial male persistence value in the population.")
    parser.add_argument("-x", "--female_resistance_trait", default=1.0, type=float, help="The initial female resistance value in the population.")
    parser.add_argument("-mh", "--mutation_host", default = 0.0001, type=float, help="The mutation rate of host traits.")
    parser.add_argument("-mp", "--mutation_parasite", default = 0.0001, type=float, help="The mutation rate of parasite exploitation.") 
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
    persis1 = truncnorm.rvs(a = (0 - args.male_persistence_trait)/std, b = numpy.inf, loc = args.male_persistence_trait, scale = std, size = args.population_size)
    persis2 = truncnorm.rvs(a = (0 - args.male_persistence_trait)/std, b = numpy.inf, loc = args.male_persistence_trait, scale = std, size = args.population_size)
#    print "persis1", persis1
#    print "persis2", persis2
    resis1 = truncnorm.rvs(a = (0 - args.female_resistance_trait)/std, b = numpy.inf, loc = args.female_resistance_trait, scale = std, size = args.population_size)
    resis2 = truncnorm.rvs(a = (0 - args.female_resistance_trait)/std, b = numpy.inf, loc = args.female_resistance_trait, scale = std, size = args.population_size)
#    print "resis1", resis1
#    print "resis2", resis2
    for i in range(args.population_size):
        population.append(Individual(sex[i], persis1[i], persis2[i], resis1[i], resis2[i]))
#        print "runningPop", population
#        #put half of the TEs on each haplotype
#        num_of_tes_haplotype1 = numpy.random.poisson(args.initial_te_number/2)
#        num_of_tes_haplotype2 = numpy.random.poisson(args.initial_te_number/2)
#        #assign each TE a position
#        haplotype1 = sorted(random.sample(range(0,args.genome_size), num_of_tes_haplotype1))
#        haplotype2 = sorted(random.sample(range(0,args.genome_size), num_of_tes_haplotype2))
#        #calculate the fitness of this individual and put it into the population
#        population.append(calculateFitness(Individual(0.0, haplotype1, haplotype2)))
#    print "population", population
    return population

def pickParents(population):     
    females = [ind for ind in population if ind.sex == 0]
    males = [ind for ind in population if ind.sex == 1]
    print "\t len(females)", len(females)
#    print "males", males
    newGeneration = []
    numberOfEncounters = numpy.random.poisson(args.encounter_rate, len(females))
    fecundity = numpy.random.poisson(args.birth_rate, len(females))
 #   print "numberOfEncounters", numberOfEncounters
    print "fecundity", fecundity
    for i in range(0, len(females)):
        print "\t female ", i
        matedMales = []
        #matedMales keeps track of which males an individual female successfully mated with and needs to be cleared for each female
#        print "numberOfMates", numberOfMates
        if numberOfEncounters[i] == 0:
            #determine if she survives to the next generation
            pickDeath = random.uniform(0,1)
            print ("pickDeath %s" % pickDeath)
            if (args.death_rate_females) <= pickDeath:
                print "\t continue"
                continue
            else: 
                newGeneration.append(females[i])
                print "added to new gen!" 
        encounteredMales = numpy.random.choice(len(males),size = numberOfEncounters[i], replace = False)
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
                matedMales.append(males[encounteredMales[j]])
            print "num of mates", len(matedMales)
            print "matedMales", matedMales
        
        dads = []
        for i in range(fecundity[i]):
            dads.append(numpy.random.randint(0, len(matedMales)))
        print ("dads %s" % (dads))
        siringSuccess = [dads.count(i) for i in range(len(matedMales))]
        print (" siringSuccess %s" % (siringSuccess))
       #siringSuccess prints out a list of how many babies each mated male sires given the fecundity of the female 
        #probs = [float(num)/len(numberOfMates) for num in numberOfMates]
        #siringSuccess = numpy.random.multinomial(len(matedMales), probs)
        #print "siringSuccess", siringSuccess
        for k in range(0, len(matedMales)):
            for babes in range(0, siringSuccess[k]):
                makeBabies(females[i], matedMales[k], siringSuccess[k], fecundity[i], newGeneration)
        pickDeath = random.uniform(0,1)
        print ("pickDeathFemales %s" % pickDeath)
        print ("double check number of mates %s" % len(matedMales))
        if (args.death_rate_females + args.cost_harassment*len(matedMales)) <= pickDeath:
            continue
        else: 
            newGeneration.append(females[i])
    for m in range(0,len(males)):
        pickDeath=random.uniform(0,1)
        print("pickDeathMales %s" % pickDeath)
        if (args.death_rate_males + args.cost_persistence*(males[m].y1 + males[m].y2)/2) <= pickDeath:
            continue
        else:
            newGeneration.append(males[m])
    print "newGeneration", newGeneration
    return newGeneration


def matingProb(female, male):
    resistanceTrait = (female.x1 + female.x2)/2
    persistenceTrait = (male.y1 + male.y2)/2
    phi = 1/(1 + math.exp(-(persistenceTrait-resistanceTrait)))
    return phi

def makeBabies(female, male, siring, fecun, newPop):
    sex = numpy.random.binomial(1,0.5,1)
    print ("sex %s" % (sex))
    chromosome1 = chooseAlleles(female)
    chromosome2 = chooseAlleles(male)
    genotype = chromosome1 + chromosome2
#    print "genotype", genotype
#    print "length", len(genotype)
    mut = numpy.random.uniform(low = 0, high = 1, size = 4)
#    print "mut", mut
    std = 1
        ###Maybe could use popStats to give current standard deviation and use that???$######
    for m in range (0,len(mut)):
        if mut[m] <= args.mutation_host:
            tempGenotype = genotype[m] + numpy.random.normal(0, std)
            genotype[m] = tempGenotype[0]
                #####Ask Robert if we can just set genotype[m] = truncnorm.rvs etc#######
        else:
            pass
    offspring = Individual(sex[0], genotype[1], genotype[3], genotype[0], genotype[2])
    print "offspring", offspring
    newPop.append(offspring)        

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
 
def popStats(population):
    yVals = []
    xVals = []
    for i in range(0,len(population)):
        yVals.append(population[i].y1)
        yVals.append(population[i].y2)
        xVals.append(population[i].x1)
        xVals.append(population[i].x2)
    avgY = numpy.average(yVals)
    stdY = numpy.std(yVals)
    avgX = numpy.average(xVals)
    stdX = numpy.std(xVals)
    females = [ind for ind in population if ind.sex == 0]
    males = [ind for ind in population if ind.sex == 1]
####Possible way to save time by using this count from pickParents####
    print avgY, stdY, avgX, stdX, len(males), len(females)
                

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
