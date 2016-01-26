print "hello world"

import argparse
import sys
import random
import numpy
import math
from collections import namedtuple
from scipy.stats import truncnorm

testFile = open('test.out','w')
sys.stdout = testFile

testFile2 = open('test.err','w')
sys.stderr = testFile2

Individual = namedtuple("Individual", "sex y1 y2 x1 x2")

def main():
    print("#%s" % args)
    populations = generatePopulation()
    for i in range(args.generations):
        newGenerations = pickParents(populations)
        populations = newGenerations
#    popStats(populations)
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
numpy.random.seed(123)
def generatePopulation(rands = None):
    population = []
    sex = numpy.random.binomial(1,0.5,args.population_size)
    #0 = female, 1 = male
#    print sex
    std = 1
    persis1 = truncnorm.rvs(a = (0 - args.male_persistence_trait)/std, b = numpy.inf, loc = args.male_persistence_trait, scale = std, size = args.population_size)
    persis2 = truncnorm.rvs(a = (0 - args.male_persistence_trait)/std, b = numpy.inf, loc = args.male_persistence_trait, scale = std, size = args.population_size)
#    print persis1
#    print persis2
    resis1 = truncnorm.rvs(a = (0 - args.female_resistance_trait)/std, b = numpy.inf, loc = args.female_resistance_trait, scale = std, size = args.population_size)
    resis2 = truncnorm.rvs(a = (0 - args.female_resistance_trait)/std, b = numpy.inf, loc = args.female_resistance_trait, scale = std, size = args.population_size)
#    print resis1
#    print resis2
    for i in range(args.population_size):
        population.append(Individual(sex[i], persis1[i], persis2[i], resis1[i], resis2[i]))
#        #put half of the TEs on each haplotype
#        num_of_tes_haplotype1 = numpy.random.poisson(args.initial_te_number/2)
#        num_of_tes_haplotype2 = numpy.random.poisson(args.initial_te_number/2)
#        #assign each TE a position
#        haplotype1 = sorted(random.sample(range(0,args.genome_size), num_of_tes_haplotype1))
#        haplotype2 = sorted(random.sample(range(0,args.genome_size), num_of_tes_haplotype2))
#        #calculate the fitness of this individual and put it into the population
#        population.append(calculateFitness(Individual(0.0, haplotype1, haplotype2)))
    print population
    return population

def pickParents(population):     
    females = [ind for ind in population if ind.sex == 0]
    males = [ind for ind in population if ind.sex == 1]
#    print females
#    print males
    newGeneration = []
    for i in range(0, len(females)):
        matedMales = []
        #matedMales keeps track of which males an individual female successfully mated with and needs to be cleared for each female
#        print "female", i
        numberOfMates = 0
        testNum =  numpy.random.choice(len(males),size = 2, replace = False)
#        print "testNum", testNum
        #print "length of testNum", len(testNum)
        for j in range(0, len(testNum)): 
            #print "testNum", testNum[j]
            test = matingProb(females[i], males[testNum[j]])
            #print "test", test
            randy = random.uniform(0,1)
            #print "randy", randy
            if randy <= test:
#                print "mated!"
                numberOfMates = numberOfMates + 1
#                print males[testNum[j]]                
                matedMales.append(males[testNum[j]])
            else:
                pass
#            print "num of mates", numberOfMates
#            print "matedMales", matedMales
        for k in range(0, len(matedMales)):
            makeBabies(females[i], matedMales[k], numberOfMates, newGeneration)
    print "newGeneration", newGeneration
    return newGeneration

def matingProb(female, male):
    resistanceTrait = (female.x1 + female.x2)/2
    persistenceTrait = (male.y1 + male.y1)/2
    phi = 1/(1 + math.exp(-(persistenceTrait-resistanceTrait)))
    return phi

def makeBabies(female, male, mates,newPop):
    mut = numpy.random.uniform(low = 0, high = 1, size = 4)
    for i in range (0,len(mut))
        if mut[i] <= args.mutation_host:
            female
        else
    sex = numpy.random.binomial(1,0.5,args.birth_rate)
    for i in range(args.birth_rate/mates):
        pickChromosome1 = random.uniform(0,1)
        if pickChromosome1 <= 0.5:
            chromosome1 = [female.x1, female.y1]
        else:
            chromosome1 = [female.x2, female.y2]
        pickChromosome2 = random.uniform(0,1)
        if pickChromosome2 < 0.5:
            chromosome2 = [male.x1, male.y1]
        else:
            chromosome2 = [male.x2, male.y2]
        offspring = Individual(sex[i],chromosome1[1], chromosome2[1], chromosome1[0], chromosome2[0])
        newPop.append(offspring)        

#def popStats(population):
    

def pickParent(population, other_parent = None, rands = None):
    '''
    >>> args = {'population_size':5, 'max_generations':10, 'cost_persistence':0.5, 'cost_harassment':0.02, 'encounter_rate':1, 'death_rate_males':0.2, 'death_rate_females':0.2, 'birth_rate':2}
    
    >>> population = [Individual(0, 1, 5),Individual(1, 2, 5),Individual(1, 2, 3), Individual(0, 2, 3), Individual(0, 1, 4)]
    
    #picks the highest fitness
    >>> pickParent(population, rands = [0.65])
    Individual(fitness=0.5, haplotype1=[1], haplotype2=[2, 3])
 
    #picks the highest fitness
    >>> parent1 = pickParent(population, rands = [1.0])
    >>> parent1
    Individual(fitness=0.5, haplotype1=[1], haplotype2=[2, 3])
    
    #would pick the highest again but fails the "other" test
    >>> pickParent(population, other_parent=parent1, rands = [0.7, 0.27])
    Individual(fitness=0.4, haplotype1=[], haplotype2=[])
 
    #without "other" defined this one should pick the highest fitness again
    >>> pickParent(population, rands = [0.7])#the rands here should match those in the test above
    Individual(fitness=0.5, haplotype1=[1], haplotype2=[2, 3])

    #picks the lowest fitness
    >>> pickParent(population, rands = [0.07])
    Individual(fitness=0.1, haplotype1=[1, 7, 8], haplotype2=[])
    
    #picks the lowest fitness
    >>> pickParent(population, rands = [0.0])
    Individual(fitness=0.1, haplotype1=[1, 7, 8], haplotype2=[])

    #picks highest when there is 1 negative fitness
    >>> population.append(Individual(-0.1, [], []))
    >>> pickParent(population, rands = [.9])
    Individual(fitness=0.5, haplotype1=[1], haplotype2=[2, 3])
    
    #picks lowest when there is 1 negative fitness
    >>> pickParent(population, rands = [0])
    Individual(fitness=-0.1, haplotype1=[], haplotype2=[])
   ''' 
    
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

#def matingProb(male, female):
#    phi = 1/(1 + math.exp(-(male.y-female.x)))
#    print phi
#    return phi

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
