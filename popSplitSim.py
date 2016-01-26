import sys
import argparse
import numpy
from collections import namedtuple
import random

Individual = namedtuple("Individual", "fitness haplotype1 haplotype2")

def main():
    print("#%s" % args)
    populations = [generatePopulation()]
    for i in range(args.generations):
        #check if we need to split the population
        if i == args.split_generation:
            populations.append(splitPopulation(populations[0]))

        for j, population in enumerate(populations):
            selfing_rate = args.selfing_rate if j == 0 else args.new_selfing_rate
            next_generation = []
            #make all the offspring
            for k in range(args.population_size):
                next_generation.append(makeOffspring(population, selfing_rate))
            #set this as the main generation and output data for it
            populations[j] = next_generation
            stats = calculateStats(next_generation)
            outputData(stats, population_name=j, generation=i)
        

def parseArgs():
    parser = argparse.ArgumentParser(description="Simulates a population who's individuals' fitness depend on their TE copy number.\
    There will be a split in the population leading to two populations with different selfing rates.") 
    parser.add_argument("-N", "--population_size", default=10, help="The population size of the starting population.", type=int)
    parser.add_argument("-G", "--generations", default=10, help="The maximum number of generations to run the simulation.", type=int)
    parser.add_argument("-T", "--transposition_rate", default=0.01, help="The proability of any one TE transposing.", type=float)
    parser.add_argument("-S", "--genome_size", default=10, type=int, help="The size of each individual's genome in number of TE insertion sites.")
    parser.add_argument("-R", "--recombination_rate", default=0.05, type=float, help="The mean number of recombination events per genome when a gamete is made. The actual number will be sampled from a poisson distribution. A minimum of 1 recombination event will happen with every gamete.")
    parser.add_argument("-F", "--selfing_rate", default=0.1, type=float, help="The selfing rate of the initial population.")
    parser.add_argument("-I", "--initial_te_number", default=1, type=int, help="The initial mean number of TEs each individual has in the population. The actual number will be sampled form a Poisson distribution.")
    parser.add_argument("-C", "--fitness_cost", default=0.01, type=float, help="The fitness cost of one deleterious TE insertion.")
    parser.add_argument("-E", "--fitness_interactions", default=2, type=float, help="The fitness interactions of TEs. The total fitness equation is: w = 1-(C*g)^E where g is the number of deleterious TE insertions.")
    parser.add_argument("-W", "--fitness_function", default="ALL", choices=["ALL", "HOMO", "HET"], help="The fitness function to be used. ")
    parser.add_argument("-K", "--split_generation", default=5, type=int, help="The generation where the population split occurs.")
    parser.add_argument("-f", "--new_selfing_rate", default=0.9, type=float, help="The selfing rate of the new population after the split.")
    parser.add_argument("-b", "--bottleneck_size", default=3, type=int, help="The number of unique individuals that will seed the new population after the split.")
    
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
def generatePopulation(rands = None):
    population = []
    for i in range(args.population_size):
        #put half of the TEs on each haplotype
        num_of_tes_haplotype1 = numpy.random.poisson(args.initial_te_number/2)
        num_of_tes_haplotype2 = numpy.random.poisson(args.initial_te_number/2)
        #assign each TE a position
        haplotype1 = sorted(random.sample(range(0,args.genome_size), num_of_tes_haplotype1))
        haplotype2 = sorted(random.sample(range(0,args.genome_size), num_of_tes_haplotype2))
        #calculate the fitness of this individual and put it into the population
        population.append(calculateFitness(Individual(0.0, haplotype1, haplotype2)))

    return population

'''
Given a population creates a new population consisting of a copy of a small subset of the inital population, and offspring generated from those individuals.
'''
def splitPopulation(population, rands = None):
    #sample down to our bottleneck size for the split
    new_population_bottleneck = random.sample(population, args.bottleneck_size)
    #make the new population using only the bottlenecked ones as parents
    new_population = [makeOffspring(new_population_bottleneck, selfing_rate=args.new_selfing_rate) for i in range(args.population_size)]
    
    return new_population

'''
Takes a population and an optional selfing rate, and picks two (or one) parents to mate and create an offspring which is returned.
'''
def makeOffspring(population, selfing_rate = None, rands = None):
    parent1, parent2 = random.sample(population, 2)

    #check if we're going to self
    if random.random() <= args.new_selfing_rate:
        parent2 = parent1

    new_individual = Individual(0.0, makeGamete(parent1), makeGamete(parent2))
    new_individual_transposed = transpose(new_individual)
    new_individual_final = calculateFitness(new_individual_transposed)

    return new_individual_final

'''
Given a population picks a parent stochastically with weights scaled by each individual's fitness.
'''
def pickParent(population, other_parent = None, rands = None):
    '''
    >>> args = {'population_size':5, 'max_generations':10, 'transposition_rate':0.5, 'recombination_rate':1, 'genome_size':5, 'selfing_rate':0.5}
    
    >>> population = [Individual(0.5, [1], [2,3]),Individual(0.2, [1,6], [4]),Individual(0.1, [1,7,8], []), Individual(0.4, [], [])]
    
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

'''
Given an individual this function returns a 'gamete', a list of TE positions, after randomly
recombining the individual's two haplotypes.
'''
def makeGamete(individual, rands = None):
    '''
    >>> args = {'population_size':5, 'max_generations':10, 'transposition_rate':0.5, 'recombination_rate':1, 'genome_size':5, 'selfing_rate':0.5}
    
    #generates a gamete with only 1 recombination 
    >>> individual = Individual(0.0, [1,2,4,7], [2,4,9,15])
    >>> makeGamete(individual, rands=[1,3,2])
    [2, 4, 7]
    
    #generates a gamete with only 1 recombination 
    >>> individual = Individual(0.0, [1,2,3,4,7], [2,4,9,15])
    >>> makeGamete(individual, rands=[1,3,1])
    [1, 2, 3, 4, 9, 15]
    
    #generates a gamete with only 1 recombination 
    >>> individual = Individual(0.0, [1,2,4,7], [2,4,9,15])
    >>> makeGamete(individual, rands=[1,3,1])
    [1, 2, 4, 9, 15]

    #generates a gamete with only 1 recombination 
    >>> individual = Individual(0.0, [1], [2,4])
    >>> makeGamete(individual, rands=[0,1,1])
    [1, 2, 4]

    #generates a gamete with only 1 recombination 
    >>> individual = Individual(0.0, [1], [2,4])
    >>> makeGamete(individual, rands=[0,1,2])
    []

    #generates a gamete with 1 recombination point that creates a new haplotype that still matches one of the old ones
    >>> individual = Individual(0.0, [1], [2,4])
    >>> makeGamete(individual, rands=[1,5,1])
    [1]

    #generates a gamete with 1 recombination point that creates a new haplotype that still matches one of the old ones
    >>> individual = Individual(0.0, [1], [2,4])
    >>> makeGamete(individual, rands=[1,5,2])
    [2, 4]

   #generates a gamete with 1 recombination that creates a new haplotype with 2 TEs
    >>> individual = Individual(0.0, [1], [2,4])
    >>> makeGamete(individual, rands=[1,3,1])
    [1, 4]
    
    #generates a gamete with 1 recombination point that creates a haplotype with 2 TEs
    >>> individual = Individual(0.0, [1], [2,4])
    >>> makeGamete(individual, rands=[1,2,1])
    [1, 4]
    
    #generates a gamete with 1 recombination point that creates a haplotype with 1 TEs
    >>> individual = Individual(0.0, [1], [2,4])
    >>> makeGamete(individual, rands=[1,2,2])
    [2]
        
    #generates a gamete with 2 recombination points to create a haplotype with 2 TEs
    >>> individual = Individual(0.0, [1], [1,2,4])
    >>> makeGamete(individual, rands=[2,2,4,1])
    [1, 4]

    #generates a gamete with 2 recombination points that creates a haplotype with 2 TEs
    >>> individual = Individual(0.0, [1], [2,4])
    >>> makeGamete(individual, rands=[2,2,3,1])
    [1]
    
    #generates a gamete with 2 recombination points that creates a haplotype with 1 TE
    >>> individual = Individual(0.0, [1], [2,4])
    >>> makeGamete(individual, rands=[2,2,3,2])
    [2, 4]
    
    '''

    num_crosses = rands.pop(0) if rands else numpy.random.poisson(args.recombination_rate*args.genome_size)
    num_crosses = max(1, num_crosses)
    new_haplotype1 = individual.haplotype1[:]
    new_haplotype2 = individual.haplotype2[:]
    
    for i in range(num_crosses):
        crossover_point = rands.pop(0) if rands else random.randint(0, args.genome_size-1) 
        haplotype1_index, haplotype2_index = getFirstGreater(new_haplotype1, crossover_point), getFirstGreater(new_haplotype2, crossover_point)
        
        new_haplotype1, new_haplotype2 = new_haplotype1[:haplotype1_index] + new_haplotype2[haplotype2_index:],\
                                         new_haplotype2[:haplotype2_index] + new_haplotype1[haplotype1_index:]

    gamete_choice = rands.pop(0) if rands else random.choice([1,2])
    if gamete_choice == 1:
        return new_haplotype1
    else:
        return new_haplotype2

'''
A helper function for makeGamere() that gives the first index in a list of an item greater than another given value.
'''
def getFirstGreater(list, value):
    '''
    >>> getFirstGreater([1,2,3,4], 5)
    4

    >>> getFirstGreater([1,2,3,4], 0)
    0

    >>> getFirstGreater([1,2,3,4], 2)
    2
    '''
    i = 0
    for i, x in enumerate(list):
        if x > value:
            return i
    return i+1

'''
Given an individudal this function returns a new individual with TEs randomly transposed across the two haplotypes.
Number of transposition events is scaled by the number of TEs in the genome.
'''
def transpose(individual, rands = None):
    '''
    >>> args.genome_size = 5

    #no TEs transpose
    >>> individual = Individual(0.0, [1], [2,4])
    >>> transpose(individual, rands=[0])
    Individual(fitness=0.0, haplotype1=[1], haplotype2=[2, 4])

    #inserts a new TE on haplotype1
    >>> individual = Individual(0.0, [1], [2,4])
    >>> transpose(individual, rands=[1, 1, 2])
    Individual(fitness=0.0, haplotype1=[1, 2], haplotype2=[2, 4])

    #inserts a new TE on haplotype2
    >>> individual = Individual(0.0, [1], [2,3])
    >>> transpose(individual, rands=[1, 2, 4])
    Individual(fitness=0.0, haplotype1=[1], haplotype2=[2, 3, 4])

    #inserts new TEs on each haplotype
    >>> individual = Individual(0.0, [1], [2,4])
    >>> transpose(individual, rands=[3, 2, 1, 1, 2, 1, 3])
    Individual(fitness=0.0, haplotype1=[1, 2, 3], haplotype2=[1, 2, 4])

    #No spots left!
    >>> individual = Individual(0.0, [0,1,2,3,4], [0,1,2,3,4])
    >>> transpose(individual, rands=[3])
    Individual(fitness=0.0, haplotype1=[0, 1, 2, 3, 4], haplotype2=[0, 1, 2, 3, 4])

    #Not enough spots left for number chosen!
    >>> individual = Individual(0.0, [0, 1, 2, 3, 4], [0,2,4])
    >>> transpose(individual, rands=[5, 2, 1, 2, 3])
    Individual(fitness=0.0, haplotype1=[0, 1, 2, 3, 4], haplotype2=[0, 1, 2, 3, 4])

    #Run out of spots on one haplotype!
    >>> individual = Individual(0.0, [0, 1, 2, 3, 4], [2,4])
    >>> transpose(individual, rands=[2, 2, 1, 2, 3])
    Individual(fitness=0.0, haplotype1=[0, 1, 2, 3, 4], haplotype2=[1, 2, 3, 4])

 '''
    if not rands:
        num_new_tes = numpy.random.poisson((len(individual.haplotype1)+len(individual.haplotype2))*args.transposition_rate)
    else:
        num_new_tes = rands.pop(0)

    #get a list of all positions that currently DONOT have a TE
    #I'm not sure how slow this will be...
    haplotype1_spots = set(individual.haplotype1).symmetric_difference(set(range(0, args.genome_size)))
    haplotype2_spots = set(individual.haplotype2).symmetric_difference(set(range(0, args.genome_size)))
    
    #check to make sure we have enough spots left to insert the new ones
    num_new_tes = min(num_new_tes, len(haplotype1_spots)+len(haplotype2_spots))
    
    #no new TEs (or there are no empty spots)
    if num_new_tes == 0:
        return individual

    #makes a list of the haplotypes with spots left
    spots = [None]
    spots.append(haplotype1_spots) if haplotype1_spots else spots.append(None)
    spots.append(haplotype2_spots) if haplotype2_spots else spots.append(None)
    for i in range(num_new_tes):
        #chose a haplotype that has spots left
        haplotype = rands.pop(0) if rands else random.choice([i for i, x in enumerate(spots) if x != None])
        position = rands.pop(0) if rands else random.choice(list(spots[haplotype]))
        
        #add the new TE
        individual[haplotype].append(position)
        
        #remove that as a valid choice for future TEs
        spots[haplotype].remove(position)
        if not spots[haplotype]: spots[haplotype] = None

    #sort the TE positions
    individual = Individual(0.0, sorted(individual.haplotype1), sorted(individual.haplotype2))
    return individual

'''
Given an individual this function returns a new Individual with the fitness corectly calculated.
Fitness is calculated as:
w = 1 - (C*g)^E
where C is the cost of a TE insertion, E is the interaction effects of multiple TEs, and g is the number
of deleterious TEs as determined by the fitness mode.
'''
def calculateFitness(individual):
    '''
    #testing ALL TEs are bad
    >>> args.fitness_function = "ALL"

    #fitness with 3 TEs
    >>> individual = Individual(0.0, [1], [1,4])

    >>> calculateFitness(individual)
    Individual(fitness=0.9991, haplotype1=[1], haplotype2=[1, 4])
    
    #fitness with 3 TEs but a previous fitness assigned
    >>> individual = Individual(0.432, [1], [1,4])

    >>> calculateFitness(individual)
    Individual(fitness=0.9991, haplotype1=[1], haplotype2=[1, 4])
     
    #fitness with 5 TEs on only 1 haplotype
    >>> individual = Individual(0.0, [], [3,1,5,2,4])

    >>> calculateFitness(individual)
    Individual(fitness=0.9975, haplotype1=[], haplotype2=[3, 1, 5, 2, 4])

    #fitness with 3 TEs on only 1 haplotype
    >>> individual = Individual(0.0, [2, 1, 3], [])

    >>> calculateFitness(individual)
    Individual(fitness=0.9991, haplotype1=[2, 1, 3], haplotype2=[])
 
    #fitness with 0 TEs
    >>> individual = Individual(0.0, [], [])

    >>> calculateFitness(individual)
    Individual(fitness=1.0, haplotype1=[], haplotype2=[])

    #######testing HET TEs are bad
    >>> args.fitness_function = "HET"
    
    #fitness with 3 TEs
    >>> individual = Individual(0.0, [1], [1,4])

    >>> calculateFitness(individual)
    Individual(fitness=0.9999, haplotype1=[1], haplotype2=[1, 4])
    
    #fitness with 3 TEs but a previous fitness assigned
    >>> individual = Individual(0.432, [1], [1,4])

    >>> calculateFitness(individual)
    Individual(fitness=0.9999, haplotype1=[1], haplotype2=[1, 4])
     
    #fitness with 5 TEs on only 1 haplotype
    >>> individual = Individual(0.0, [], [3,1,5,2,4])

    >>> calculateFitness(individual)
    Individual(fitness=0.9975, haplotype1=[], haplotype2=[3, 1, 5, 2, 4])

    #fitness with 3 TEs on only 1 haplotype
    >>> individual = Individual(0.0, [2, 1, 3], [])

    >>> calculateFitness(individual)
    Individual(fitness=0.9991, haplotype1=[2, 1, 3], haplotype2=[])
 
    #fitness with 0 TEs
    >>> individual = Individual(0.0, [], [])

    >>> calculateFitness(individual)
    Individual(fitness=1.0, haplotype1=[], haplotype2=[])

    ########testing HOMO TEs are bad
    >>> args.fitness_function = "HOMO"
    
    #fitness with 3 TEs
    >>> individual = Individual(0.0, [1], [1,4])

    >>> calculateFitness(individual)
    Individual(fitness=0.9999, haplotype1=[1], haplotype2=[1, 4])
    
    #fitness with 3 TEs but a previous fitness assigned
    >>> individual = Individual(0.432, [1], [1,4])

    >>> calculateFitness(individual)
    Individual(fitness=0.9999, haplotype1=[1], haplotype2=[1, 4])
     
    #fitness with 5 TEs on only 1 haplotype
    >>> individual = Individual(0.0, [], [3,1,5,2,4])

    >>> calculateFitness(individual)
    Individual(fitness=1.0, haplotype1=[], haplotype2=[3, 1, 5, 2, 4])

    #fitness with 3 TEs on only 1 haplotype
    >>> individual = Individual(0.0, [2, 1, 3], [])

    >>> calculateFitness(individual)
    Individual(fitness=1.0, haplotype1=[2, 1, 3], haplotype2=[])
 
    #fitness with 0 TEs
    >>> individual = Individual(0.0, [], [])

    >>> calculateFitness(individual)
    Individual(fitness=1.0, haplotype1=[], haplotype2=[])

    '''
    if args.fitness_function == "ALL":
        number_deleterious_tes = len(individual.haplotype1)+len(individual.haplotype2)
    elif args.fitness_function == "HET":
        number_deleterious_tes = len(set(individual.haplotype1).symmetric_difference(set(individual.haplotype2)))
    elif args.fitness_function == "HOMO":
        number_deleterious_tes = len(set(individual.haplotype1) & set(individual.haplotype2))
    
    fitness = 1 - (args.fitness_cost*number_deleterious_tes)**args.fitness_interactions
    return Individual(fitness, individual.haplotype1, individual.haplotype2)

'''
Given a population calculates the stats to output:
* mean number of TEs
* std dev in number of TEs
* mean fitness
* stdev in fitness
'''
def calculateStats(population):
    '''
    >>> args = {'population_size':5, 'max_generations':10, 'transposition_rate':0.5, 'recombination_rate':1, 'genome_size':5, 'selfing_rate':0.5}
   
    #all individuals have the same number of TEs and fitness
    >>> population = [Individual(0.5, [1], [2,3]),Individual(0.5, [1,6], [4]),Individual(0.5, [1,7,8], [])]
   
    >>> result = calculateStats(population)
    >>> result == {'mean_fitness': 0.5, 'deviation_fitness': 0.0, 'mean_te_num': 3.0, 'deviation_te_num': 0.0}
    True

    #All individuals have different #TEs adn fitness
    >>> population = [Individual(0.5, [1], [2,3]),Individual(0.4, [1], [4]),Individual(0.3, [1], []),]
    >>> result = calculateStats(population)
    >>> result ==  {'mean_fitness': 0.39999999999999997, 'deviation_fitness': 0.081649658092772609, 'mean_te_num': 2.0, 'deviation_te_num': 0.81649658092772603}
    True

    #Individuals haev even more variation in fitness and TE number
    >>> population = [Individual(0.5, [1], [2,3]),Individual(0.9, [1,3,2,5,7], [4]),Individual(0.0, [1], []),]
    >>> result = calculateStats(population)
    >>> result ==  {'mean_fitness': 0.46666666666666662, 'deviation_fitness': 0.36817870057290869, 'mean_te_num': 3.3333333333333335, 'deviation_te_num': 2.0548046676563256}
    True
    '''
    stats = {}
    fitnesses = [individual.fitness for individual in population]
    stats['mean_fitness'] = numpy.mean(fitnesses)
    stats['deviation_fitness'] = numpy.std(fitnesses)

    te_nums = [len(individual.haplotype1) + len(individual.haplotype2) for individual in population]
    stats['mean_te_num'] = numpy.mean(te_nums)
    stats['deviation_te_num'] = numpy.std(te_nums)

    return stats

'''
Outputs a string listing the stats for this population in this generation.
'''
def outputData(stats, population_name, generation):
    if generation == 0:
        print("Generation Population Mean_TE SD_TE Mean_Fitness SD_Fitness")
    print("%s %s %s %s %s %s" % (generation, population_name, stats['mean_te_num'], stats['deviation_te_num'],\
           stats['mean_fitness'], stats['deviation_fitness'] ))


if __name__ == "__main__":
    args = parseArgs()
    if not args.test:
        import time
        start = time.time()
        main()
        end = time.time()
        m, s = divmod(end-start, 60)
        h, m = divmod(m, 60)
        sys.stderr.wrtie("TIME: %s:%s:%s\n" % (h,m,s))
    else:
        import doctest
        doctest.testmod()
