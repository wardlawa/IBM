print "hello world"

import argparse
import sys


def main():
    print("#%s" % args)
#    populations = [generatePopulation()]
#    for i in range(args.generations):
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
    parser.add_argument("-K", "--carrying_Capacity", default=10, help="The carrying capacity of the population.", type=int)
    parser.add_argument("-G", "--generations", default=10, help="The maximum number of generations to run the simulation.", type=int)
    parser.add_argument("-b", "--birth_Rate", default=2, help="The intrinsic birth rate of females.", type=int)
    parser.add_argument("-c", "--cost_Persistence", default=0.5, help="The cost to males of possessing persistence value y.", type=float)
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

