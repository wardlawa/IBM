import sys
import sqlite3

databaseName = "sexualConflictData.db"

def setup():
    try:
        con = sqlite3.connect(databaseName)
        cur = con.cursor()
        #make the tables if they don't exist
        cur.execute("CREATE TABLE IF NOT EXISTS input(ID integer PRIMARY KEY, seed double, populationSize integer, carryingCapacity integer, numberGenerations integer, birthRate integer, recombinationRate double, costPersistence double, costHarassment double, encounterRate double, deathRateMales double, deathRateFemales double, exploitationRate double, maleResistance double, femaleResistance double, tradeoff double, malePersistenceTrait double, femaleResistanceTrait double, sensitivity double, mutationRateHostMales double, mutationRateHostFemales double, mutationRateParasite double, windowLength integer, introduceSTD integer)");

        cur.execute("CREATE TABLE IF NOT EXISTS output(ID integer, generation integer, avgY double, varY double, avgX double, varX double, avgE double, varE double, maleSus integer, femaleSus integer, maleInf integer, femaleInf integer, PRIMARY KEY(ID, generation), FOREIGN KEY (ID) REFERENCES input (ID))");
        con.close()
    except sqlite3.Error, e:
        if con:
            con.rollback()
        sys.stderr.write("Database Error: %s\n" % e.args[0])
        sys.exit(1)
    return

def addData(seed, args):
    try:
        con = sqlite3.connect(databaseName)
        cur = con.cursor()
        argumentList = [seed, args.population_size, args.carrying_capacity, args.generations, args.birth_rate, args.recombination_rate, args.cost_persistence, args.cost_harassment, args.encounter_rate, args.death_rate_males, args.death_rate_females, args.exploitation, args.male_resistance, args.female_resistance, args.tradeoff, args.male_persistence_trait, args.female_resistance_trait, args.sensitivity, args.mutation_host_male, args.mutation_host_female, args.mutation_parasite, args.window_length, args.introduce_STD]
        cmd = "INSERT INTO input(seed, populationSize, carryingCapacity, numberGenerations, birthRate, recombinationRate, costPersistence, costHarassment, encounterRate, deathRateMales, deathRateFemales, exploitationRate, maleResistance, femaleResistance, tradeoff, malePersistenceTrait, femaleResistanceTrait, sensitivity, mutationRateHostMales, mutationRateHostFemales, mutationRateParasite, windowLength, introduceSTD) VALUES (%s)" % ",".join(map(str,argumentList))
        cur.execute(cmd)
        myID = cur.lastrowid
        infile = open(args.outfile)
        data = []

        for line in infile:
            sline = line.split(" ")
            if len(sline) == 11:
                data.append([myID] + sline)
            else:
                sys.stderr.write("Weird line length:\n\t%s\n" % line)
        cur = con.cursor()
        cmd = "INSERT INTO output(ID, generation, avgY, varY, avgX, varX, avgE, varE, maleSus, femaleSus, maleInf, femaleInf) VALUES (%s)" % ", ".join(["?"]*12)
        cur.executemany(cmd, data)

        infile.close()
        con.commit()
        con.close()
    except sqlite3.Error, e:
        if con:
            con.rollback()
        sys.stderr.write("Database Error: %s\n" % e.args[0])
        sys.exit(1)
    return
