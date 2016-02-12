#FUNCTIONS

timePlotter <- function(data, myColumn, IDs, colors, myWidth, yLabel, yLims = NULL, fun = (function(x){x})){
  sdhs = c()
  #colors = c()
  #yLims = c(min(fun(data[IDs[1],myColumn])), max(fun(data[IDs[1],myColumn])))
  
  if (is.null(yLims)) {
    yLims = c(min(fun(data[,myColumn])), max(fun(data[,myColumn])))
  } 
  
  subData = data[data$ID == IDs[1],]
  plot(subData$generation,fun(subData[,myColumn]), type = "l", xlim = c(0,max(data$generation)), ylim = yLims, col = colors[1], lwd = myWidth, xlab = "Generation", ylab = yLabel)
  sdhs = append(sdhs, paste(subData$selectionAgainstDelMutHigh[1],subData$tempAutoLevel[1]))
  #colors = append(colors, subData$colors[1])
  for(i in 2:length(IDs)){
    subData = data[data$ID == IDs[i],]
    sdhs = append(sdhs, paste(subData$selectionAgainstDelMutHigh[1],subData$tempAutoLevel[1]))
    
    points(subData$generation,fun(subData[,myColumn]), type = "l", col = colors[i], lwd = myWidth)
  }
  #legend("topright", legend = sdhs, col = colors, bty = "n", lty = 1, lwd = myWidth, ncol = 2)
}

#list(whatYouWantTheColumnToBeCalledInTheNewTable = data$whatTheColumnWasCalledInTheOldTable)
averageReplicates <- function(data, myColumn, fun){
  aggData = aggregate(data[,myColumn], list(seed = data$seed, populationSize = data$populationSize, carryingCapacity = data$carryingCapacity, numberGenerations = data$numberGenerations, birthRate = data$birthRate, recombinationRate = data$recombinationRate, costPersistence = data$costPersistence, costHarassment = data$costHarassment, encounterRate = data$encounterRate, deathRateMales = data$deathRateMales, deathRateFemales = data$deathRateFemales, exploitationRate = data$exploitationRate, maleResistance = data$maleResistance, femaleResistance = data$femaleResistance, tradeoff = data$tradeoff, malePersistenceTrait = data$malePersistenceTrait, femaleResistanceTrait = data$femaleResistanceTrait, sensitivity = data$sensitivity, mutationRateHostMales = data$mutationRateHostMales, mutationRateHostFemales = data$mutationRateHostFemales, mutationRateParasite = data$mutationRateParasite, windowLength = data$windowLength, introduceSTD = data$introduceSTD, generation = data$generation, avgY = data$avgY, varY = data$varY, avgX = data$avgX, varX = data$varX, avgE = data$avgE, varE = data$varE, maleSus = data$maleSus, femaleSus = data$femaleSus, maleInf = data$maleInf, femaleInf = data$femaleInf), fun)
}

#################################

data1 = read.csv("/Users/alisonwardlaw/SkyDrive/Professional/Research/sexualConflict/IBM/sexualConflictData.csv", header = FALSE)
names(data1) = c("ID", "seed", "populationSize", "carryingCapacity", "numberGenerations", "birthRate", "recombinationRate", "costPersistence", "costHarassment", "encounterRate", "deathRateMales", "deathRateFemales", "exploitationRate", "maleResistance", "femaleResistance", "tradeoff", "malePersistenceTrait", "femaleResistanceTrait", "sensitivity", "mutationRateHostMales", "mutationRateHostFemales", "mutationRateParasite", "windowLength", "introduceSTD", "ID2", "generation", "avgY", "varY", "avgX", "varX", "avgE", "varE", "maleSus", "femaleSus", "maleInf", "femaleInf")
#sdhs = c()
IDs = unique(data1$ID)
colors = rainbow(length(IDs))

setwd("/Users/alisonwardlaw/SkyDrive/Professional/Research/sexualConflict/IBM")
dataTest = read.table("test.out", header=FALSE)
colnames(dataTest) =c("replicate","generation", "avgY", "stdY", "avgX", "stdX", "avgE", "stdE", "mSus", "fSus", "mInf", "fInf")
