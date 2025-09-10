import torch
import math
import random
import model

#We are using arbitrary units
#But we are traveling at 3units/reinforcement step
SPEED = 3
#Discount val
DIS_VAL = 0.9

if __name__ == "__main__":
    #Initialize model
    linearModel = model.Model()

    #Initialize optimizer
    optimzer = torch.optim.sgd(linearModel.parameters(), lr=0.01)

    #Initialize the state values (floats)
    pos = random.random() * 10.0
    angle = 0.0

    #set the model to train becuase we are adjusting the parameters to not crash
    linearModel.train()

    #Set crash variable so we can exit when crash
    crashed = False

    #intilizae the arrays for each step in order to get the overall loss
    logProbArray = []

    #initialize steps to set a maximum of 100 steps where it is a success
    steps = 0

    while not crashed or steps >= 100:
        #brute force just getting the logits from the tensor
        stateTensor = torch.tensor((pos, angle))
        outputList = linearModel(stateTensor).tolist()
        mean = outputList[0]
        logStd = outputList[1]

        #Getting the sampled value in order to get the action
        std = pow(math.e, logStd)
        sampledVal = random.normalvariate(mean, std)

        logP = (-1/2)*math.log10(2*math.pi)- math.log(std) - ((sampledVal - mean)**2)/(2*std**2)

        #Get the new state given the action
        angleChange = sampledVal * 90

        #Set constraints so as to not go backwards
        if angle + angleChange >= 90:
            angle = 90.0
        elif angle + angleChange <= -90:
            angle = -90.0
        else:
            angle +=angleChange

        #Moniter if crash
        pos+=math.sin(angle) * SPEED

        #Add to the arrays early so that if crash we take into account the last decision
        logProbArray.append(logP)

        if pos >= 10:
            crashed = True
        elif pos <= 0:
            crashed = True
        else:
            #Handle the non-crash scenerio
            # This involves calculating log(sampleVal)r(t) and adding it to the total loss
            # With r(t) = 1 when not crashing
            steps += 1
    
    #Calcualte the loss
    loss = 0
    for i in range(steps):

        #calculate the discounted reward (each step has reward 1)
        discountedReward = 0
        for j in range(i, steps):
            discountedReward+=pow(DIS_VAL, j-i)
        #increase loss by -logp*reward
        loss+=logProbArray[i]*discountedReward

    #mean loss
    meanLoss = loss/steps

    #Clear the optimzer 
    optimzer.zero_grad()

    #Idk if that works
    meanLoss.backward()

    #Just apply the shit
    optimzer.step()










