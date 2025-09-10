import torch
import math
import random
import model
import torch.distributions as dist

#We are using arbitrary units
#But we are traveling at 3units/reinforcement step
SPEED = 3
#Discount val
DIS_VAL = 0.9

def train_one_epoch(linearModel, optimzer):
    #Initialize the state values (floats)
    #pos = random.random() * 10.0
    pos = random.uniform(3, 7)
    angle = random.uniform(-10, 10)
    #set the model to train bec
    #uase we are adjusting the parameters to not crash
    linearModel.train()

    #Set crash variable so we can exit when crash
    crashed = False

    #intilizae the arrays for each step in order to get the overall loss
    logProbArray = []
    rwdArray = []

    #initialize steps to set a maximum of 100 steps where it is a success
    steps = 0

    while not crashed and steps <= 100:
        #brute force just getting the logits from the tensor
        normailzedAngle = angle / 90
        normailzedPos = (pos - 5.0) / 5.0
        stateTensor = torch.tensor((normailzedPos, normailzedAngle))
        mean, logStd = linearModel(stateTensor)

        #Getting the sampled value in order to get the action
        logStd = logStd.clamp(-20, 2)

        std = torch.exp(logStd)
        distribution = dist.Normal(mean, std)
        valTensor = distribution.sample()

        logP = distribution.log_prob(valTensor)

        #Get the new state given the action

        sampleVal = valTensor.item()
        angleChange = (sampleVal/3) * 45


        #Set constraints so as to not go backwards
        if angle + angleChange >= 45:
            angle = 45.0
        elif angle + angleChange <= -45:
            angle = -45.0
        else:
            angle +=angleChange

        #Moniter if crash
        pos+=math.sin(math.radians(angle)) * SPEED

        #Add to the arrays early so that if crash we take into account the last decision
        logProbArray.append(logP)
        
        #Add to steps
        steps+=1
        rwd = 0.1
        if pos >= 10:
            crashed = True
            print("Failed at" + str(steps))
            rwd = -10
        elif pos <= 0:
            crashed = True
            print("Failed at" + str(steps))
            rwd = -10
        if (steps == 100):
            print("Succeed")
            rwd = 10
        rwdArray.append(rwd)


    #Calcualte the loss
    loss = 0
    returns = []

    G_t = 0

    for r in reversed(rwdArray):
        G_t = r + G_t * DIS_VAL
        returns.insert(0, G_t)

    returns = torch.tensor(returns)

    #normalize ????
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

    for logP, ret in zip(logProbArray, returns):
        loss += -1 * logP * ret

    #Clear the optimzer 
    optimzer.zero_grad()

    #Idk if that works
    loss.backward()

    #Just apply the shit
    optimzer.step()

if __name__ == "__main__":
    #Initialize model
    linearModel = model.Model()

    #Initialize optimizer
    optimizer = torch.optim.Adam(linearModel.parameters(), lr=0.001)

    for i in range(100):
        train_one_epoch(linearModel, optimizer)









