# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        currGhostStates = currentGameState.getGhostStates()
        currScaredTimes = [nowGhostState.scaredTimer for nowGhostState in currGhostStates]
        currFood = currentGameState.getFood()
        currPos = currentGameState.getPacmanPosition()

        "*** YOUR CODE HERE ***"
        
        #print "currFood \n", currFood
        
        num = 100000
        gdist = 0
        fdist = 0
        
        
        for g in newGhostStates:
            gdist += util.manhattanDistance(g.getPosition(), newPos)
            
        for f in newFood.asList():
            fdist += util.manhattanDistance(newPos, f)
        
        if gdist < 2:
            return -num
        
        if fdist == 0:
            return num**2

        #return (2.0 * gdist - fdist + successorGameState.getScore()**5)
        
        num = (1.0/(fdist)**0.5) - (2.0/(gdist)**2)

        #return successorGameState.getScore()**5 + num                   # Use with some tweaks in 'num' for a constant score of 609 on the autograder :) 
        return successorGameState.getScore()**2 + num

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    def maximus(self, state, depth, index):
        v = -float('inf')
        mmact = ''
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        
        actions = state.getLegalActions(index)
        for action in actions:
            nextState = state.generateSuccessor(index, action)
            maxVal = self.miniMax(nextState, depth, index + 1)
            if v < maxVal:
                v = maxVal
                mmact = action
            
        if depth == 1:
            return mmact
        else:
            return v
            
        
    def minimus(self, state, depth, index):
        v = float('inf')
        mmact = ''
        agentCount = state.getNumAgents()
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        actions = state.getLegalActions(index)
        for action in actions:
            nextState = state.generateSuccessor(index, action)
            if index == agentCount - 1:
                if depth == self.depth:
                    minVal = self.evaluationFunction(nextState)
                else:
                    minVal = self.miniMax(nextState, depth + 1, 0)
        
            else:
                minVal = self.miniMax(nextState, depth, index + 1)
            
            if v > minVal:
                v = minVal
                mmact = action
            
        return v
        
        
        
    def miniMax(self, state, depth, index):
        if index == 0:
            return (self.maximus(state, depth, index))                                      # Agent Pacman, call max function
        else:
            return (self.minimus(state, depth, index))                                      # Not Pacman, call min function
        
                      

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
    
        #print 'gameState.getLegalActions(0)', gameState.getLegalActions(0)
        
        state = gameState
        depth = 1
        index = 0
        v = self.miniMax(state, depth, index)
        return v
     
     
    
    

      
    
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    
    def alBeAgent(self, state, depth, index, alpha, beta):
        
        if index == 0:
            return (self.maxAgent(state, depth, index, alpha, beta))
        else:
            return (self.minAgent(state, depth, index, alpha, beta))
        
    
    def maxAgent(self, state, depth, index, alpha, beta):
        
        v = -float('inf')
        mmact = ''
        
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        
        actions = state.getLegalActions(index)
        for action in actions:
            nextState = state.generateSuccessor(index, action)
            val = self.alBeAgent(nextState, depth, index + 1, alpha, beta)
        
            if val > beta:
                return val
            
            if val > v:
                v = val
                mmact = action
                
            alpha = max(alpha,v)
            
            
        if depth == 1:
            return mmact
        else:
            return v
    
    def minAgent(self, state, depth, index, alpha, beta):
        
        v = float('inf')
        mmact = ''
        agentCount = state.getNumAgents()
        
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        actions = state.getLegalActions(index)
        for action in actions:
            nextState = state.generateSuccessor(index, action)
            if index == agentCount - 1:
                if depth == self.depth:
                    val = self.evaluationFunction(nextState)
                else:
                    val = self.maxAgent(nextState, depth + 1, 0, alpha, beta)
        
            else:
                val = self.minAgent(nextState, depth, index + 1, alpha, beta)
            
            if val < alpha:
                return val
            
            if val < v:
                v = val
                mmact = action
            beta = min(beta,v)
            
        return v


    
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        
        state = gameState
        depth = 1
        index = 0
        alpha = -float('inf')
        beta = float('inf')
        act = self.alBeAgent(state, depth, index, alpha, beta)
        return act
        #util.raiseNotDefined()
        
        
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def maximus(self, state, depth, index):
        v = -float('inf')
        mmact = ''
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        
        actions = state.getLegalActions(index)
        for action in actions:
            nextState = state.generateSuccessor(index, action)
            val = self.expectiMax(nextState, depth, index + 1)
            if v < val:
                v = val
                mmact = action
            
        if depth == 1:
            return mmact
        else:
            return v
            
        
    def minimus(self, state, depth, index):
        v = 0
        mmact = ''
        agentCount = state.getNumAgents()
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        actions = state.getLegalActions(index)
        prob = 1.0/ len(actions)
        for action in actions:
            nextState = state.generateSuccessor(index, action)
            if index == agentCount - 1:
                if depth == self.depth:
                    val = self.evaluationFunction(nextState)
                else:
                    val = self.expectiMax(nextState, depth + 1, 0)
        
            else:
                val = self.expectiMax(nextState, depth, index + 1)
            
            v += val * prob
            
        return v
        
        
        
    def expectiMax(self, state, depth, index):
        if index == 0:
            return (self.maximus(state, depth, index))
        else:
            return (self.minimus(state, depth, index))
        
                    

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        
        state = gameState
        depth = 1
        index = 0
        v = self.expectiMax(state, depth, index)
        return v

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION:
      Created a loop to calculate the Manhattan distance between Pacman position and food pellets,
      used a similar loop to calculate distance between ghost positions and Pacman.
      Calculated the minimum distance from food. Used the values of foodDistance, ghostDistance
      and Score to return a value for the utility function for each state.
    """
    "*** YOUR CODE HERE ***"
    
    """
    successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        currGhostStates = currentGameState.getGhostStates()
        currScaredTimes = [nowGhostState.scaredTimer for nowGhostState in currGhostStates]
        currFood = currentGameState.getFood()
        currPos = currentGameState.getPacmanPosition()
    """
    
    state = currentGameState                            
    newFood = state.getCapsules()
    #newFood = newFood.asList()                            #food pos as (x,y), already as list
    ghostPos = state.getGhostPositions()
    gstate = state.getGhostStates()
    scareTime = [g.scaredTimer for g in gstate] 
    foodCount = state.getNumFood()
    pacPos = state.getPacmanPosition()
    score = state.getScore()
    fdist = []
    gdist = 0
    fmin = 0
    if foodCount:
        food = lambda x: manhattanDistance(x, pacPos)
        for f in newFood:
            fdist.append(food(f))
        if fdist:
            fmin = min(fdist)
        
    else:
        return float('inf')
    
    for g in ghostPos:
        gdist = manhattanDistance(pacPos, g)
        #if g.scareTime > 20:
         #   return float('inf')
         
         
    if gdist == 0:
        return -float('inf')
    
    if fdist == 0:
        return float('inf')
        
        
    return -(1.0/(fmin+1.0))**3 + 1.0/(gdist + 1.0) + score**2

# Abbreviation
better = betterEvaluationFunction

