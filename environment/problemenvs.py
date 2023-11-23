from scienceworld import ScienceWorldEnv
from models.buildenvmodel import *

class world_exception(Exception):
    def __init__(self, message={}):            
        # Call the base class constructor with the parameters it needs
        super().__init__(message)
        self.error = message


class scienv():
    def __init__(self,task = "1-1", objective = None):
        self.env = ScienceWorldEnv(task)
        self.MINREWARD = -100
        self.MAXREWARD = 100
        obs1, info1 = self.env.reset()
        self.model = envmodel()
        self.trace = []
        self.actiontrace = []
        self.reward = -1
        self.totalreward = 0
        self.goalreached = False
        predescription = "An AI agent helping execute a science experiment in a simulated environment with limited number of objects and actions available at each step. "
        prioraxioms = """
        an agent situated in textual task environment. Generate a sequence of actions to meet the objective.
        you may reset the environment if you feel stuck and need to start over.
        FOCUS is a extremely critical action that can be only used the number of times 'focus' is mentioned in the task description and in the exact same sequence. Using it more than that or inappropiately (such as on a wrong object) will terminate the session and the task will be rendered as incomplete. focus can be used on the object which is available in current state.
        Do not make up new actions or objects. If some events need some time to occur after some action is taken then take the action wait to observe the effect after some time.
        
        DO NOT TAKE ANY ACTION ON ANY OBJECT that is NOT IN ACCESSIBLE OBJECTS in CURRENT STATE
        
        Here are the following set of allowed actions. where OBJ should be replaced by any object that you can find in your current state.
        Set of parameter values:
          """+str(self.env.getPossibleActions())
          
        if objective == None:
            objective = self.env.getTaskDescription()
        self.environment = {"description": predescription + objective, "objective": objective, "prior axioms": prioraxioms, "belief axioms": "", "current state": self.getstate()}
        
        self.examples = """
        Example 1:
           ["look around", "open door to greenhouse"]
        Example 2:
           ["go door to hallway", "open door to kitchen"]
        """
        return
        
    def getstate(self):
        obs, _,_,_ = self.env.step("look around")
        state = """
        Currently you see the following things:
          """+ obs+"""
               
        Currently you can access the following objects::
          """+str(self.env.getPossibleObjects())+"""
               
        The agent have following things in its inventory.
        """+ str(self.env.inventory()).replace("In your inventory, you see:","")
        
        return state
        
    def getfeedback(self):
        feedback = self.success_map('reward',self.totalreward)
        self.totalreward = 0
        #if max(self.reward) == 0:
        #    reward = -0.5
        #else:
        #    reward = max(self.reward)
        #obs1, info1 = self.env.reset()
        return  feedback#"Observation: "+'\n'.join([i.replace("\n", "; ") for i in self.observation])+" External feedback: "+ str(self.success_map('reward',self.reward)), self.getstate()
    
    def traceact(self):
        for actiontext in self.actiontrace:
            observation, reward, self.goalreached, info = self.env.step(actiontext)
        return observation
    
    def act(self,actiontext):
        self.actiontrace.append(actiontext)
        observation, reward, self.goalreached, info = self.env.step(actiontext)
            
        if actiontext == "reset task":
            self.totalreward = 0
        else:
            self.totalreward += reward

        if observation == "No known action matches that input.":
            self.trace.append({"action":actiontext, "observation" : observation.replace("\n", "; "), "state": self.getstate(), "reward":float('-Inf'),"totactions": 0})
            #raise world_exception("invalid action")
        elif actiontext.startswith("focus") and reward < 0:
            observation += " You focused on the wrong object and that resulted in a critical mistake the environment was reset"
            self.goalreached = False
            self.trace.append({"action":actiontext, "observation" : observation.replace("\n", "; "), "state": self.getstate(), "reward":float('-Inf'),"totactions":0})  #( "{ Action taken: "+actiontext+" ; Observation : "+ observation.replace("\n", "; ")+"}")
            return self.observation#raise world_exception("invalid action")
        else:
            normalizedreward = 2*(reward - self.MINREWARD) /(self.MAXREWARD - self.MINREWARD) - 1  ## in -1 to 1 scale
            totalpossibleactions = len(self.env.getValidActionObjectCombinationsWithTemplates())
            self.trace.append({"action":actiontext, "observation" : observation.replace("\n", "; "), "state": self.getstate(), "reward": normalizedreward, "totactions": totalpossibleactions})        
        return observation
    
    
    def updatemodel(self):
        self.model.parseacpt_trace(self.trace,self.environment["current state"])
        self.environment["current state"] = self.trace[-1]["state"]
        self.model.updatevalue()
        #self.trace = []
    
    def getinstructions(self):
        return self.model.getplandetails(self.environment["current state"])
        
        
    def checkgoal(self):
        #obs1, info1 = self.env.reset()
        self.observation = []
        self.reward = -1
        print ("total reward", self.totalreward)
        
        return self.goalreached
    
    def success_map(self, metric, score):
        feedback = ''
        if metric == 'reward':
            if score < -50:
                feedback += "The agent made critical mistakes and the task was terminated and reset."
            if score < 0:
                feedback += "The agent performed very poorly and could not make any critical progress."
            if score >= 0 and score < 20:
                feedback += "The agent performed poorly and made some progress but not enough to solve the task."
            if score >= 20 and score < 50:
                feedback += "The agent performed moderately and made some critical progress but not enough to solve the task."
            if score >= 50 and score < 90:
                feedback += "The agent performed very well and made significant critical progress but not enough to solve the task."
            if score >= 90 and score < 100:
                feedback += "The agent performed exceptionally well and made significant critical progress, was just slight away from solving the task."
            if score == 100:
                feedback += "The agent performed exceptionally well and successfully solved the task."
        
        return feedback