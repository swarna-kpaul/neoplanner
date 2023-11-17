import random
import string
import math 
alpha = 0.2
gamma = 0.8
TDTHRESHOLD = 0.2
ucb_c = 1.5

class envmodel():
    def __init__(self):
        self.rootnodeid = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        self.invalidnodeid = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        self.DEFAULTVALUE = 0.5
        self.statespace = {"nodes":{
                                   self.rootnodeid: { "state": "start",
                                                    "value" : self.DEFAULTVALUE,
                                                    "trial" : 0,
                                                    "ucb": 0
                                                    },
                                   self.invalidnodeid: { "state": "invalid",
                                                    "value" : self.DEFAULTVALUE,
                                                    "trial" : 0,
                                                    "ucb": 0
                                                       }
                                   },
                            "edges": { self.rootnodeid+"-"+self.rootnodeid:
                                   {"action": "dummy","from":self.rootnodeid,"to":self.rootnodeid, "reward" : 0}
                                   }}
        self.totaltrials = 0
        self.defaultucb = self.DEFAULTVALUE + 1
        
        
    def addaction(self,action,startstate, endstate, reward):
        ############## add retrieve start state
        startnodeid = [id for id,node in self.statespace["nodes"].items() if node["state"] == startstate] 
        if startnodeid:
            startnodeid = startnodeid[0]
        else:
            startnodeid = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
            self.statespace["nodes"][startnodeid] = {"state": startstate, "value" : self.DEFAULTVALUE,"trial" : 0,}
            ########## add root node edge
            #rootnodeid = self.statespace["nodes"]["start"]["id"]
            self.statespace["edges"][self.rootnodeid+"-"+startnodeid+"-"+"dummy"] = {"action": "dummy", "reward": 0,"from":self.rootnodeid,"to":startnodeid}
        
        ############## add retrieve end state
        if reward == float('-Inf'):
            ############3 invalid action
            endnodeid = self.invalidnodeid
        else:
            endnodeid = [id for id,node in self.statespace["nodes"].items() if node["state"] == endstate]         
            if endnodeid:
                endnodeid = endnodeid[0]
                self.statespace["nodes"][endnodeid]["trial"] +=1
            else:
                endnodeid = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
                self.statespace["nodes"][endnodeid] = {"state": endstate, "value" : self.DEFAULTVALUE,"trial" : 0}
        
        ############## add update edge
        #if startnodeid+"-"+endnodeid+"-"+action in self.statespace["edges"]:
            ########## action edge already present
        #    self.statespace["edges"][startnodeid+"-"+endnodeid]["reward"] = reward
        #else:
        self.statespace["edges"][startnodeid+"-"+endnodeid] = {"action": action, "reward": reward,"from":startnodeid,"to":endnodeid}
        self.totaltrials += 1
        
    def parseacpt_trace(self,ACPT,startstate):    
        for actionperception in ACPT:
            self.addaction(actionperception["action"], startstate, actionperception["state"], actionperception["reward"])
            startstate = actionperception["state"]
        self.updatevalue()        

        
    def updatevalue(self):
        maxvaluediff = 0
        for i in range(10): ###### run for n iterations
            for node in self.statespace["nodes"]: ######## update value of all nodes
                fromnodeid = node["id"]
                fromnodevalue = node["value"]
                tonodes = [(edge["to"],edge["reward"]) for edge in self.statespace["edges"] if edge["from"] == fromnodeid and edge["reward"] != float('-Inf')]
                
                tonodevalues = [ self.statespace["nodes"][tonode[0]]["value"] for tonode in tonodes]
                ####### update value of from node for each tonode independently
                for tonode in zip(tonodes,tonodevalues):
                    fromnodevalue += alpha*(tonode[0][1] + gamma*tonode[1] - fromnodevalue)
                    maxvaluediff = max(maxvaluediff, (tonode[0][1] + gamma*tonode[1] - fromnodevalue))
            if  maxvaluediff < TDTHRESHOLD:
                break            
            
        ########## update UCB
        for id,node in self.statespace["nodes"].items():
            self.statespace["nodes"][id]["ucb"] = self.statespace["nodes"][id]["value"] + ucb_c*math.sqrt(math.log(self.totaltrials)/self.statespace["nodes"][id]["trial"])
        
        self.defaultucb = self.DEFAULTVALUE + ucb_c*math.sqrt(math.log(self.totaltrials))
        
        
        
    def getplandetails(self,currentstate):
        actionpath = []
        fromnodeid,fromnode = [(id,node) for id,node in self.statespace["nodes"].items() if node["state"] == currentstate][0]
        tonode = ""
        avoidactions = []
        while True:
            tonodes = [[edge["to"],edges["action"]] for edge in self.statespace["edges"] if edge["from"] == fromnodeid]        
            tonodes_ucb = [[self.statespace["nodes"][tonodeid]["ucb"],tonodeid] for tonodeid in tonodes if self.statespace["nodes"][tonodeid]["state"] != "invalid"]
            if not tonodes:
               break
            tonode = max(tonodes_ucb, key=lambda x: x[0])
            #if tonode[1] == fromnodeid:
            #   break
            if tonode[0] < self.defaultucb:
               avoidactions = [i[1] for i in tonodes]
               break
            actionpath.append(self.statespace["edges"][fromnodeid+"-"+tonode[1]]["action"])
            currentstate = self.statespace["nodes"][tonode[1]]["state"]
            fromnodeid = tonode[1]
        prompt = ""
        if actionpath:
            prompt = "You need to take the following actions in sequence \n"+ "\n".join(actionpath)+"\n\n You you arrive at the following state after taking the above actions\n"+currentstate+"\n\n"
        else:
            prompt = "You are at the state: \n"+currentstate +"\n\n" 
        if avoidactions:
            prompt += "find rest of the action plan. You should avoid the following immediate actions from the current state. \n" + "\n".join(avoidactions)
        else:
            prompt += "find rest of the action plan."
        return prompt,actionpath,avoidactions