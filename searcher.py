from config.keys import *
from config.prompts import *
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
import traceback
from environment.problemenvs import *

import ast
import pickle

llm_model = ChatOpenAI(temperature=0.7, request_timeout=50, model="gpt-3.5-turbo-1106",openai_api_key=OPENAIAPIKEY)
llm_inst_model = OpenAI(temperature=0.7, request_timeout=50, model="gpt-3.5-turbo-instruct",openai_api_key=OPENAIAPIKEY)
llm_defn_model = OpenAI(temperature=0, request_timeout=50, model="gpt-3.5-turbo-instruct",openai_api_key=OPENAIAPIKEY)
llm_gpt4 = ChatOpenAI(temperature=0.7, request_timeout=50, model="gpt-4-0613",openai_api_key=OPENAIAPIKEY)
llm_gpt4_turbo = ChatOpenAI(temperature=0.7, request_timeout=50, model="gpt-4-1106-preview",openai_api_key=OPENAIAPIKEY)
llm_gpt4_turbo_hightemp = ChatOpenAI(temperature=1, request_timeout=50, model="gpt-4-1106-preview",openai_api_key=OPENAIAPIKEY)

class neoplanner():
    def __init__ (self, task="2-1", stmloadfile =None, stmstoragefile =None, beliefstorefile = None, beliefloadfile = None, counter = 0):
    
        self.stmstoragefile = stmstoragefile
        self.beliefstorefile = beliefstorefile
        self.env = scienv(task)
        self.explore = True
        if stmloadfile != None:
            with open(stmloadfile, 'rb') as f:
               rootnodeid,invalidnodeid, DEFAULTVALUE,statespace,totaltrials, actiontrace,environment = pickle.load(f)
            self.env.model.rootnodeid = rootnodeid
            self.env.model.invalidnodeid = invalidnodeid
            self.env.model.DEFAULTVALUE = DEFAULTVALUE
            self.env.model.statespace = statespace
            self.env.model.totaltrials = totaltrials

            self.env.environment = environment
            self.env.actiontrace = actiontrace
            ############# execute action trace
            self.env.traceact()
        if beliefloadfile != None:
            with open(beliefloadfile, 'rb') as f:
               beliefaxioms,totalexplore = pickle.load(f)
            self.env.environment["belief axioms"] = beliefaxioms
            self.env.totalexplore = totalexplore
        
            print (self.env.getstate())
            input("Press a key to continue ....")
        self.counter = counter    
        self.SEARCHERPROMPT = PromptTemplate(input_variables=SEARCHERPROMPTINPUTVARIABLES, template=searchertemplate)
        self.ACTPLANPROMPT = PromptTemplate(input_variables=ACTPLANPROMPTINPUTVARIABLES, template=actionplantemplate)
        self.COMBINERPROMPT = PromptTemplate(input_variables=COMBINERVARIABLES, template=combinertemplate)

    
    def searcher (self,EnvTrace,feedback,counter):
        currentenvironment = self.env.environment
        currentbelief = "  objective:"+ currentenvironment['objective']+"\n  belief axioms:"+ str(currentenvironment["belief axioms"])
        EnvTrace_text = "\n\n".join([str(i) for i in EnvTrace])
        messages = self.SEARCHERPROMPT.format(
                        beliefenvironment = str(currentbelief),
                        EnvTrace = EnvTrace_text,
                        feedback = feedback)
            #print(messages)
        print("SEARCHERPROMPT:",messages)
        while True:
            try:
                output = llm_gpt4_turbo.predict(messages)
                print("SEARCHERPROMPT output:",output)
                beliefaxioms = ast.literal_eval(output)
                break
            except Exception as e:
                input("press any key....")
                continue                
        if counter == 0 or len(beliefaxioms) > 50:
            messages = self.COMBINERPROMPT.format(beliefaxioms = beliefaxioms)
            print("COMBINERPROMPT:",messages)
            while True:
                try:
                    output = llm_gpt4_turbo.predict(messages)
                    print("COMBINERPROMPT output:",output)
                    beliefaxioms = ast.literal_eval(output)
                    break
                except Exception as e:
                    input("press any key....")
                    continue 
        self.env.environment["belief axioms"] = beliefaxioms
        
        return output
    
    def actplan(self, additionalinstructions = "", explore =True, envtrace = [],ucbfactor = 1):
        currentenvironment = pickle.loads(pickle.dumps(self.env.environment,-1))
        beliefaxioms = "\n".join(currentenvironment["belief axioms"])
        
        explore_probability = 0.6/ math.log(self.env.totalexplore)
        probabilities = [explore_probability, 1-explore_probability]
        population = ["explore","objective"]
        item = random.choices(population, probabilities)[0]
        
        if  item == "explore":
            currentenvironment["objective"] = exploreobjective
            self.env.totalexplore += 1
        if envtrace:
           envtrace = "\n".join(["action: "+i["action"]+"; observation: "+i["observation"] for i in envtrace])
        else:
           envtrace = ""
        if not additionalinstructions :
            currentenvironmenttext = "    objective: \n" + currentenvironment["objective"] +"\n\n"+" prior axioms: \n"+currentenvironment["prior axioms"]+"\n\n"+ "     belief axioms:\n"+beliefaxioms+"\n\n"+"    current state:\n"+ currentenvironment["current state"]
        else:
            currentenvironmenttext = "    objective: \n" + currentenvironment["objective"] +"\n\n"+" prior axioms: \n"+currentenvironment["prior axioms"]+"\n\n"+ "     belief axioms:\n"+beliefaxioms+"\n\n"

        messages = self.ACTPLANPROMPT.format(beliefenvironment = currentenvironmenttext, \
                        actionplanexamples = self.env.examples,\
                        envtrace = envtrace, \
                        instructions = additionalinstructions)
        
        print("ACTPLANPROMPT:",messages)
        while True:
            output = llm_gpt4_turbo.predict(messages)
            print("ACTPLANPROMPT output:",output)
            try:
                output = ast.literal_eval(output)
            except Exception  as e:
                    #errorfeedback = "Here is the last actionplan generated. "+ output+ "\n But this action plan has the following error. Modify the plan to remove the error.\n"+str(e)
                print(traceback.format_exc())
                input("Press any key to continue...")
                continue    
            
            break
        
        return output
        
    
    def train (self, lifetime = float("Inf")):   
        counter = self.counter
        while True:
            print("GOAL REACHED",self.env.goalreached)
            if lifetime <= 0: # or self.env.goalreached:
                break
            EnvTrace = []
            for i in range(4):    
            ####### get additional instructions
                instructions,preactionplan,_,explore,ucbfactor = self.env.getinstructions()
                
                
            ###### Run actor
                print("Running actionplan....")
                actionplan = []
            
                actionplan = self.actplan(instructions,explore,EnvTrace,ucbfactor)
                k = input("Press any button to continue ...")
                actionplan = preactionplan + actionplan
                
                print ("FULL ACTION PLAN: ", actionplan)
            ########### Take actions
                try:
                    for action in actionplan:
                        self.env.act(action)
                except world_exception as e:
                    pass
                
                
                self.env.updatemodel()
                feedback = self.env.getfeedback()
                EnvTrace += [{"action": trace["action"], "observation": trace["observation"]} for trace in self.env.trace]
                self.env.trace = []
                
            self.searcher(EnvTrace, feedback, counter)
            input("Press any key to continue...")
            self.env.reset()
            counter += 1
            with open(self.stmstoragefile, 'wb') as f:
                pickle.dump((self.env.model.rootnodeid,self.env.model.invalidnodeid, self.env.model.DEFAULTVALUE,self.env.model.statespace,self.env.model.totaltrials, self.env.actiontrace,self.env.environment),f)

            with open(self.beliefstorefile, 'wb') as f:
                pickle.dump((self.env.environment["belief axioms"],self.env.totalexplore),f)
   
            
        
    