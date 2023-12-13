# Neoplanner

## Abstract
This repo contains the implementation of a sequential planning agent called, “neoplanner”. This planner is suitable for text based environments with large state space and action space. It synergizes both state space search with queries to foundational LLM to get the best action plan. The reward signals are quantitatively used to drive the search. A balance of exploration and exploitation is maintained by maximizing upper confidence bounds of values of states. In places where random exploration is needed, the LLM is queried to generate an action plan. Learnings from each trial are stored as entity relationships in text format. Those are used in future queries to the LLM for continual improvement. Experiments in the Scienceworld environment reveals a 124% improvement from the current best method in terms of average reward gained across multiple tasks.


## Get Started
First, clone the repo and navigate into the neoplanner directory and install the requirements

```
git clone https://github.com/swarna-kpaul/neoplanner
cd neoplanner
python3 -m pip install -r requirements.txt
```

Then you need to modify the ***config/keys.py*** file to update the ***OPENAIAPIKEY***. You can get your API key by registering yourself into openAI portal. First time users can get a free $5 credit to experiment with. You can get your API key from this [url](https://platform.openai.com/api-keys)

Thereafter the package can be imported

```
from solver import neoplanner
```

Initialize the solver object

```
# task is the identifier of tasks as specified in 
# stmloadfile is the name of the file (with full path) that contains saved state. The state will be loaded initially. default value is None
# stmstoragefile is the name of the file (with full path) whare intermediate states can be saved. default value is None
# beliefstorefile is the name of the file (with full path) whare intermediate learnings can be saved. default value is None
# beliefloadfile is the name of the file (with full path) that contains intermediate learnings. The learnings will be loaded initially. default value is None
solverobj = neoplanner(task = "2-1", stmloadfile =  None, stmstoragefile = None, beliefstorefile = None, beliefloadfile= None)
```

Run the solver. 

```
env = solverobj.train()
######## get actionplan from statespace graph
additionalinstructions,actionplan,_,_,_ = env.getinstructions()

```

The training will continue running until goal is reached. You may interrupt the training process in between. In that case make sure you provide the stmstoragefile and beliefstorefile, so that intermediate states and beliefs are saved. 

You can load the stmstoragefile and query the env object to get the action plan from state space graph.

```
from solver import scienv
env = scienv("2-1")
stmstoragefile = <file name with full path>
with open(stmstoragefile, 'rb') as f:
  rootnodeid,invalidnodeid, DEFAULTVALUE,statespace,totaltrials, actiontrace,environment = pickle.load(f)
env.model.rootnodeid = rootnodeid
env.model.invalidnodeid = invalidnodeid
env.model.DEFAULTVALUE = DEFAULTVALUE
env.model.statespace = statespace
env.model.totaltrials = totaltrials
env.environment = environment
env.reset()

additionalinstructions,actionplan,_,_,_ = env.getinstructions()
```