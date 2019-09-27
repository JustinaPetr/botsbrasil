# Exercise

This file contains the walkthrough of the exercise to be completed using this repository.

## 0. Conversation design 

Conversation design is a very important step in chatbot development where you have to ask yourself who your users are, what they will use your assistant for and what conversations they are likely to have with your assistant. Crafting those conversations will help you start generating the training data. In this exercise we will build a travel assistant capable of suggesting what documents (VISA, etc) the user should take care of before traveling to the selected destination. To get started, write down some of those conversations:

U: Hello  
B: Hi! I am your travel assistant. What would you like to know?  
U: What are the document requirements for travelling to the US?  
B: What is your nationality?  
U: Brazilian  
B: You will need a VISA. You can order it here.  
U: Thanks  
B: Happy to help! Safe travels.

And one more:

U: Hi. What documents do I need when travelling to Sao Paulo with the american passport? 
B: Hello. You don't need any additional documents.  
U: Perfect. Thanks.  
B: Happy to help! Safe travels.

**TODO**: add more conversations like this with different dialogue turns.

## 1. Natural language understanding with Rasa 1.0

To train a NLU model with Rasa, you will need two things - training data and model configuration. Reuse the conversations generated in the previous step to define the intents and some training examples for the NLU.

---

#### 1.1 Create NLU training data
Looking at the conversations above, we can see that our assistant will have to understand 4 types of inputs to handle them: *greetings, questions about the document requirements, inputs where the user provides their nationality, thanks you's*.

NLU training examples should be defined inside the `./data/nlu.md` file. NLU training data follows the format:

```
## intent:intent_name
 - example1
 - example2
 ...
```
Create the intents and write down some training examples for each intent from the conversations created above. For example:  

```
## intent:greet
- hey
- hello
- hi
- good morning
- good evening
- hey there

## intent:thanks
- Thanks
- thank you
- thank you so much
- many thanks


## intent:documents
- What are the document requirements for travelling to the US?
- What documents do I need when travelling to Sao Paulo with the american passport?
- I am going to Europe next month. Do I need a VISA?
- What documents do I need when traveling to Australia?
- Do I need to get a VISA to travel to Germany?


## intent:inform
- American
- I am american
- French
- My nationality is Norvegian
- Brasilian
 ``` 
---

**TODO**: Add 5 more examples to each intent.

#### 1.2 Label the entities
Entities are the details that the assistant might need at a specific context. For example, knowing the destination and a nationality will come handy when the assistant will have to lookup the travel documentation requirements. Therefore these details should be extracted as entities. The format of entities looks as follows: [entity](entity_label). Label the entities in the created examples:
 ``` 
## intent:greet
- hey
- hello
- hi
- good morning
- good evening
- hey there

## intent:thanks
- Thanks
- thank you
- thank you so much
- many thanks


## intent:documents
- What are the document requirements for travelling to the [US](destination)?
- What documents do I need when travelling to [Sao Paulo](destination) with the [american](nationality) passport?
- I am going to [Europe](destination) next month. Do I need a VISA?
- What documents do I need when traveling to [Australia](destination)?
- Do I need to get a VISA to travel to [Germany](destination)?


## intent:inform
- [American](nationality)
- I am [american](nationality)
- [French](nationality)
- My nationality is [Norvegian](nationality)
- [Brasilian](nationality)
  ``` 

#### 1.3 Add synonyms
If you define entities as having the same value they will be treated as synonyms. This comes handy when you want to use some entity values to emulate the API or a database. Synonyms are defined as follows: [entity](entity_label:synonym_value). Add some synonyms to the training data we created.

``` 
## intent:documents
- What are the document requirements for travelling to the [US](destination:USA)?
- What documents do I need when travelling to [Sao Paulo](destination:BR) with the [american](nationality) passport?
- I am going to [Europe](destination) next month. Do I need a VISA?
- What documents do I need when traveling to [Australia](destination:AUS)?
- Do I need to get a VISA to travel to [Germany](destination:GE)?

    ``` 

#### 1.4 Create NLU model configuration pipeline

`config.yml` file contains a sample configuration of the model pipeline. We will modify it for a custom pipeline:

``` 
  language: en
pipeline:
- name: "WhitespaceTokenizer"
- name: "CRFEntityExtractor"
- name: "EntitySynonymMapper"
- name: "CountVectorsFeaturizer"
- name: "EmbeddingIntentClassifier"

    ``` 

---

#### 1.4 Train the NLU model
Once you update the training data and model configuration, train the NLU model by running:

`rasa train nlu`

Once mode is trained, it will be saved in a `./models` directory of you project. Test the model by running:

`rasa shell nlu`

---


## 2. Dialogue management with Rasa 1.0

Dialogue management system is responsible of predicting how an assistant should respond based on the state of the conversation as well as context. To train a dialogue management model you will need some training stories, domain file and model configuration. You can also create custom actions which, when predicted, can call an API, retrieve data from the database or perform some other integrations.

**TODO**: update the conversations with a question for how long the user will stay at specific location.
---

#### 2.1 Create training stories
The file `./data/stories.md` already contains some training stories. Create the training stories using the previously created conversations:


```
## user1
* greet
  - utter_greet
* documents{"destination":"US"}
  - utter_ask_nationality
* inform{"nationality":"American"}
  - action_check_documents
* thanks
  - utter_you_are_welcome

## user2
* greet
  - utter_greet
* documents{"destination":"US", "nationality":"Brazilian"}
  - action_check_documents
* thanks
  - utter_you_are_welcome

## user 3
* documents{"destination":"US", "nationality":"Brazilian"}
  - action_check_documents
* thanks
  - utter_you_are_welcome

    ```
    **TODO**: Add a new training story to the training data.
---

#### 2.2 Create the domain
A file called `domain.yml` must contain the domain configuration of your assistant. It contains all the information an assistant needs to know to operate. Define the domain to include all intents, entities, slots, actions and templates.

```
intents:
  - greet
  - documents
  - inform
  - thanks


actions:
- utter_greet
- utter_you_are_welcome
- utter_ask_nationality
- action_check_documents

entities:
  - destination
  - nationality

slots:
  destination:
    type: "unfeaturized"
  nationality:
    type: "unfeaturized"

templates:
  utter_greet:
  - text: "Hey! I am your travel assistant. How can I help?"

  utter_ask_nationality:
  - text: "What is your nationality"

  utter_you_are_welcome:
  - text: "Happy to help. Safe travels."
    ```

**TODO**: Add more possible responses for each template.

---
#### 2.3 Implement a custom action
The responses of your assistant can go beyond simple text templates. The responses where an assistant actually takes an action (makes an API call, extracts some data from the database, etc) are called custom actions and have to implemented as a custom action class inside the file called `actions.py`. Finish implementing the custom action so that the assistant responds with a message depending on what the user asked for:

```

from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import requests


class ActionSubscribe(Action):

     def name(self) -> Text:
         return "action_check_documents"

     def run(self, dispatcher: CollectingDispatcher,
             tracker: Tracker,
             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
         country = tracker.get_slot('destination')

         dispatcher.utter_message("You need a visa")

         return []
```
**TODO:** update the custom action to handle more logic.

---


#### 2.4 Create model configuration
A file `config.yml` also contains the configuration of the dialogue model. Let's define a simple training policy first:


```
policies:
  - name: MemoizationPolicy
  - name: KerasPolicy
    epochs: 200
```

---

#### 2.5 Train the dialogue model
Train the dialogue model by running:

`rasa train`

Start the server for Rasa custom actions:

`rasa run actions`

Once the server is up and running you can talk to your bot by running:

`rasa shell`

Rasa also comes with handy functions which allow to visualize training stories. To do that, run:

`rasa visualize`


---

#### 2.6 Add a MappingPolicy

MappinPolicy allows you to add some cusom logic to your conversations. For example, if you know that specific intent should always trigger a specific action regardless of what happened before throughout the conversations, you can implement such behaviour you can use the MappingPolicy.

First, edit the domain specifying which intent should trigger which action:

```
intents:
  - greet
  - documents
  - inform
  - thanks
      triggers: action_you_are_welcome
```

The action which is triggered by custom action will then have to be implemented as a cusom actions:

```
class ActionWelcome(Action):
"""Revertible mapped action for utter_is_bot"""

def name(self):
    return "action_you_are_welcome"

def run(self, dispatcher, tracker, domain):
    dispatcher.utter_template("utter_you_are_welcome", tracker)
    return [UserUtteranceReverted()]
```

#### 2.7 Add a fallback policy
It's likely that at some point your assistant will make a mistake - the NLU or dialogue model might get uncertain about some predictions once it gets some unexpected user inputs. Once way to go around this problem is using a FallbackPolicy which invokes a fallback action if the intent recognition has a confidence below `nlu_threshold`. Update your model configuration in `config.yml` to include the fallback policy:

```
policies:
  - name: "FallbackPolicy"
    nlu_threshold: 0.3
    core_threshold: 0.3
    fallback_action_name: 'action_default_fallback'
```

**TODO**: Test the assistant on more inputs and update the stories, custom actions depending on the performance.

## 3. Close the feedback loop with Rasa X
After you build and test a simple version of your assistant, it's important to give it to the
actual users, collect the feedback from them and improve your assistant continuously. 

---

#### 3.1 Use Rasa X to improve your assistant
The best way to improve your assistant using real conversational data is using Rasa X. In this repository,
you will find files called `rasa.db` and `tracked.db` which already contain a few real-world conversations
between our bot and the users. We will reuse them to improve the assistant using the Rasa X. Start your 
assistant with Rasa X using:

`rasa x`

In the `Conversations` tab of the Rasa X you will find existing conversations between users and your assistant.
Correct them using interactive learning and add them to the training data.

Hit the button `Train` to retrain the model and select a newly trained one inside the `Models` tab of the Rasa X UI.

Test the improved assistant by talking to it inside the `Talk to your bot` tab of the Rasa X UI or share it with 
your friends by generating and sharing a link to your bot inside the `Conversations` tab. All conversations that new
generate with your assistant will end up in `Conversations` tab for you to look into and reuse to improve your assistant.

---

#### 3.2 Expose your assistant to the outside world 
To share your assistant with friends, you will have to expose your locally running assistant to the outside world.
To do that, you can deploy your assistant on a server, or use ngrok (recommended for non-production environments only).
To do that, start ngrok by running:

`ngrok http 5002`

Copy the ngrok URL and run:

`export RASA_X_HOSTNAME=https://xxxxxx.ngrok.io; rasa x`

Once the Rasa X launches, you should be able to share your assistant with your friends around the world!

**TODO:** Share the assistant with the friend next to you and let them talk to your bot.


