# This is the starting point for developing task-b of the chatbot project.
# Note: Do not start the whole project from this source code! 
# rather, once you have your task-a ready, manage to incorporate it with this source code.
# Notice to the incomplete tasks below - indicated with ">>"

#######################################################
#  Initialise NLTK Inference
from nltk.sem import Expression
from nltk.inference import ResolutionProver
read_expr = Expression.fromstring

#  Initialise Knowledgebase. 
import pandas
kb=[]
data = pandas.read_csv('logical-kb.csv', header=None)
[kb.append(read_expr(row)) for row in data[0]]
# >>> ADD SOME CODES here for checking KB integrity (no contradiction), 
# otherwise show an error message and terminate

#  Initialise AIML agent
import aiml
# Create a Kernel object. No string encoding (all I/O is unicode)
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="mybot-logic.xml")

# Welcome user
print("Welcome to this chat bot. Please feel free to ask questions from me!")

# Main loop
while True:
    #get user input
    try:
        userInput = input("> ")
    except (KeyboardInterrupt, EOFError) as e:
        print("Bye!")
        break
    responseAgent = 'aiml'
    #activate selected response agent
    if responseAgent == 'aiml':
        answer = kern.respond(userInput)
    #post-process the answer for commands
    if answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])
        if cmd == 0:
            print(params[1])
            break
        # >> YOU should already had some other "if" blocks here from task-a - you may incorporate them here.
        
        # Here are the processing of the new logical component:
        elif cmd == 31: # if input pattern is "I know that * is *"
            object,subject=params[1].split(' is ')
            expr=read_expr(subject + '(' + object + ')')
            # >>> ADD SOME CODES HERE to make sure expr does not contradict 
            # with the KB before appending, otherwise show an error message.
            kb.append(expr) 
            print('OK, I will remember that',object,'is', subject)
        elif cmd == 32: # if the input pattern is "check that * is *"
            object,subject=params[1].split(' is ')
            expr=read_expr(subject + '(' + object + ')')
            answer=ResolutionProver().prove(expr, kb, verbose=True)
            if answer:
               print('Correct.')
            else:
               print('It may not be true.') 
               # >> This is not an ideal answer.
               # >> ADD SOME CODES HERE to find if expr is false, then give a
               # definite response: either "Incorrect" or "Sorry I don't know." 
             
        elif cmd == 99:
            print("I did not get that, please try again.")
    else:
        print(answer)