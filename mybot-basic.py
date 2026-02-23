#!/usr/bin/env python3
import time
if not hasattr(time, 'clock'):
    time.clock = time.perf_counter

import aiml
import wikipedia
import fivbvis
# Create a Kernel object. 
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="mybot-basic.xml")

# Welcome user
print("Welcome to this chat bot. Please feel free to ask questions from me!")

# Main loop
while True:
    #get user input
    try:
        userInput = input("> ")
    except (KeyboardInterrupt, EOFError):
        print("Bye!")
        break
    #pre-process user input and determine response agent (if needed)
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
        elif cmd == 1:
            try:
                wSummary = wikipedia.summary(params[1], sentences=3,auto_suggest=True)
                print(wSummary)
            except:
                print("Sorry, I do not know that. Be more specific!")
        elif cmd ==2:
            match_id = params[1]
            v = fivbvis.Volleyball()
            # Get the data, requesting specific fields to make the response clear
            match_data = v.getVolleyMatch(no=match_id, fields="City TeamNameA TeamNameB DateLocal ScoreA ScoreB")
            # Parse match_data (you'll need to handle the XML/JSON) and print a nice response
            print(
                f"Match {match_id} was played in {match_data.City} on {match_data.DateLocal}. The score was {match_data.TeamNameA} {match_data.ScoreA} - {match_data.ScoreB} {match_data.TeamNameB}.")
        elif cmd == 99:
            print("I did not get that, please try again.")
    else:
        print(answer)