#!/usr/bin/env python3
import time
if not hasattr(time, 'clock'):
    time.clock = time.perf_counter
import xml.etree.ElementTree as ET
import aiml
import wikipedia
import fivbvis
# Create a Kernel object. 
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="mybot-basic.xml")

# Welcome user
print("Welcome to this chat bot. Please feel free to ask questions from me!\nI am specifically designed for volleyball questions.\nIf you want to know more about a specific match type in 'Volleyball Match' followed by an ID (e.g. 11500).")

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
        elif cmd == 2:
                match_id = params[1]
                try:
                    v = fivbvis.Volleyball()
                    # Get the data, requesting specific fields
                    match_data = v.getVolleyMatch(no=match_id,
                                                  fields="City TeamNameA TeamNameB CountryName DateLocal MatchPointsA MatchPointsB")

                    root = ET.fromstring(match_data)

                    # Extract data from XML attributes
                    city = root.get('City', 'Unknown')
                    team_a = root.get('TeamNameA', 'Unknown')
                    team_b = root.get('TeamNameB', 'Unknown')
                    country = root.get('CountryName', 'Unknown')
                    date = root.get('DateLocal', 'Unknown')
                    points_a = root.get('MatchPointsA', 'Unknown')
                    points_b = root.get('MatchPointsB', 'Unknown')

                    # Check if this is a valid match (has at least a match number)
                    match_no = root.get('No', '')

                    # If the match number doesn't match what we requested or is missing, it's invalid
                    if not match_no or match_no != match_id:
                        print(f"No match found with ID {match_id}")

                    # Check if all essential fields are 'Unknown' (likely invalid match)
                    elif city == 'Unknown' and team_a == 'Unknown' and team_b == 'Unknown':
                        print(f"No match found with ID {match_id}")

                    # We have at least some real data
                    else:
                        # Check if points exist (not Unknown and not empty)
                        if points_a not in ['Unknown', ''] and points_b not in ['Unknown', '']:
                            print(
                                f"Match {match_id} was played in {city}, {country} on {date} between {team_a} and {team_b} and finished {points_a} - {points_b}.")

                        # We have match info but no points
                        elif team_a not in ['Unknown', ''] and team_b not in ['Unknown', '']:
                            print(
                                f"Match {match_id} was played in {city}, {country} on {date} between {team_a} and {team_b}. No points are available for this match.")

                        # Very limited info
                        else:
                            print(
                                f"Match {match_id} was played in {city}, {country} on {date}. Limited information is available for this match.")

                except ET.ParseError as e:
                    print(f"Sorry, I couldn't parse the match data. Error: {e}")
                except Exception as e:
                    print(f"Sorry, I couldn't find that volleyball match. Error: {e}")
        elif cmd == 3:
            player_name = params[1]
        elif cmd == 99:
            print("I did not get that, please try again.")
    else:
        print(answer)