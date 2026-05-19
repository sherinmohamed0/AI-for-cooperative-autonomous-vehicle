import os
import traci
import random

# SUMO binary and configuration
sumoBinary = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe"
sumoCmd = [sumoBinary, "-c", "scenario1.sumocfg"]

traci.start(sumoCmd)

collision_count = 0
step = 0

# List of all 40 vehicle IDs
all_veh_ids = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
    "20", "21", "22", "23", "24", "25", "26", "27", "28", "29",
    "30", "31", "32", "33", "34", "35", "36", "37", "38", "39"
]

used_followers = set()

while traci.simulation.getMinExpectedNumber() > 0:

    traci.simulationStep()
    step += 1

    # Wait a few steps so vehicles spread out naturally
    if step < 20:
        continue

    # Only force collision if we haven't reached 5 yet
    if collision_count < 15:

        # Get list of available followers
        available_followers = [
            v for v in all_veh_ids 
            if v in traci.vehicle.getIDList() and v not in used_followers
        ]

        if available_followers:
            follower = random.choice(available_followers)
            leader_info = traci.vehicle.getLeader(follower)

            if leader_info is not None:
                leader = leader_info[0]
                try:
                    
                    # # 1. Set the trap (as you were doing)
                    # traci.vehicle.setSpeed(leader, 0)
                    # traci.vehicle.setSpeedMode(follower, 0)
                    # # traci.vehicle.setSpeed(follower, 35) # Leave this commented out for a natural crash

                    # # 2. Check for actual physical collisions happening right now in the simulation
                    # actual_collisions = traci.simulation.getCollidingVehiclesIDList()

                    # # 3. Print the reality
                    # if actual_collisions:
                    #     print(f"SUMO Engine detected actual physical collisions this frame: {actual_collisions}")
                    
                    # Stop leader suddenly
                    traci.vehicle.setSpeed(leader, 0)

                    # Disable follower safety
                    traci.vehicle.setSpeedMode(follower, 0)

                    # Force high speed follower
                    #traci.vehicle.setSpeed(follower, 35)

                    used_followers.add(follower)
                    collision_count += 1

                    print(f"Collision forced between {follower} and {leader}")

                except:
                    continue

# Close simulation only when all vehicles have finished
traci.close()