# run_simulation.py

import sys
import os
import random

# Add CARLA .egg to the Python path to enable the 'carla' import.
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    egg_path = os.path.join(script_dir, 'carla-0.9.14-py3.7-linux-x86_64.egg')
    
    if not os.path.exists(egg_path):
        raise FileNotFoundError(f"Could not find carla .egg file at {egg_path}")

    sys.path.append(egg_path)

except IndexError:
    pass

import carla

def main():
    client = None
    vehicle = None # Define vehicle here to ensure it's in scope for the finally block
    try:
        # NOTE: Ensure your firewall allows traffic to this IP on ports 2000-2001.
        client = carla.Client('34.148.135.236', 2000)
        client.set_timeout(10.0) 

        print("Successfully connected to CARLA server.")

        world = client.get_world()
        print(f"Current map: {world.get_map().name}")

        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            print("No spawn points found in the map!")
            return
            
        spawn_point = random.choice(spawn_points)

        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        print(f"Spawned {vehicle.type_id} at location {spawn_point.location}")
        
        # Pause script to observe the spawned vehicle in the simulator.
        input("Press Enter to destroy the actor and exit...")

    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        # Ensure the spawned vehicle is destroyed on exit.
        if vehicle is not None:
            print("Destroying actors...")
            vehicle.destroy()
            print("Done.")

if __name__ == '__main__':
    main()
