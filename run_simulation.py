import sys
import os
import random
import time
import zmq

# --- Create a directory for the sensor output ---
# This ensures the folder exists before the script tries to save images.
os.makedirs('_output', exist_ok=True)

# --- Add CARLA .egg to the Python path ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    egg_path = os.path.join(script_dir, 'carla-0.9.14-py3.7-linux-x86_64.egg')
    
    if not os.path.exists(egg_path):
        raise FileNotFoundError(f"Could not find carla .egg file at {egg_path}")

    sys.path.append(egg_path)
    
    # Import carla now that the path is set
    import carla

except (IndexError, FileNotFoundError, ImportError) as e:
    print(f"Error importing CARLA: {e}")
    sys.exit()

from carla_actor_factory import CarlaActorFactory

def camera_callback(image, socket):
    """
    This function is called every time the camera sensor gets a new image.
    It sends the image data to the C++ server via ZMQ.
    """

    try:
        metadata = dict(
            height=image.height,
            width=image.width,
            channels=4,
            frame=image.frame
        )

        socket.send_json(metadata, flags=zmq.SNDMORE)

        socket.send(image.raw_data)

        reply_message = socket.recv_string()

        print(f"Received reply form C++: [{reply_message}] for frame {metadata['frame']}")

    except Exception as e:
        print(f"Error in camers callback: {e}")

def main():
    actors_list = []
    socket = None

    try:
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect("tcp://host.docker.internal:5555")

        client = carla.Client('34.148.135.236', 2000)
        client.set_timeout(10.0)
        world = client.get_world()

        factory = CarlaActorFactory(world, world.get_blueprint_library())
        spawn_point = random.choice(world.get_map().get_spawn_points())

        vehicle = factory.create_vehicle('vehicle.tesla.model3', spawn_point)
        actors_list.append(vehicle)
        vehicle.set_autopilot(True)

        camera = factory.create_camera(vehicle)
        actors_list.append(camera)

        camera.listen(lambda image: camera_callback(image, socket))

        print("\n Simulation running. Streaming camera data to C++ server.")

        while True:
            time.sleep(1)

    except Exception as e:
        print(f"\nAn error occured in main: {e}")

    finally:
        if socket:
            socket.close()
        if actors_list:
            print("Destroying actors...")
            client.apply_batch([carla.command.DestroyActor(x) for x in actors_list])
            print("Done")

if __name__ == '__main__':
    main()
