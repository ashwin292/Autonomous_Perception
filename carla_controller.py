import carla
import random
import queue
import numpy as np
import cv2
import zmq

def main():
    # --- ZMQ Setup ---
    context = zmq.Context()
    socket = context.socket(zmq.REQ) # REQ (Request) socket
    socket.connect("tcp://localhost:5555")

    # --- CARLA Setup ---
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # Get a vehicle blueprint and spawn it
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = random.choice(blueprint_library.filter('vehicle.tesla.model3'))
    transform = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(vehicle_bp, transform)
    
    # Store spawned actors for cleanup
    actor_list = [vehicle]
    print(f'Spawned vehicle: {vehicle.type_id}')

    # Set vehicle to autopilot to drive around
    vehicle.set_autopilot(True)

    # --- CARLA Sensor Setup ---
    # Create an RGB camera sensor
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '1280')
    camera_bp.set_attribute('image_size_y', '720')
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    actor_list.append(camera)

    # Create a queue to store images from the camera
    image_queue = queue.Queue()
    camera.listen(image_queue.put)

    try:
        while True:
            # Get image from the queue
            image = image_queue.get()
            
            # Convert CARLA image to a NumPy array
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]  # Convert BGRA to BGR
            
            # Encode image to JPEG to reduce network size
            _, buffer = cv2.imencode('.jpg', array)

            # --- Communication Loop ---
            # 1. Send image to C++ server
            socket.send(buffer)

            # 2. Receive command from C++ server
            command = socket.recv_string()
            print(f"Received command: {command}")
            
            # 3. Apply control based on command
            if command == "BRAKE":
                vehicle.set_autopilot(False) # Disable autopilot to take control
                vehicle.apply_control(carla.VehicleControl(brake=1.0))
            else: # "CONTINUE"
                vehicle.set_autopilot(True) # Re-enable autopilot

    finally:
        print('Cleaning up actors...')
        for actor in actor_list:
            actor.destroy()
        print('Done.')

if __name__ == '__main__':
    main()
