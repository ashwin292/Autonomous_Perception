import carla

class CarlaActorFactory:
    def __init__(self, world, blueprint_library):
        self.world = world
        self.bp_lib = blueprint_library

    def create_vehicle(self, blueprint_id, spawn_point):
        vehicle_bp = self.bp_lib.find(blueprint_id)
        vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        return vehicle
    
    def create_camera(self, parent_actor):
        camera_bp = self.bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1280')
        camera_bp.set_attribute('image_size_y', '720')
        camera_transform = carla.Transform(carla.Location(z=1.8))
        camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to = parent_actor)
        return camera
