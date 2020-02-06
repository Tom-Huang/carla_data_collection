#!/usr/bin/env python
import glob
import os
import sys
from PIL import Image as PImage
import numpy as np
import io
import datetime

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla
import random
import queue
import pygame
from BufferedImageSaver import BufferedImageSaver
from BufferedLidarSaver import BufferedLidarSaver

# TIME INTERVAL BETWEEN FRAMES
TIME_INTER = 1

# STOP AFTER HOW MANY MILLISECONDS
STOP_AFTER = 3600 * 10 * TIME_INTER


# define parameters
OTHER_VEH_NUM = 80
OTHER_PED_NUM = 0

# semantic segmentation sensor parameters
SENSOR_TICK = 0.0
SAVE_PATH = "./images/"
NAME_WITH_TIME = "_".join("_".join(str(datetime.datetime.now()).split(" ")).split(":"))[:16]
SENSOR_TYPE_1 = "CameraSemSeg"
SENSOR_TYPE_1_CARLA = "sensor.camera.semantic_segmentation"

# format of all image sensor data
WINDOW_HEIGHT = 375
WINDOW_WIDTH = 1242
BUFFER_SIZE = 100
FOV = 120

# lidar sensor parameters
SENSOR_TYPE_2 = 'Lidar'
SENSOR_TYPE_2_CARLA = "sensor.lidar.ray_cast"
CHA_NUM = 64
RANGE = 10000
PTS_PER_SEC = 90000
ROT_FREQ = 10
UPPER_FOV = 10
LOWER_FOV = -30
#FRM_PTS_NUM = int(PTS_PER_SEC / (SENSOR_TICK * CHA_NUM))

# if (PTS_PER_SEC % float(SENSOR_TICK * CHA_NUM)) != 0:
#     print("Points per second can not be divided by FPS * \
#         channel_number! Please change the parameter!")

# depth sensor parameters
SENSOR_TYPE_3 = 'CameraDepth'
SENSOR_TYPE_3_CARLA = 'sensor.camera.depth'
# DEPTH_IMG_SIZE_HEIGHT = 375
# DEPTH_IMG_SIZE_WIDTH = 1242
DEPTH_FOV = 120
DEPTH_SENSOR_TICK = 0.0

# normal sensor parameters
SENSOR_TYPE_4 = 'NormalCam'
SENSOR_TYPE_4_CARLA = 'sensor.camera.normal'
# NORMAL_IMG_SIZE_HEIGHT = 375
# NORMAL_IMG_SIZE_WIDTH = 1242
NORMAL_FOV = 120
NORMAL_SENSOR_TICK = 0.0

# albedo sensor parameters
SENSOR_TYPE_5 = 'AlbedoCam'
SENSOR_TYPE_5_CARLA = 'sensor.camera.albedo'
# ALBEDO_IMG_SIZE_HEIGHT = 375
# ALBEDO_IMG_SIZE_WIDTH = 1242
ALBEDO_FOV = 120
ALBEDO_SENSOR_TICK = 0.0

# albedo sensor parameters
SENSOR_TYPE_6 = 'RealCam'
SENSOR_TYPE_6_CARLA = 'sensor.camera.rgb'
# ALBEDO_IMG_SIZE_HEIGHT = 375
# ALBEDO_IMG_SIZE_WIDTH = 1242
RGB_FOV = 120
RGB_SENSOR_TICK = 0.0

# normal to camera frame sensor parameters
SENSOR_TYPE_7 = 'NormalCamCamFrame'
SENSOR_TYPE_7_CARLA = 'sensor.camera.normal_cam_frame'
# ALBEDO_IMG_SIZE_HEIGHT = 375
# ALBEDO_IMG_SIZE_WIDTH = 1242
RGB_FOV = 120
RGB_SENSOR_TICK = 0.0

# albedo sensor parameters
SENSOR_TYPE_8 = 'ReflectionCam'
SENSOR_TYPE_8_CARLA = 'sensor.camera.reflection'
# ALBEDO_IMG_SIZE_HEIGHT = 375
# ALBEDO_IMG_SIZE_WIDTH = 1242
RGB_FOV = 120
RGB_SENSOR_TICK = 0.0

buffer_channel = {}
buffer_channel["CameraSemSeg"] = 1
buffer_channel["CameraRGB"] = 3
buffer_channel["CameraDepth"] = 1

# image_saver_front = BufferedImageSaver(SAVE_PATH+NAME_WITH_TIME+"/"+"front/", BUFFER_SIZE,
#                                        WINDOW_HEIGHT, WINDOW_WIDTH, buffer_channel[SENSOR_TYPE_1], SENSOR_TYPE_1)
# image_saver_left = BufferedImageSaver(SAVE_PATH+NAME_WITH_TIME+"/"+"left/", BUFFER_SIZE,
#                                       WINDOW_HEIGHT, WINDOW_WIDTH, buffer_channel[SENSOR_TYPE_1], SENSOR_TYPE_1)
# image_saver_right = BufferedImageSaver(SAVE_PATH+NAME_WITH_TIME+"/"+"right/", BUFFER_SIZE,
#                                        WINDOW_HEIGHT, WINDOW_WIDTH, buffer_channel[SENSOR_TYPE_1], SENSOR_TYPE_1)
# image_saver_rear = BufferedImageSaver(SAVE_PATH+NAME_WITH_TIME+"/"+"rear/", BUFFER_SIZE,
#                                       WINDOW_HEIGHT, WINDOW_WIDTH, buffer_channel[SENSOR_TYPE_1], SENSOR_TYPE_1)
# lidar_saver = BufferedLidarSaver(SAVE_PATH+NAME_WITH_TIME+"/", BUFFER_SIZE, CHA_NUM, FRM_PTS_NUM, "Lidar")
image_saver_front = BufferedImageSaver(SAVE_PATH+NAME_WITH_TIME+"/"+"depth/", BUFFER_SIZE,
                                       WINDOW_HEIGHT, WINDOW_WIDTH, buffer_channel[SENSOR_TYPE_3], SENSOR_TYPE_3)


# create folders for storing data


if not os.path.exists(SAVE_PATH+NAME_WITH_TIME+"/"+"depth/"):
    os.makedirs(SAVE_PATH+NAME_WITH_TIME+"/"+"depth/")

if not os.path.exists(SAVE_PATH+NAME_WITH_TIME+"/"+"normal_to_world/"):
    os.makedirs(SAVE_PATH+NAME_WITH_TIME+"/"+"normal_to_world/")

if not os.path.exists(SAVE_PATH+NAME_WITH_TIME+"/"+"albedo/"):
    os.makedirs(SAVE_PATH+NAME_WITH_TIME+"/"+"albedo/")

if not os.path.exists(SAVE_PATH+NAME_WITH_TIME+"/"+"semantic/"):
    os.makedirs(SAVE_PATH+NAME_WITH_TIME+"/"+"semantic/")

if not os.path.exists(SAVE_PATH+NAME_WITH_TIME+"/"+"rgb/"):
    os.makedirs(SAVE_PATH+NAME_WITH_TIME+"/"+"rgb/")

if not os.path.exists(SAVE_PATH+NAME_WITH_TIME+"/"+"normal_to_cam/"):
    os.makedirs(SAVE_PATH+NAME_WITH_TIME+"/"+"normal_to_cam/")

if not os.path.exists(SAVE_PATH+NAME_WITH_TIME+"/"+"reflection/"):
    os.makedirs(SAVE_PATH+NAME_WITH_TIME+"/"+"reflection/")

# if not os.path.exists(SAVE_PATH+NAME_WITH_TIME+"/"+"left/"):
#     os.makedirs(SAVE_PATH+NAME_WITH_TIME+"/"+"left/")

# if not os.path.exists(SAVE_PATH+NAME_WITH_TIME+"/"+"right/"):
#     os.makedirs(SAVE_PATH+NAME_WITH_TIME+"/"+"right/")

# if not os.path.exists(SAVE_PATH+NAME_WITH_TIME+"/"+"rear/"):
#     os.makedirs(SAVE_PATH+NAME_WITH_TIME+"/"+"rear/")

# if not os.path.exists(SAVE_PATH+NAME_WITH_TIME):
#     os.makedirs(SAVE_PATH+NAME_WITH_TIME+"/"+"Lidar/")


def save_images(image_saver, image):
    buffer = bytes(image.raw_data)
    print("image_saver index: ", image_saver.index)
    image_saver.add_image(buffer, SENSOR_TYPE_1)
    print("add image %d success." % image.frame_number)


def save_lidar(data):
    buffer = bytes(data.raw_data)

    print("lidar_saver index: ", lidar_saver.index)
    lidar_saver.add_image(buffer, SENSOR_TYPE_2)
    print("add lidar data % success." % data.frame_number)


def main():
    actor_list = []
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(2.0)

    try:
        world = client.load_world('Town04') #get_world()  ### new
        print('enabling synchronous mode.')
        settings = world.get_settings()
        settings.fixed_delta_seconds = 0.1 ### new
        settings.synchronous_mode = True
        world.apply_settings(settings)

        blueprints = world.get_blueprint_library()
        vehicle_blueprint = blueprints.filter("vehicle.*")
        pedestrain_blueprint = blueprints.filter("walker.*")

        sensor_blueprint = blueprints.find(SENSOR_TYPE_1_CARLA)
        sensor_blueprint.set_attribute('image_size_x', str(WINDOW_WIDTH))
        sensor_blueprint.set_attribute('image_size_y', str(WINDOW_HEIGHT))
        sensor_blueprint.set_attribute('fov', str(FOV))
        sensor_blueprint.set_attribute('sensor_tick', str(SENSOR_TICK))

        # lidar_blueprint = blueprints.find(SENSOR_TYPE_2_CARLA)
        # lidar_blueprint.set_attribute('channels', str(CHA_NUM))
        # lidar_blueprint.set_attribute('range', str(RANGE))
        # lidar_blueprint.set_attribute('points_per_second', str(PTS_PER_SEC))
        # lidar_blueprint.set_attribute('rotation_frequency', str(ROT_FREQ))
        # lidar_blueprint.set_attribute('upper_fov', str(UPPER_FOV))
        # lidar_blueprint.set_attribute('lower_fov', str(LOWER_FOV))
        # lidar_blueprint.set_attribute('sensor_tick', str(SENSOR_TICK))

        # sensor_blueprint = blueprints.find(SENSOR_TYPE_1_CARLA)
        # sensor_blueprint.set_attribute('image_size_x', str(WINDOW_WIDTH))
        # sensor_blueprint.set_attribute('image_size_y', str(WINDOW_HEIGHT))
        # sensor_blueprint.set_attribute('fov', str(FOV))
        # sensor_blueprint.set_attribute('sensor_tick', str(SENSOR_TICK))

        depth_blueprint = blueprints.find(SENSOR_TYPE_3_CARLA)
        depth_blueprint.set_attribute('image_size_x', str(WINDOW_WIDTH))
        depth_blueprint.set_attribute('image_size_y', str(WINDOW_HEIGHT))
        depth_blueprint.set_attribute('fov', str(DEPTH_FOV))
        depth_blueprint.set_attribute('sensor_tick', str(DEPTH_SENSOR_TICK))

        normal_to_world_blueprint = blueprints.find(SENSOR_TYPE_4_CARLA)
        normal_to_world_blueprint.set_attribute('image_size_x', str(WINDOW_WIDTH))
        normal_to_world_blueprint.set_attribute('image_size_y', str(WINDOW_HEIGHT))
        normal_to_world_blueprint.set_attribute('fov', str(NORMAL_FOV))
        normal_to_world_blueprint.set_attribute('sensor_tick', str(NORMAL_SENSOR_TICK))

        albedo_blueprint = blueprints.find(SENSOR_TYPE_5_CARLA)
        albedo_blueprint.set_attribute('image_size_x', str(WINDOW_WIDTH))
        albedo_blueprint.set_attribute('image_size_y', str(WINDOW_HEIGHT))
        albedo_blueprint.set_attribute('fov', str(ALBEDO_FOV))
        albedo_blueprint.set_attribute('sensor_tick', str(ALBEDO_SENSOR_TICK))

        rgb_blueprint = blueprints.find(SENSOR_TYPE_6_CARLA)
        rgb_blueprint.set_attribute('image_size_x', str(WINDOW_WIDTH))
        rgb_blueprint.set_attribute('image_size_y', str(WINDOW_HEIGHT))
        rgb_blueprint.set_attribute('fov', str(RGB_FOV))
        rgb_blueprint.set_attribute('sensor_tick', str(RGB_SENSOR_TICK))

        normal_to_cam_blueprint = blueprints.find(SENSOR_TYPE_7_CARLA)
        normal_to_cam_blueprint.set_attribute('image_size_x', str(WINDOW_WIDTH))
        normal_to_cam_blueprint.set_attribute('image_size_y', str(WINDOW_HEIGHT))
        normal_to_cam_blueprint.set_attribute('fov', str(RGB_FOV))
        normal_to_cam_blueprint.set_attribute('sensor_tick', str(RGB_SENSOR_TICK))

        reflection_blueprint = blueprints.find(SENSOR_TYPE_8_CARLA)
        reflection_blueprint.set_attribute('image_size_x', str(WINDOW_WIDTH))
        reflection_blueprint.set_attribute('image_size_y', str(WINDOW_HEIGHT))
        reflection_blueprint.set_attribute('fov', str(RGB_FOV))
        reflection_blueprint.set_attribute('sensor_tick', str(RGB_SENSOR_TICK))

        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        # test whether vehicle number exceeds spawn points number
        if len(spawn_points) >= OTHER_VEH_NUM:
            veh_num = OTHER_VEH_NUM
            rest = len(spawn_points) - OTHER_VEH_NUM
            if rest >= OTHER_PED_NUM:
                ped_num = OTHER_PED_NUM
            else:
                ped_num = rest
                print("Do not have so many spawn points. %d vehicles and %d pedestrains created." % (veh_num, ped_num))
        else:
            veh_num = len(spawn_points)
            print("Do not have so many spawn points. %d vehicles and %d pedestrains created." % (veh_num, 0))

        for i in range(veh_num):
            other_veh = world.spawn_actor(random.choice(vehicle_blueprint), spawn_points[i])
            print(other_veh.attributes)
            other_veh.set_autopilot(1)
            actor_list.append(other_veh)

        for i in range(OTHER_PED_NUM):
            ped = world.spawn_actor(random.choice(pedestrain_blueprint), spawn_points[i+OTHER_VEH_NUM])
            print(ped.attributes)
            player_control = carla.WalkerControl()
            player_control.speed = 3
            pedestrian_heading = 90
            player_rotation = carla.Rotation(0, pedestrian_heading, 0)
            player_control.direction = player_rotation.get_forward_vector()
            ped.apply_control(player_control)
            actor_list.append(ped)

        # create my own car
        my_car = world.spawn_actor(random.choice(vehicle_blueprint), random.choice(spawn_points))
        print(my_car.attributes)
        my_car.set_autopilot(1)

        transform = carla.Transform(carla.Location(x=0.8, z=1.65))
        transform_front = carla.Transform(carla.Location(x=0.8, z=1.65))

        # create sensors and attach them to my_car
        depth_front = world.spawn_actor(depth_blueprint, transform_front, attach_to=my_car)
        print(depth_front.attributes)
        # semantic_front.listen(lambda image: save_images(image_saver_front, image))

        normal_to_world_front = world.spawn_actor(normal_to_world_blueprint, transform_front, attach_to=my_car)
        print(normal_to_world_front.attributes)
        # semantic_front.listen(lambda image: save_images(image_saver_front, image))

        albedo_front = world.spawn_actor(albedo_blueprint, transform_front, attach_to=my_car)
        print(albedo_front.attributes)
        # semantic_front.listen(lambda image: save_images(image_saver_front, image))

        semantic_front = world.spawn_actor(sensor_blueprint, transform_front, attach_to=my_car)
        print(semantic_front.attributes)
        # semantic_front.listen(lambda image: save_images(image_saver_front, image))

        rgb_front = world.spawn_actor(rgb_blueprint, transform_front, attach_to=my_car)
        print(rgb_front.attributes)
        # semantic_front.listen(lambda image: save_images(image_saver_front, image))

        normal_to_cam_front = world.spawn_actor(normal_to_cam_blueprint, transform_front, attach_to=my_car)
        print(normal_to_cam_front.attributes)
        # semantic_front.listen(lambda image: save_images(image_saver_front, image))

        reflection_front = world.spawn_actor(reflection_blueprint, transform_front, attach_to=my_car)
        print(reflection_front.attributes)
        # semantic_front.listen(lambda image: save_images(image_saver_front, image))

        # Make sync queue for sensor data.
        image_queue_1 = queue.Queue()
        image_queue_2 = queue.Queue()
        image_queue_3 = queue.Queue()
        image_queue_4 = queue.Queue()
        image_queue_5 = queue.Queue()
        image_queue_6 = queue.Queue()
        image_queue_7 = queue.Queue()

        depth_front.listen(image_queue_1.put)
        normal_to_world_front.listen(image_queue_2.put)
        albedo_front.listen(image_queue_3.put)
        semantic_front.listen(image_queue_4.put)
        rgb_front.listen(image_queue_5.put)
        normal_to_cam_front.listen(image_queue_6.put)
        reflection_front.listen(image_queue_7.put)

        actor_list.append(depth_front)
        actor_list.append(normal_to_world_front)
        actor_list.append(albedo_front)
        actor_list.append(semantic_front)
        actor_list.append(rgb_front)
        actor_list.append(normal_to_cam_front)
        actor_list.append(reflection_front)

        # actor_list.append(sensor_2)
        actor_list.append(my_car)

        display = pygame.display.set_mode(
            (800, 600),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        clock = pygame.time.Clock()
        # begin to record time for stopping
        counter = 0

        while True:
            world.tick()
            clock.tick()
            counter += 1
            

            ts = world.wait_for_tick()

            # image = image_queue.get()
            # image.save_to_disk(
            #     SAVE_PATH+NAME_WITH_TIME+"/"+"rgb/%06d.png" % image.frame_number)

            # image = image_queue.get()
            # image.save_to_disk(
            #     SAVE_PATH+NAME_WITH_TIME+"/"+"albedo/%06d.png" % image.frame_number)

            # image = image_queue.get()
            # image.save_to_disk(
            #     SAVE_PATH+NAME_WITH_TIME+"/"+"normal/%06d.png" % image.frame_number)

            # image = image_queue.get()
            # image.save_to_disk(
            #     SAVE_PATH+NAME_WITH_TIME+"/"+"depth/%06d.png" % image.frame_number)
            if (counter % (TIME_INTER * 10) == 0):
                
                image = image_queue_1.get()
                image.save_to_disk(
                    SAVE_PATH+NAME_WITH_TIME+"/"+"depth/%06d.png" % image.frame_number)

                image = image_queue_2.get()
                image.save_to_disk(
                    SAVE_PATH+NAME_WITH_TIME+"/"+"normal_to_world/%06d.png" % image.frame_number)

                image = image_queue_3.get()
                image.save_to_disk(
                    SAVE_PATH+NAME_WITH_TIME+"/"+"albedo/%06d.png" % image.frame_number)

                image = image_queue_4.get()
                image.save_to_disk(
                    SAVE_PATH+NAME_WITH_TIME+"/"+"semantic/%06d.png" % image.frame_number)

                image = image_queue_5.get()
                image.save_to_disk(
                    SAVE_PATH+NAME_WITH_TIME+"/"+"rgb/%06d.png" % image.frame_number)

                image = image_queue_6.get()
                image.save_to_disk(
                    SAVE_PATH+NAME_WITH_TIME+"/"+"normal_to_cam/%06d.png" % image.frame_number)

                image = image_queue_7.get()
                image.save_to_disk(
                    SAVE_PATH+NAME_WITH_TIME+"/"+"reflection/%06d.png" % image.frame_number)
                
                
                # depth_front.listen(lambda image: )
                # normal_front.listen(lambda image: )
                # albedo_front.listen(lambda image: )
                # rgb_front.listen(lambda image: )
            elif counter % (TIME_INTER * 10) == (TIME_INTER * 10 - 1):    
                for i in range(TIME_INTER * 10 - 1):
                        image_queue_1.get()
                        image_queue_2.get()
                        image_queue_3.get()
                        image_queue_4.get()
                        image_queue_5.get()
                        image_queue_6.get()
                        image_queue_7.get()
            print(counter)
            # if((stop_time - start_time)%1000 == 0):
            #     print("already run {0} seconds.\n".format((stop_time-start_time)/1000))
            if(counter >= STOP_AFTER):
                break

    finally:
        #client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        for actor in actor_list:
            id = actor.id
            actor.destroy()
            print("Actor %d destroyed.\n" % id)

        print('\ndisabling synchronous mode.')
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)

        pygame.quit()
        print('done.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
