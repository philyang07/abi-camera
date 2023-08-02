#!/usr/bin/env python3

from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import time
import argparse
from datetime import datetime

import rospy
from std_msgs.msg import String, Float32
from geometry_msgs.msg import Vector3

labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

nnPathDefault = str((Path(__file__).parent / Path('../models/mobilenet-ssd_openvino_2021.4_5shave.blob')).resolve().absolute())
# parser = argparse.ArgumentParser()
# parser.add_argument('nnPath', nargs='?', help="Path to mobilenet detection network blob", default=nnPathDefault)
# parser.add_argument('-ff', '--full_frame', action="store_true", help="Perform tracking on full RGB frame", default=False)

# args = parser.parse_args()

fullFrameTracking = False

# Constants for start and end hour
START_HOUR = 7 # 7am
END_HOUR = 19 # 7pm

class CameraInterface:
    def __init__(self):
        # Coordinates publisher node
        rospy.init_node('camera_interface', anonymous=False)
        self.pub_location = rospy.Publisher('camera/person_location', Vector3, queue_size=10)
        # self.pub_angle = rospy.Publisher('camera/person_angle', Float32, queue_size=10)

        self.vec_history = [] # Store last 5 values

        # Is running or not
        self.running = False

        # Create pipeline
        self.pipeline = dai.Pipeline()

        # Define sources and outputs
        camRgb = self.pipeline.create(dai.node.ColorCamera)
        spatialDetectionNetwork = self.pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
        monoLeft = self.pipeline.create(dai.node.MonoCamera)
        monoRight = self.pipeline.create(dai.node.MonoCamera)
        stereo = self.pipeline.create(dai.node.StereoDepth)
        objectTracker = self.pipeline.create(dai.node.ObjectTracker)

        xoutRgb = self.pipeline.create(dai.node.XLinkOut)
        trackerOut = self.pipeline.create(dai.node.XLinkOut)

        xoutRgb.setStreamName("preview")
        trackerOut.setStreamName("tracklets")

        # Properties
        camRgb.setPreviewSize(300, 300)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        # setting node configs
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        # Align depth map to the perspective of RGB camera, on which inference is done
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())

        spatialDetectionNetwork.setBlobPath(nnPathDefault)
        spatialDetectionNetwork.setConfidenceThreshold(0.5)
        spatialDetectionNetwork.input.setBlocking(False)
        spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
        spatialDetectionNetwork.setDepthLowerThreshold(100)
        spatialDetectionNetwork.setDepthUpperThreshold(5000)

        objectTracker.setDetectionLabelsToTrack([15])  # track only person
        # possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS, SHORT_TERM_IMAGELESS, SHORT_TERM_KCF
        objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
        # take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
        objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

        # Linking
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

        camRgb.preview.link(spatialDetectionNetwork.input)
        objectTracker.passthroughTrackerFrame.link(xoutRgb.input)
        objectTracker.out.link(trackerOut.input)

        if fullFrameTracking:
            camRgb.setPreviewKeepAspectRatio(False)
            camRgb.video.link(objectTracker.inputTrackerFrame)
            objectTracker.inputTrackerFrame.setBlocking(False)
            # do not block the pipeline if it's too slow on full frame
            objectTracker.inputTrackerFrame.setQueueSize(2)
        else:
            spatialDetectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)

        spatialDetectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
        spatialDetectionNetwork.out.link(objectTracker.inputDetections)
        stereo.depth.link(spatialDetectionNetwork.inputDepth)

    # Returns true if time is 7am to 7pm
    def time_enabled(self):
        now = datetime.now()
        todayStart = now.replace(hour=START_HOUR, minute=0, second=0, microsecond=0)
        todayEnd = now.replace(hour=END_HOUR, minute=0, second=0, microsecond=0)
        return now >= todayStart and now <= todayEnd

    def run(self):
        self.running = True

        # Connect to device and start pipeline
        with dai.Device(self.pipeline) as device:

            preview = device.getOutputQueue("preview", 4, False)
            tracklets = device.getOutputQueue("tracklets", 4, False)

            startTime = time.monotonic()
            counter = 0
            fps = 0
            color = (255, 255, 255)

            rate = rospy.Rate(3)
            while self.time_enabled():
                imgFrame = preview.get()
                track = tracklets.get()

                counter+=1
                current_time = time.monotonic()
                if (current_time - startTime) > 1 :
                    fps = counter / (current_time - startTime)
                    counter = 0
                    startTime = current_time

                frame = imgFrame.getCvFrame()
                trackletsData = track.tracklets

                closest_vec = None

                for t in trackletsData:
                    roi = t.roi.denormalize(frame.shape[1], frame.shape[0])
                    x1 = int(roi.topLeft().x)
                    y1 = int(roi.topLeft().y)
                    x2 = int(roi.bottomRight().x)
                    y2 = int(roi.bottomRight().y)

                    try:
                        label = labelMap[t.label]
                    except:
                        label = t.label

                    # cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    # cv2.putText(frame, f"ID: {[t.id]}", (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    # cv2.putText(frame, t.status.name, (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    # cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

                    # cv2.putText(frame, f"X: {int(t.spatialCoordinates.x)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    # cv2.putText(frame, f"Y: {int(t.spatialCoordinates.y)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    # cv2.putText(frame, f"Z: {int(t.spatialCoordinates.z)} mm", (x1 + 10, y1 + 95), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    vec = Vector3()
                    vec.x =  int(t.spatialCoordinates.x)
                    vec.y = int(t.spatialCoordinates.y)
                    vec.z = int(t.spatialCoordinates.z)
                    
                    # self.pub_location.publish(vec)
                    # self.pub_angle.publish(lerp((x1+x2)/2, 0, 300, -49, 49))
                    # print(vec.x, vec.y, vec.z)

                    if (closest_vec == None or vec.x ** 2 + vec.y ** 2 < closest_vec.x ** 2 + closest_vec.y ** 2):
                        closest_vec = vec

                # If invalid vec or no vec then append a z=1500
                if closest_vec == None:
                    closest_vec = Vector3(x=0, y=0, z=0)
                if closest_vec.z <= 0:
                    closest_vec.z = 1500
                self.vec_history.append(closest_vec)
                while len(self.vec_history) > 5:
                    self.vec_history.pop(0)
                # Get mean vector
                x_sum = 0
                y_sum = 0
                z_sum = 0
                for vec in self.vec_history:
                    x_sum += vec.x
                    y_sum += vec.y
                    z_sum += vec.z
                return_vec = Vector3()
                return_vec.x = x_sum/len(self.vec_history)
                return_vec.y = y_sum/len(self.vec_history)
                return_vec.z = z_sum/len(self.vec_history)

                self.pub_location.publish(return_vec)
                rate.sleep()
                # cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)

                # cv2.imshow("tracker", frame)
            self.running = False

def lerp(var, in_start, in_end, out_start, out_end):
    slope = (out_end - out_start)/(in_end - in_start)
    return out_start + slope * (var - in_start)

if __name__ == '__main__':
   camera_interface = CameraInterface()
   try:
       rate = rospy.Rate(2)
       while not rospy.is_shutdown():
           if camera_interface.time_enabled() and not camera_interface.running: 
               camera_interface.run()
           rate.sleep()
   except rospy.ROSInterruptException:
    pass