#USAGE
# python client_mac.py -a 172.30.1.52

import socket
import time
from imutils.video import VideoStream
import imagezmq
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-a", "--address", required=True, help="IP Address")
args = vars(ap.parse_args())

sender = imagezmq.ImageSender(connect_to='tcp://'+args["address"]+':5555')

rpi_name = socket.gethostname() # send RPi hostname with each image

picam = VideoStream(0).start()
time.sleep(2.0)  # allow camera sensor to warm up

print('[Client] is Activate!')
while True:  # send images as stream until Ctrl-C
  image = picam.read()
  sender.send_image(rpi_name, image)
