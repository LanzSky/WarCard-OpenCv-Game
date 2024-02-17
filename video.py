from threading import Thread
import cv2

class VideoStream:
    
    def __init__(self, resolution=(640,480),framerate=30,PiOrUSB=1,stream_url = 'http://10.187.85.100:8080/video'):

        
        self.PiOrUSB = PiOrUSB

        if self.PiOrUSB == 1: 
            
            self.stream = cv2.VideoCapture(stream_url)
            ret = self.stream.set(3,resolution[0])
            ret = self.stream.set(4,resolution[1])
            

            
            (self.grabbed, self.frame) = self.stream.read()

	
        self.stopped = False

    def start(self):
        Thread(target=self.update,args=()).start()
        return self

    def update(self):

        if self.PiOrUSB == 1: 

            
            while True:
                
                if self.stopped:
                    
                    self.stream.release()
                    return

                
                (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
