from myro import *
import time

class driver():

    def __init__(self):
        import identifier
        self.identifier = identifier.identifier()

        self.x = 0
        self.y = 0
        self.theta = 0

    def survey(self):
        turnBy(39, "deg")

    def drive_to_target_1(self):
        pic = takePicture()
        for i in range(0,10):
            self.identifier.identify_targets(pic, i=i)


driver = driver()
init("COM4")
driver.drive_to_target_1()