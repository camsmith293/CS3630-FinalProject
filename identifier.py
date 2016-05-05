import math
import numpy as np
import cv2

class identifier():

    def __init__(self):
        self.imageWidth = 1280
        self.imageHeight = 800
        self.numParticles = 1000
        self.initialScale = 50
        self.predictionSigma = 150
        self.x0 = np.array([600, 300])  #seed location for particles
        self.particles = [] #YOUR CODE HERE: make some normally distributed particles
        self.weights = [] #YOUR CODE HERE: make some weights to go along with the particles

        num_spots = (self.imageHeight * self.imageWidth) / self.numParticles

        for i in range(self.numParticles):
            ith = i * num_spots
            x = int(ith % self.imageWidth)
            y = int(ith / self.imageWidth)
            self.particles.append(np.array([x,y]))
            self.weights.append(1.0/self.numParticles)

    def detectBlobs(self, im):
        """ Takes and image and locates the potential location(s) of the red marker
            on top of the robot

        Hint: bgr is the standard color space for images in OpenCV, but other color
              spaces may yield better results

        Note: you are allowed to use OpenCV function calls here

        Returns:
          keypoints: the keypoints returned by the SimpleBlobDetector, each of these
                     keypoints has pt and size values that should be used as
                     measurements for the particle filter
        """

        #YOUR CODE HERE
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = 120
        params.maxThreshold = 200

        params.filterByArea = True
        params.minArea = 5000
        params.maxArea = 100000000

        params.filterByColor = False
        params.blobColor=255
        params.filterByCircularity = False
        params.filterByInertia = False
        params.filterByConvexity = False

        splitted = cv2.split(im)
        thresholded = cv2.threshold(splitted[1], 90, 255, cv2.THRESH_BINARY)[1]

        detector = cv2.SimpleBlobDetector(params)
        keypoints_t = detector.detect(thresholded)
        keypoints = []

        for keypoint in keypoints_t:
            point = keypoint.pt
            if not math.isnan(point[0]):
                keypoints.append(keypoint)

        print(keypoints)
        return keypoints

    def predict(self):
        """ Predict particles one step forward. The motion model should be additive
            Gaussian noise with sigma predictSigma

        Returns:
          particles: list of predicted particles (same size as input particles)
        """

        #YOUR CODE HERE
        new_particles = []
        for particle in self.particles:
            x_t = np.random.normal(scale=self.predictionSigma)
            y_t = np.random.normal(scale=self.predictionSigma)
            new_particles.append(np.array([particle[0]+x_t, particle[1]+y_t]))

        return new_particles

    def update(self, keypoints):
        """ Resample particles and update weights accordingly after particle filter
            update

        Returns:
          newParticles: list of resampled partcles of type np.array
          weights: weights updated after sampling
        """

        #YOUR CODE HERE
        sigma = 100.0
        for i,particle in enumerate(self.particles):
            for keypoint in keypoints:
                point = keypoint.pt
                d = np.linalg.norm(point - particle)
                p = (1.0/np.sqrt(2.0*3.1415*sigma**2)*np.exp(-(d**2)/(2.0*sigma**2)))
                self.weights[i] *= p

        normalized = [float(weight)/float(sum(self.weights)) for weight in self.weights]

        return self.particles, normalized

    def resample(self, particles, weights):
        """ Resample particles and update weights accordingly after particle filter
            update

        Returns:
          newParticles: list of resampled partcles of type np.array
          wegiths: weights updated after sampling
        """
        #YOUR CODE HERE
        newParticles = []
        newWeights = []
        # for i in range(len(particles)):
        #     index = np.random.choice(range(numParticles), p=weights)
        #     p = particles[index]
        #     w = weights[index]
        #     #print(newParticles)
        #     # if not (p in newParticles):
        #     if any((p == x).all() for x in particles):
        #         newParticles.append(p)
        #         newWeights.append(w)
        #     weights = np.linalg.norm(weights)
        indices = (np.random.choice(range(self.numParticles), size=self.numParticles, p=weights).tolist())
        for index in indices:
            newParticles.append(particles[index])
            newWeights.append(1.0/self.numParticles)

        for i in range(self.numParticles - len(indices)):
            x = np.random.randint(0, self.imageWidth)
            y = np.random.randint(0, self.imageHeight)
            newParticles.append(np.array([x,y]))
            newWeights.append(1.0/self.numParticles)

        return newParticles, newWeights

    def visualizeParticles(self, im, particles, weights, color=(0,0,255)):
        """ Plot particles as circles with radius proportional to weight, which
            should be [0-1], (default color is red). Also plots weighted average
            of particles as blue circle. Particles should be a numpy.ndarray of
            [x, y] particle locations.

        Returns:
          im: image with particles overlaid as red circles
        """
        im_with_particles = im.copy()
        s = (0, 0)
        for i in range(0, len(particles)):
            s += particles[i]*weights[i]
            cv2.circle(im_with_particles, tuple(particles[i].astype(int)), radius=int(weights[i]*250), color=(0,0,255), thickness=3)
        cv2.circle(im_with_particles, tuple(s.astype(int)), radius=3, color=(255,0,0), thickness=6)
        return im_with_particles

    def visualizeKeypoints(self, im, keypoints, color=(0,255,0)):
        """ Draw keypoints generated by blob detector on image in color specified
            (default is green)

        Returns:
          im_with_keypoints: the image with keypoints overlaid
        """
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), color, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return im_with_keypoints

    def convert_image_to_array(self, image):
        im_list = []

        for i in range(0, self.imageHeight):
            row = []
            for j in range(0, self.imageWidth):
                rgb = image.getPixel(j, i).getRGB()
                row.append([int(rgb[0]), int(rgb[1]), int(rgb[2])])
            im_list.append(row)

        im = np.array(im_list)
        cv2.imshow("Reading", im)
        cv2.waitKey(0)
        return im


    def identify_targets(self, image, i=0):
        im = self.convert_image_to_array(image)
        yuv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)

        #visualize particles
        im_to_show = self.visualizeParticles(im, self.particles, self.weights)
        cv2.imwrite("particles_&d.jpg" % i, im_to_show)

        #predict forward
        particles = self.predict(self.particles, self.predictionSigma)
        im_to_show = self.visualizeParticles(im, particles, self.weights)
        cv2.imwrite('predictions_%d.jpg' % i, im_to_show)

        #detected keypoint in measurement
        keypoints = self.detectBlobs(yuv)

        #update paticleFilter using measurement if there was one]
        if keypoints:
            particles, weights = self.update(particles, self.weights, keypoints)

        im_to_show = self.visualizeKeypoints(im, keypoints)
        im_to_show = self.visualizeParticles(im_to_show, particles, weights)
        cv2.imwrite('updates_%d.jpg' % i, im_to_show)

        #resample particles
        particles, weights = self.resample(particles, weights)
        im_to_show = self.visualizeKeypoints(im, keypoints)
        im_to_show = self.visualizeParticles(im_to_show, particles, weights)
        cv2.imwrite('resampled_%d.jpg' % i, im_to_show)
        #cv2.waitKey(0)