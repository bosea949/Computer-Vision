import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import operator
from collections import defaultdict
def get_interest_points(image, feature_width):
    """
    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale of color (your choice)
    -   feature_width: integer representing the local feature width in pixels.

    Returns:
    -   x: A numpy array of shape (N,) containing x-coordinates of interest points
    -   y: A numpy array of shape (N,) containing y-coordinates of interest points
    -   confidences (optional): numpy nd-array of dim (N,) containing the strength
            of each interest point
    -   scales (optional): A numpy array of shape (N,) containing the scale at each
            interest point
    -   orientations (optional): A numpy array of shape (N,) containing the orientation
            at each interest point
    """
    confidences, scales, orientations = None, None, None
    local_max_window=5
    #############################################################################
    #  HARRIS CORNER DETECTOR CODE                                              #
    #############################################################################


    def harris(image,feature_width):

          Ix=cv.Sobel(image,cv.CV_64F,1,0,3)
          Iy=cv.Sobel(image,cv.CV_64F,0,1,3)
          Sxx=cv.GaussianBlur(Ix**2,(5,5),1)
          Sxy=cv.GaussianBlur(Ix*Iy,(5,5),1)
          Syy=cv.GaussianBlur(Iy**2,(5,5),1)

          Corner_response={}
          R={}
          for row in range(image.shape[0]):
              for column in range(image.shape[1]):
                det=Sxx[row,column]*Syy[row,column]-Sxy[row,column]*Sxy[row,column]
                trace=Sxx[row,column]+Syy[row,column]
                R[row,column]=det - 0.06 * (trace**2)

          r_max=max(R.values())
          for row in range((feature_width//2),image.shape[0]-(feature_width//2)):
            for column in range((feature_width//2),image.shape[1]-(feature_width//2)):
              if R[row,column]>0.0001*r_max:  
                local_neighbour_responses=[]
                neighbour_coordinates=list((row + i, column + j) for j in range(-(local_max_window//2), local_max_window//2 + 1) for i in range(-(local_max_window//2), local_max_window//2 + 1))
                neighbour_coordinates.remove((row, column))
                
                max_neighbour_responses=max(list(R[i] for i in neighbour_coordinates if i in R.keys()))

                if max_neighbour_responses<R[row,column]:
                    Corner_response[row,column]=R[row,column]
          return Corner_response
    
    corner_responses=harris(image,feature_width)
    
   
    
    #raise NotImplementedError('`get_interest_points` function in ' +
    #'`harris.py` needs to be implemented')

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    #############################################################################
    # ADAPTIVE NON-MAXIMAL SUPPRESSION CODE :                                   #
    # While most feature detectors simply look for local maxima in              #
    # the interest function, this can lead to an uneven distribution            #
    # of feature points across the image, e.g., points will be denser           #
    # in regions of higher contrast. To mitigate this problem, Brown,           #
    # Szeliski, and Winder (2005) only detect features that are both            #
    # local maxima and whose response value is significantly (10%)              #
    # greater than that of all of its neighbors within a radius r. The          #
    # goal is to retain only those points that are a maximum in a               #
    # neighborhood of radius r pixels. One way to do so is to sort all          #
    # points by the response strength, from large to small response.            #
    # The first entry in the list is the global maximum, which is not           #
    # suppressed at any radius. Then, we can iterate through the list           #
    # and compute the distance to each interest point ahead of it in            #
    # the list (these are pixels with even greater response strength).          #
    # The minimum of distances to a keypoint's stronger neighbors               #
    # (multiplying these neighbors by >=1.1 to add robustness) is the           #
    # radius within which the current point is a local maximum. We              #
    # call this the suppression radius of this interest point, and we           #
    # save these suppression radii. Finally, we sort the suppression            #
    # radii from large to small, and return the n keypoints                     #
    # associated with the top n suppression radii, in this sorted               #
    # orderself. Feel free to experiment with n, we used n=1500.                #
    #                                                                           #
    # See:                                                                      #
    # https://www.microsoft.com/en-us/research/wp-content/uploads/2005/06/cvpr05.pdf
    # or                                                                        #
    # https://www.cs.ucsb.edu/~holl/pubs/Gauglitz-2011-ICIP.pdf                 #
    #############################################################################

    def Adaptive_Non_maximal_suppression(image,harris_responses):
  
          harris_responses={k: v for k, v in sorted(harris_responses.items(), key=lambda item: item[1],reverse=True)}
          suppression_radii={}
          suppression_radii[max(harris_responses.items(), key=operator.itemgetter(1))[0]]=image.shape[0]*image.shape[1]

          for i in list(harris_responses.keys())[1:]:
            neighbours=list(harris_responses.keys())[:list(harris_responses.keys()).index(i)]
            distances={}
            
            for j in neighbours:
              
              distances[np.linalg.norm(np.array([i[0],i[1]])-np.array([j[0],j[1]]))]=[j[0],j[1]]

            minimum_distance=min(list(distances.keys()))
            
            if harris_responses[i]>=0.9*harris_responses[distances[minimum_distance][0],distances[minimum_distance][1]]:
              suppression_radii[i]=minimum_distance

          suppression_radii={k: v for k, v in sorted(suppression_radii.items(), key=lambda item: item[1],reverse=True)}
          
          return list(suppression_radii.keys())
    
    interest_points=Adaptive_Non_maximal_suppression(image,corner_responses)
    
    print("There are ",len(interest_points)," of interest points.")
    x=np.array([i[1]for i in interest_points])
    y=np.array([i[0] for i in interest_points])
	
    # raise NotImplementedError('adaptive non-maximal suppression in ' +
    # '`harris.py` needs to be implemented')

    #############################################################################
    #                             END OF CODE                                   #
    #############################################################################
    return x,y


