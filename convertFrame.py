# Importing all necessary libraries
import cv2
import os
  
def extractFrameLoc(path_of_file, cordinate):
    # Read the video from specified path
    cam = cv2.VideoCapture(path_of_file)
    
    try:
        
        # creating a folder named data
        if not os.path.exists('data'):
            os.makedirs('data')
    
    # if not created then raise error
    except OSError:
        print ('Error: Creating directory of data')
    
    # frame
    currentframe = 0
    
    while(True):
        
        # reading from frame
        ret,frame = cam.read()
    
        if ret:
            # if video is still left continue creating images
            name = './data/' + cordinate + '_' + str(currentframe) + '.jpg'
            print ('Creating...' + name)
    
            # writing the extracted images
            cv2.imwrite(name, frame)
    
            # increasing counter so that it will
            # show how many frames are created
            currentframe += 1
        else:
            break
    
    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()


path1 =  "s1.mp4"
cordinate1 = "12.841837-80.155035"
path2 = "s2.mp4"
cordinate2 = "12.843405-80.154573"
extractFrameLoc(path1, cordinate1)