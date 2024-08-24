# Drone-Based-Waste-Monitoring

### ABSTRACT 
The concept of automation utilized in waste management systems falls under the category of cleanliness and hygiene. In all developing nations, it is normal practice to dump trash on the ground and in public spaces, which has a negative impact on the environment and leads to various unsanitary problems. In order to deal with these problems Smart drone is an ideology put forward which is a combination of hardware and software technologies i.e., connecting Wi-Fi system to the normal drone in order to provide free internet facilities to the user for a particular period of time. 
The technology helps the government officials for keeping the surrounding clean and thus work hand in hand for the proper waste management in a locality. Drones can be equipped with cameras and sensors to monitor and map waste in real-time, enabling efficient and accurate waste finding. Smart drones use multiple technologies, firstly the technology for classifying trash dumped, secondly the movement of the waste and lastly sending necessary signals and connecting the government officials to the Wi-Fi system. The use of drones in waste management has the potential to improve the efficiency and effectiveness of waste collection and disposal processes, reduce costs, and promote environmental sustainability. The proposed system will function on a cause that will assure a clean environment, good health, and pollution free society. 
  
  
 
 
### PROBLEM STATEMENT  
As we know our surroundings are not neat and clean which leads to various problems such as blockage of drains, polluted environment, unhealthy atmosphere, unhygienic environment and other problems. 
This mainly occurred because municipality and government officials fail to clean the public places, so to solve the problem we propose an idea by which we can monitor our surroundings through drones. The drone will help us to identify where garbage is present in our surrounding this will be done by camera module present in the drone which will classify the image based on image classification model and their respective locations will be send to the government officials so that they can have the proper locations of the garbage beforehand and then they can clean it. This will help us to keep our surroundings neat and clean.  
### OBJECTIVES: 
- Real-time monitoring: IoT sensors on drones can provide real-time information on waste levels, helping waste management companies to optimize their collection routes and schedules. This can reduce fuel consumption and carbon emissions, while also improving the overall efficiency of waste management operations. 
- Environmental protection: Effective waste management through drones can reduce the negative impact of waste on the environment, including air and water pollution, greenhouse gas emissions, and damage to natural habitats. This can help protect the environment and promote sustainable development. 
- Efficient waste collection: Where municipal trucks of garbage cannot go, drones can help in finding the place and municipalities can send small vehicles to collect the garbage there. This reduces the time for heavy trucks to go the narrow roads and saves time. It maintains a high compaction rate. 

### Problem Identification in the Domain 
1. Data collection approach and strategies: 
  
- GITHUB dataset  
 
Images collected from the Drone camera and converting them into a required dataset format. 
  
2. Data Analysis Approach 
  
Choosing the best Deep Learning model where it identifies the percentage of waste as well as analyzing the waste present in the particular area so that the municipality must be alerted to clean that area and also introduce a garbage dustbin in that area for this we have used mobilenetV3 model for image classification. MobileNetV3 is a convolutional neural network architecture designed for efficient and lightweight computer vision tasks on mobile devices and embedded systems for drones. It is a state-of-the-art deep learning model that is optimized for running on low-power devices with limited computational resources and memory. MobileNetV3 for drones is used for a variety of computer vision applications, such as object detection, tracking, and recognition. The lightweight nature of the network allows it to be run in real-time on small computing devices, making it ideal for use on drones, which typically have limited computing power and battery life. 
  
### PROPOSED METHODOLOGY – 
  
We came up with the idea of smart drones. We will use a drone simulator so that the drone will fly in the given locality and will click images of areas in the locality. Then we will perform image classification on the images to identify the waste or garbage present in the locality and we will send the data to the municipality to clean the localities using automatic email simulation. We will be using deep learning models to train the dataset and classify them into 9 classes of waste namely 'Aluminum', 'Carton', 'Glass', 'Organic Waste', 'Other Plastics', 'Paper and Cardboard', 'Plastic', 'Textiles', 'Wood' and then we will test the model using our drone that will take video of locality and transform them into frames using python code. Then the frame or picture will be tested from our model and waste will be identified and then the email will be sent to the nearby municipal council to clean that respective area.  
  
This will help to maintain a clean environment as no mentality change needs to be established on the people of society and also it will be reliable. 
 
### PROPOSED ARCHITECTURE – 
  
- Drone deployment: The first step in this architecture is to deploy drones equipped with high-resolution cameras to fly over designated areas and collect images of the waste. The drones can be manually operated or programmed to fly autonomously using GPS navigation. To cover a larger area, multiple drones can be used simultaneously. 
 
- Image Transferring: The images captured by the drones are then send to the cloud for further processing. Cloud processing requires a stable internet connection and sufficient cloud resources. 
 
- Image analysis: The image data from the drone should be transmitted to the cloud or to the main pc  for further analysis. The cloud-based system will have a Deep Learning Model that can classify the waste into different categories such as plastic, metal, glass, organic, etc. and also detect the presence of waste. The Deep Learning Model is trained using a large dataset of waste images to ensure high accuracy. 
 
- Waste Detection: Based on the classification results, the waste can be classified and detected. Now using this, the appropriate coordinates of the areas where waste was detected is further passed as an output from the DL model.  
 
- Data analytics: The data collected from the drone and cloud-based system can be used to generate insights on the waste generation patterns, waste composition, and recycling rates. These insights can be used to optimize the waste management process and make informed decisions. For example, the data can be used to identify areas with high waste generation rates and prioritize waste management efforts accordingly. 
 
- Maintenance and monitoring: The drones and the cloud-based system should be regularly maintained and monitored to ensure optimal performance and reliability. Any issues or faults should be promptly addressed to minimize downtime and ensure the smooth functioning of the system. Regular maintenance and monitoring can help extend the lifespan of the equipment and reduce the risk of costly repairs. 
 
 
### Communication Model Used 
  
Publisher-Subscriber 
This model comprises three entities: Publishers, Brokers, and Consumers. 
- Publishers are the source of data. It sends the data to the topic which are managed by the broker. They are not aware of consumers. 
- Consumers subscribe to the topics which are managed by the broker. 
- Hence, Brokers responsibility is to accept data from publishers and send it to the appropriate consumers. The broker only has the information regarding the consumer to which a particular topic belongs to which the publisher is unaware of. 
 
The Publisher can be any component on the drone that generates data, such as a sensor or a flight controller. The Subscribers can be other components on the drone, such as an autopilot or a camera, or they can be external devices, such as a ground station or a remote controller. 
The Publisher-Subscriber model in drones enables efficient and flexible communication between different components, allowing them to share information without the need for direct communication or knowledge of each other's existence. This can improve the overall performance and functionality of the drone, as well as make it easier to integrate with other devices and systems. 
 
### IMPLEMENTATION AND RESULTS :  
 
We have used AirSim drone simulator to make our drone simulator  
In AirSim we can add the coordinates of the drone and we can also manage the speed of the drone  
  
  
Once coordinates are set it will record the video of the environment and we are recording the video at 30fps .  
Then we will extract frames from the video and it will be converted to image  
 
We will save the image name with its coordinate so that we can easily get the location of the waste  
 
Then this image will be send to our deep learning model which is trained using 
Mobilenetv3 model and we have a total of 9 classes to classify different waste type 
 
  
### CONCLUSION -  
In conclusion, the proposed architecture for solid waste management through drones and cloud analysis offers a scalable, efficient, and cost-effective solution for managing waste. It can help reduce the environmental impact of waste and contribute to the sustainable development of our communities. Waste management through drones has the potential to revolutionize the way we manage waste. With the help of advanced sensors and cameras, drones can be used to classify and sort waste into different categories such as 'Aluminum', 'Carton', 'Glass', 'Organic Waste', 'Other Plastics', 'Paper and Cardboard', 'Plastic', 'Textiles', 'Wood'. Drones can also cover a larger area in a shorter amount of time, which can result in faster and more effective waste management. Drones will tell the exact place of waste and type of waste from where we need to collect the waste in a shorter period of time. This can help in improving the efficiency and effectiveness of waste management processes. Overall, the use of drones for waste management can bring about significant improvements in the way we handle and dispose of waste. With further advancements in technology and increased adoption of this approach, we can expect to see a cleaner and more sustainable environment. However, the implementation of this architecture requires significant investment in hardware, software, and cloud resources, as well as regulatory approvals and community support. 
 
### FUTURE ENHANCEMENTS -  
There are numerous future developments that could increase the efficacy and efficiency of this strategy, which has the potential to revolutionize the waste management sector. Using drones with sophisticated sensors and machine learning algorithms to classify the types of waste existing in a certain location is one potential improvement. 
Drones might gather comprehensive data on the garbage that is present in a specific area, including its composition, volume, and location, using sensors like cameras, LIDAR, and other types of remote sensing technology. The exact sorts of garbage that are present, such as plastics, paper, or organic materials, might then be identified using machine learning algorithms that interpret this information. 
The drones might then be used to transfer the material to the proper disposal or recycling site after it has been sorted. For instance, drones could be programmed to deliver rubbish to a nearby recycling plant that specialized in processing plastic materials if they detect a considerable amount of plastic waste in a specific location. 
In general, employing drones for trash management has the potential to greatly increase the effectiveness and efficiency of this vital sector. Drones could assist to classify waste more precisely and carry it to the proper facilities more rapidly by integrating cutting-edge sensors and machine learning algorithms, minimizing the environmental impact of waste and promoting a more sustainable future. 
