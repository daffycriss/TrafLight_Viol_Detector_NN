General Project Information
- This is a scenario simulating a traffic control station at an intersection supported by a traffic light system.
- The system is supported by a trained artificial intelligence model in order to distinguish vehicles passing through the intersection and indicate those passing through a specific road, violating the red light.

Dataset
- https://www.kaggle.com/datasets/andreasmoegelmose/multiview-traffic-intersection-dataset
- It contains a directory with the photos from the selected intersection and .json files with the positions and outlines of the vehicles.
- The dataset includes 35070 photos from consecutive frames of a selected intersection through which vehicles of various types pass.
- Details of the vehicles that appear in each photo (position, dimension, type, etc.) are included in corresponding .json files.
- To convert the data into usable data by the object recognition model, the following folders were created:

data/train/images: To store the photos that will be used in training the model

data/train/label: To store the files that will be used in classicizing the photos

data/val/images: To store the photos that will be used in verifying the model

data/val/label: To store the files that will be used in verifying the model

- The model training is carried out according to the technical specifications of YOLOv8, which require a folder with the photos, and a folder with files of the same name as the corresponding photo, containing the model class and the position and dimensions of the frame that includes the object of the class (normalized) [(e.g. (1 0.3212 0.1515 0.5465 0.2132): (class center_position_x center_position_y distance_x distance_y)]
- For the preparation of the dataset and for the data integrity checks, as well as for
training and using the model, the Python language was used.
- The training was implemented by selecting 30% of the total number of photos of the selected set, 80% was selected for training and the remaining 20% ​​for verification.

[Assumption: The part of the dataset used for training the model is also used to run the simulation along with the remaining 70% of the data]

- The idea of ​​the scenario is to simulate the screen that a traffic control operator watches through a page that is created with Python Flask after calling and checking the trained model.
- The scenario starts with the operator observing incorrect results of the model and then the response team is called to investigate the incident.

Development
- Ubuntu Server 24.04 LTS
- Python ML Libraries
- YOLO: Algorithm for Object Detection

- The data was split into training/execution (30%)
- The training data was split into training/evaluation (80%/20%)
- Image selection is done randomly and not serially

#create_training_data.py

- Simulating the operation of the traffic light during execution
#add_traffic_light.py

- Create a .txt file for each photo, which contains information about the objects (vehicles) such as class, location and size
- The source of the information is the original .json file containing detailed locations and classes of the objects
#add_traffic_light.py

Training the model
------------------
#train_model.py

Information about the model used
--------------------------------
#model_info.py

- Run the simulation
- Create a violation control area
- Use api to call the model live and return a prediction
- Cross-reference all information
- Use Flask (web service) to display the information in a Web environment

#run_simulation.py
