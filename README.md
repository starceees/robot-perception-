# robot-perception

## System Overview:
The cornerstone of our system is the KeyboardPlayerPyGame class, which facilitates both manual and autonomous navigation. By utilizing libraries such as PyGame for interaction and OpenCV for computer vision, the class enables the robot to explore, detect targets, and record navigational commands for future use.

## Exploration Phase:
During the manual exploration phase, the operator directs the robot using keyboard inputs. The system captures and records each command, along with the corresponding visual data from the robot's first-person view (FPV). This phase lays the groundwork for autonomous navigation by building a repository of navigational data.
![switch 4](https://github.com/starceees/robot-perception-/assets/51673079/dcc11aa4-de2a-4a4c-82aa-0dd57dd82e40)
![switch 3](https://github.com/starceees/robot-perception-/assets/51673079/9978a542-6df1-4bed-89b2-8d8f7c30b2c5)
![switch2](https://github.com/starceees/robot-perception-/assets/51673079/bc53292a-7c0f-44c3-b335-619328a89754)
![Switch1](https://github.com/starceees/robot-perception-/assets/51673079/64036399-60d1-4e86-8576-724a23da78f9)


## Autonomous Navigation Phase:
Leveraging the recorded data, the robot can autonomously retrace its steps, using visual cues to navigate through previously explored areas. The ORB detector and FLANN matcher work in tandem to enable real-time target detection and path plotting.

## Feature Matching and Path Plotting:
Our approach uses the ORB algorithm to detect keypoints and compute descriptors. The FLANN matcher then identifies similar features in sequential images, allowing the robot to understand its movement and detect dead-ends.
![result - 2](https://github.com/starceees/robot-perception-/assets/51673079/5bb229c8-4459-4672-b40f-07d08850d8d9)

## Command Recording:
Each action taken by the robot is meticulously recorded, capturing the command type, duration, and associated FPV image. This historical log is crucial for the robot's learning and provides a rich dataset for navigation optimization.

## Target Detection and Positioning:
Through continuous feature matching, the system detects when a target enters the robot's FPV. It then records the robot's position, updating the map with the target's location.

## Navigation Commands and Path Visualization:
The robot executes recorded commands to navigate, with the path and targets visualized in a 2D space. This visualization helps in monitoring the robot's progress and analyzing its navigational efficiency.

![result - 1](https://github.com/starceees/robot-perception-/assets/51673079/fdb641b2-a34d-447e-840f-0fa32d2c857d)
