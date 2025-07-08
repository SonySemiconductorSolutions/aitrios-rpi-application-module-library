---
title: Calculate
sidebar_position: 5
---
import ApiLink from '@site/src/components/ApiLink';

# Calculate

The <ApiLink to="/api-reference/apps/calculate">Calculate</ApiLink> module is designed to contain components focused around calculating values for applications. It contains SpeedCalculator, a class focused on calculating the speed of tracked objects over time by calculating the change in distance over time. Also contains angle calculation, a function to calculate the angle between 3 keypoints values. Finally a few small functions to calculate the distance between two points in 2D pixel space and a function to calculate the center point of a bbox/4 points.

## SpeedCalculator

The SpeedCalculator can be used to calulate the speed of objects, such as:
- Calculating the speed of traffic.
- Calculating the speed of objects for sports games.

The SpeedCalculator required the use of a tracker, such as [BYTETracker](examples/tracker.md) to track objects over time. 

![Speed](gifs/speed.gif)

Below an example of how one can use the SpeedCalculator in the Application Module Library.


```python title="speed_calc.py"
import numpy as np 
from modlib.apps import Annotator, ColorPalette, BYTETracker
from modlib.apps.calculate import SpeedCalculator
from modlib.devices import AiCamera
from modlib.models.zoo import NanoDetPlus416x416

class BYTETrackerArgs:
	track_thresh: float = 0.30
	track_buffer: int = 30
	match_thresh: float = 0.8
	aspect_ratio_thresh: float = 3.0
	min_box_area: float = 1.0
	mot20: bool = False

model = NanoDetPlus416x416()
device = AiCamera()
device.deploy(model)

distance_per_pixel = 0.00742 # Need to recalibrate when camera is repositioned
tracker = BYTETracker(BYTETrackerArgs())
speed = SpeedCalculator()
annotator = Annotator(color=ColorPalette.default(), thickness=1, text_thickness=1, text_scale=0.4)

with device as stream:
	for frame in stream:
		detections = frame.detections[frame.detections.confidence > 0.50]
		detections = tracker.update(frame, detections)
		
		# Calculate and retrieve speed per tracked object.
		speed.calculate(frame, detections)
		current_speeds = [speed.get_speed(t, average=False) for t in detections.tracker_id]
		
		labels = [f"{s*distance_per_pixel*3.6:0.2f}kph" if s is not None else "..." for s in current_speeds]
		frame.image = annotator.annotate_boxes(frame=frame, detections=detections, labels=labels)
		frame.display()
```

## Distance Calculator

Distance Calculator can be used to calculate the distance between objects. It returns the distance in pixels. To convert to real world values you can use multiple by a factor where 1 pixel is a certain distance. speed_cal.py does this to convert from pixels per hour to kph.

![Distance](gifs/distance.gif)

```python title="distance.py"
import cv2
import numpy as np

from modlib.apps import Annotator, ColorPalette
from modlib.apps.calculate import calculate_distance_matrix
from modlib.devices import AiCamera
from modlib.models.zoo import NanoDetPlus416x416

model = NanoDetPlus416x416()
device = AiCamera()
device.deploy(model)

annotator = Annotator(color=ColorPalette.default(), thickness=1, text_thickness=1, text_scale=0.4)

with device as stream:
	for frame in stream:
		detections = frame.detections[frame.detections.confidence > 0.50]

		# Calculate distance matrix
		xc, yc = detections.center_points
		xc, yc = xc*frame.width, yc*frame.height
		dist_matrix = calculate_distance_matrix(xc, yc)

		# Display distance to each objects
		indeces = np.triu_indices(len(xc), k=1)
		for i, j in zip(*indeces):
			distance = dist_matrix[i, j]
			p1 = (int(xc[i]), int(yc[i]))
			p2 = (int(xc[j]), int(yc[j]))

			cv2.line(frame.image, p1, p2, (255, 255, 255), 1)
			cv2.putText(frame.image, f"{distance:.1f}", ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

		labels = [f"{model.labels[class_id]}: {score:0.2f}" for _, score, class_id, _ in detections]
		frame.image = annotator.annotate_boxes(frame=frame, detections=detections, labels=labels)

		frame.display()
```
