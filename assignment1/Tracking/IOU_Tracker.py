import numpy as np

class IOUTracker:
    def __init__(self, iou_threshold=0.7):
        # 1. Initialize the IOU threshold and required variables like trackers and next_id.
        self.next_id=0
        # pass
        self.thresh = iou_threshold
        self.trackers = []

    def _compute_iou(self, box1, box2):
        """
        2. Implement the IOU computation between two bounding boxes.
        - This involves calculating the intersection and union areas of the boxes.
        - Return the IOU score (intersection / union).
        """
        # pass
        #finding coordinates of intersecting box if any???
        #top left corner
        xtl = max(box1[0], box2[0])
        ytl = max(box1[1], box2[1])
        #bottom right corner
        xbr = min(box1[2], box2[2])
        ybr = min(box1[3], box2[3])

        if xtl >= xbr or ytl >= ybr:
            return 0 #no area of intersection, so iou score is 0

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        intersection_area = (xbr - xtl) * (ybr - ytl)
        union_area = box1_area + box2_area - intersection_area

        return intersection_area / union_area #iou score
        

    def update(self, detections):
        """
        3. Implement the update method to update the trackers with new detections.
        - Perform greedy matching using IOU: compare each tracker with new detections.
        - If IOU > threshold, match the detection to the tracker.
        - If no match is found for a detection, create a new tracker.
        - Remove trackers that don't match any detection from the previous frame.
        - Input Format : [xmax,ymin,xmax,ymax,score,class]
        - Return the updated list of trackers in the format: [xmin, ymin, xmax, ymax, track id, class, score].
        """
        # pass
        new_trackers = []

        for det in detections:
            match = False
            for track in self.trackers:
                iou = self._compute_iou(track[:4], det[:4])
                if iou > self.thresh:
                    match = True
                    new_trackers.append([det[0],det[1],det[2],det[3],track[-3],det[5],det[4]]) #same trackID and new coords
                    break

            if not match: #create new tracker if no match
                new_trackers.append([det[0],det[1],det[2],det[3],self.next_id,det[5],det[4]]) #new trackID and new coords
                self.next_id += 1

        self.trackers = new_trackers
        return new_trackers