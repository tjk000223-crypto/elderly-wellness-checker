"""
Face Analysis Module for Elderly Wellness Checker
Analyzes fatigue, discomfort, skin color changes, and facial symmetry
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Tuple, Optional


class FaceAnalyzer:
    def __init__(self):
        """Initialize MediaPipe face mesh for facial landmark detection"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Eye landmark indices (MediaPipe 468 landmarks)
        # Left eye: [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        # Right eye: [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Cheek and lip indices
        self.LEFT_CHEEK_INDICES = [116, 117, 118, 119, 120, 121]
        self.RIGHT_CHEEK_INDICES = [345, 346, 347, 348, 349, 350]
        self.UPPER_LIP_INDICES = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
        self.LOWER_LIP_INDICES = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324]
        
        # Facial symmetry points
        self.SYMMETRY_POINTS = {
            'left_eyebrow': [70, 63, 105, 66, 107],
            'right_eyebrow': [300, 293, 334, 296, 336],
            'left_mouth': [61, 146, 91, 181, 84],
            'right_mouth': [291, 375, 321, 308, 324],
            'nose_tip': [4],
            'chin': [175]
        }
        
        # State tracking for blink detection
        self.eye_ar_history = []
        self.blink_count = 0
        self.blink_threshold = 0.25  # Eye aspect ratio threshold for blink
        self.ear_consecutive_frames = 0
        self.ear_consecutive_threshold = 3
        
    def calculate_eye_aspect_ratio(self, eye_landmarks, landmarks):
        """Calculate Eye Aspect Ratio (EAR) for fatigue detection"""
        # Get 2D coordinates of eye landmarks
        eye_points = []
        for idx in eye_landmarks:
            if idx < len(landmarks.landmark):
                x = landmarks.landmark[idx].x
                y = landmarks.landmark[idx].y
                eye_points.append([x, y])
        
        if len(eye_points) < 6:
            return None
        
        eye_points = np.array(eye_points)
        
        # Calculate distances
        # Vertical distances
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        # Horizontal distance
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        
        # EAR formula
        ear = (A + B) / (2.0 * C)
        return ear
    
    def detect_fatigue(self, landmarks) -> Dict:
        """Detect fatigue based on eye openness and blink rate"""
        left_ear = self.calculate_eye_aspect_ratio(self.LEFT_EYE_INDICES, landmarks)
        right_ear = self.calculate_eye_aspect_ratio(self.RIGHT_EYE_INDICES, landmarks)
        
        if left_ear is None or right_ear is None:
            return {
                'fatigue_detected': False,
                'eye_openness': None,
                'blink_rate': None,
                'confidence': 0.0
            }
        
        avg_ear = (left_ear + right_ear) / 2.0
        self.eye_ar_history.append(avg_ear)
        
        # Keep only last 30 frames for blink rate calculation
        if len(self.eye_ar_history) > 30:
            self.eye_ar_history.pop(0)
        
        # Detect blink
        if avg_ear < self.blink_threshold:
            self.ear_consecutive_frames += 1
        else:
            if self.ear_consecutive_frames >= self.ear_consecutive_threshold:
                self.blink_count += 1
            self.ear_consecutive_frames = 0
        
        # Calculate blink rate (blinks per minute, assuming ~30 FPS)
        blink_rate = (self.blink_count / len(self.eye_ar_history)) * 30 * 60 if len(self.eye_ar_history) > 0 else 0
        
        # Normal blink rate: 15-20 blinks per minute
        # Fatigue indicators: low eye openness (< 0.2) or abnormal blink rate
        fatigue_detected = avg_ear < 0.2 or (blink_rate < 10 or blink_rate > 30)
        confidence = 1.0 - min(avg_ear / 0.3, 1.0) if avg_ear < 0.3 else 0.0
        
        return {
            'fatigue_detected': fatigue_detected,
            'eye_openness': float(avg_ear),
            'blink_rate': float(blink_rate),
            'confidence': float(confidence),
            'left_ear': float(left_ear),
            'right_ear': float(right_ear)
        }
    
    def analyze_skin_color(self, image, landmarks, image_shape) -> Dict:
        """Analyze skin color changes in cheeks and lips"""
        h, w = image_shape[:2]
        
        # Extract cheek regions
        left_cheek_colors = []
        right_cheek_colors = []
        lip_colors = []
        
        # Sample colors from cheek regions
        for idx in self.LEFT_CHEEK_INDICES:
            if idx < len(landmarks.landmark):
                x = int(landmarks.landmark[idx].x * w)
                y = int(landmarks.landmark[idx].y * h)
                if 0 <= x < w and 0 <= y < h:
                    # Sample a small region around the point
                    region = image[max(0, y-5):min(h, y+5), max(0, x-5):min(w, x+5)]
                    if region.size > 0:
                        left_cheek_colors.append(region.mean(axis=(0, 1)))
        
        for idx in self.RIGHT_CHEEK_INDICES:
            if idx < len(landmarks.landmark):
                x = int(landmarks.landmark[idx].x * w)
                y = int(landmarks.landmark[idx].y * h)
                if 0 <= x < w and 0 <= y < h:
                    region = image[max(0, y-5):min(h, y+5), max(0, x-5):min(w, x+5)]
                    if region.size > 0:
                        right_cheek_colors.append(region.mean(axis=(0, 1)))
        
        # Sample colors from lip regions
        lip_indices = self.UPPER_LIP_INDICES + self.LOWER_LIP_INDICES
        for idx in lip_indices:
            if idx < len(landmarks.landmark):
                x = int(landmarks.landmark[idx].x * w)
                y = int(landmarks.landmark[idx].y * h)
                if 0 <= x < w and 0 <= y < h:
                    region = image[max(0, y-3):min(h, y+3), max(0, x-3):min(w, x+3)]
                    if region.size > 0:
                        lip_colors.append(region.mean(axis=(0, 1)))
        
        if not left_cheek_colors or not right_cheek_colors or not lip_colors:
            return {
                'cheek_color': None,
                'lip_color': None,
                'color_change_detected': False,
                'confidence': 0.0
            }
        
        # Calculate average colors
        avg_left_cheek = np.mean(left_cheek_colors, axis=0)
        avg_right_cheek = np.mean(right_cheek_colors, axis=0)
        avg_cheek = (avg_left_cheek + avg_right_cheek) / 2.0
        avg_lip = np.mean(lip_colors, axis=0)
        
        # Convert BGR to RGB for analysis
        cheek_rgb = avg_cheek[::-1]  # BGR to RGB
        lip_rgb = avg_lip[::-1]
        
        # Analyze color changes
        # Pale: low saturation, high lightness
        # Red: high red component
        # Purple/blue: high blue component relative to red
        
        # Calculate saturation and lightness
        cheek_max = np.max(cheek_rgb)
        cheek_min = np.min(cheek_rgb)
        cheek_saturation = (cheek_max - cheek_min) / (cheek_max + 1e-6)
        cheek_lightness = np.mean(cheek_rgb) / 255.0
        
        lip_max = np.max(lip_rgb)
        lip_min = np.min(lip_rgb)
        lip_saturation = (lip_max - lip_min) / (lip_max + 1e-6)
        
        # Detect abnormal colors
        # Pale cheeks: high lightness, low saturation
        pale_cheeks = cheek_lightness > 0.7 and cheek_saturation < 0.3
        # Red cheeks: high red component
        red_cheeks = cheek_rgb[0] > 180 and cheek_rgb[0] > cheek_rgb[1] * 1.2
        # Pale/blue lips: low red, high blue
        pale_lips = lip_rgb[0] < 120 and lip_rgb[2] > lip_rgb[0] * 1.1
        purple_lips = lip_rgb[2] > 150 and lip_rgb[2] > lip_rgb[0] * 1.3
        
        color_change_detected = pale_cheeks or red_cheeks or pale_lips or purple_lips
        
        return {
            'cheek_color': {
                'r': int(cheek_rgb[0]),
                'g': int(cheek_rgb[1]),
                'b': int(cheek_rgb[2]),
                'saturation': float(cheek_saturation),
                'lightness': float(cheek_lightness)
            },
            'lip_color': {
                'r': int(lip_rgb[0]),
                'g': int(lip_rgb[1]),
                'b': int(lip_rgb[2]),
                'saturation': float(lip_saturation)
            },
            'color_change_detected': color_change_detected,
            'pale_cheeks': pale_cheeks,
            'red_cheeks': red_cheeks,
            'pale_lips': pale_lips,
            'purple_lips': purple_lips,
            'confidence': 0.7 if color_change_detected else 0.0
        }
    
    def detect_facial_tension(self, landmarks) -> Dict:
        """Detect possible discomfort through facial tension indicators"""
        # Analyze eyebrow position and mouth shape
        # Tension indicators: raised eyebrows, tightened mouth
        
        h, w = 1, 1  # Normalized coordinates
        
        # Get eyebrow points
        left_eyebrow_y = []
        right_eyebrow_y = []
        for idx in self.SYMMETRY_POINTS['left_eyebrow']:
            if idx < len(landmarks.landmark):
                left_eyebrow_y.append(landmarks.landmark[idx].y)
        for idx in self.SYMMETRY_POINTS['right_eyebrow']:
            if idx < len(landmarks.landmark):
                right_eyebrow_y.append(landmarks.landmark[idx].y)
        
        # Get mouth corner points
        mouth_left_y = landmarks.landmark[61].y if 61 < len(landmarks.landmark) else None
        mouth_right_y = landmarks.landmark[291].y if 291 < len(landmarks.landmark) else None
        mouth_center_y = landmarks.landmark[13].y if 13 < len(landmarks.landmark) else None
        
        if not left_eyebrow_y or not right_eyebrow_y or mouth_center_y is None:
            return {
                'tension_detected': False,
                'confidence': 0.0
            }
        
        avg_eyebrow_y = (np.mean(left_eyebrow_y) + np.mean(right_eyebrow_y)) / 2.0
        
        # Raised eyebrows (tension indicator)
        eyebrow_raised = avg_eyebrow_y < 0.35  # Lower y = higher position
        
        # Mouth tension (tightened or asymmetric)
        if mouth_left_y and mouth_right_y:
            mouth_asymmetry = abs(mouth_left_y - mouth_right_y)
            mouth_tight = mouth_asymmetry > 0.02  # Significant asymmetry
        else:
            mouth_tight = False
        
        tension_detected = eyebrow_raised or mouth_tight
        confidence = 0.6 if tension_detected else 0.0
        
        return {
            'tension_detected': tension_detected,
            'eyebrow_raised': eyebrow_raised,
            'mouth_tension': mouth_tight,
            'confidence': float(confidence)
        }
    
    def check_facial_symmetry(self, landmarks) -> Dict:
        """Check facial symmetry for stroke symptom screening"""
        h, w = 1, 1  # Normalized coordinates
        
        # Get key symmetry points
        symmetry_measurements = {}
        
        # Nose tip as reference
        nose_tip_x = landmarks.landmark[4].x if 4 < len(landmarks.landmark) else None
        if nose_tip_x is None:
            return {
                'asymmetry_detected': False,
                'symmetry_score': 1.0,
                'confidence': 0.0
            }
        
        # Compare left and right features
        # Eyebrows
        left_eyebrow_x = np.mean([landmarks.landmark[idx].x for idx in self.SYMMETRY_POINTS['left_eyebrow'] 
                                  if idx < len(landmarks.landmark)])
        right_eyebrow_x = np.mean([landmarks.landmark[idx].x for idx in self.SYMMETRY_POINTS['right_eyebrow'] 
                                   if idx < len(landmarks.landmark)])
        
        # Mouth corners
        mouth_left_x = landmarks.landmark[61].x if 61 < len(landmarks.landmark) else None
        mouth_right_x = landmarks.landmark[291].x if 291 < len(landmarks.landmark) else None
        
        # Eye corners
        left_eye_outer_x = landmarks.landmark[33].x if 33 < len(landmarks.landmark) else None
        right_eye_outer_x = landmarks.landmark[263].x if 263 < len(landmarks.landmark) else None
        
        # Calculate symmetry scores
        eyebrow_symmetry = 1.0 - abs((left_eyebrow_x - nose_tip_x) - (nose_tip_x - right_eyebrow_x))
        mouth_symmetry = 1.0 - abs((mouth_left_x - nose_tip_x) - (nose_tip_x - mouth_right_x)) if mouth_left_x and mouth_right_x else 1.0
        eye_symmetry = 1.0 - abs((left_eye_outer_x - nose_tip_x) - (nose_tip_x - right_eye_outer_x)) if left_eye_outer_x and right_eye_outer_x else 1.0
        
        # Overall symmetry score (0-1, higher is more symmetric)
        overall_symmetry = (eyebrow_symmetry + mouth_symmetry + eye_symmetry) / 3.0
        
        # Asymmetry detected if score < 0.85
        asymmetry_detected = overall_symmetry < 0.85
        
        return {
            'asymmetry_detected': asymmetry_detected,
            'symmetry_score': float(overall_symmetry),
            'eyebrow_symmetry': float(eyebrow_symmetry),
            'mouth_symmetry': float(mouth_symmetry),
            'eye_symmetry': float(eye_symmetry),
            'confidence': float(1.0 - overall_symmetry) if asymmetry_detected else 0.0
        }
    
    def analyze_face(self, image: np.ndarray) -> Dict:
        """Main function to analyze all face indicators"""
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return {
                'face_detected': False,
                'error': 'No face detected in image'
            }
        
        landmarks = results.multi_face_landmarks[0]
        image_shape = image.shape
        
        # Perform all analyses
        fatigue = self.detect_fatigue(landmarks)
        skin_color = self.analyze_skin_color(image, landmarks, image_shape)
        tension = self.detect_facial_tension(landmarks)
        symmetry = self.check_facial_symmetry(landmarks)
        
        return {
            'face_detected': True,
            'fatigue': fatigue,
            'skin_color': skin_color,
            'tension': tension,
            'symmetry': symmetry
        }
    
    def reset_blink_counter(self):
        """Reset blink detection counters"""
        self.eye_ar_history = []
        self.blink_count = 0
        self.ear_consecutive_frames = 0

