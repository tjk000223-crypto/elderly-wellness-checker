"""
Tongue Analysis Module for Elderly Wellness Checker
Analyzes tongue color, coating thickness, and moisture/dryness
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Optional, Tuple


class TongueAnalyzer:
    def __init__(self):
        """Initialize MediaPipe face mesh for mouth/tongue region detection"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # Mouth and tongue region indices
        # Inner mouth region (where tongue is visible)
        self.MOUTH_INNER_INDICES = [
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324,  # Lower lip inner
            61, 84, 17, 314, 405, 320, 307, 375, 321, 308  # Upper lip inner
        ]
        
        # Reference points for tongue region estimation
        self.MOUTH_CORNER_LEFT = 61
        self.MOUTH_CORNER_RIGHT = 291
        self.MOUTH_CENTER_TOP = 13
        self.MOUTH_CENTER_BOTTOM = 14
        
    def extract_tongue_region(self, image: np.ndarray, landmarks) -> Optional[np.ndarray]:
        """Extract the tongue region from the mouth area with improved accuracy"""
        h, w = image.shape[:2]
        
        # Method 1: Use inner mouth landmarks
        mouth_points = []
        for idx in self.MOUTH_INNER_INDICES:
            if idx < len(landmarks.landmark):
                x = int(landmarks.landmark[idx].x * w)
                y = int(landmarks.landmark[idx].y * h)
                mouth_points.append([x, y])
        
        if len(mouth_points) < 4:
            return None
        
        mouth_points = np.array(mouth_points, dtype=np.int32)
        
        # Get bounding box
        x, y, w_box, h_box = cv2.boundingRect(mouth_points)
        if w_box < 15 or h_box < 15:
            return None
        
        # Extract ROI with more padding for better detection
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w_box = min(w - x, w_box + 2 * padding)
        h_box = min(h - y, h_box + 2 * padding)
        
        # Extract the full mouth region first
        mouth_roi = image[y:y+h_box, x:x+w_box].copy()
        
        if mouth_roi.size == 0:
            return None
        
        # Create a more refined mask using the mouth shape
        mask = np.zeros((h_box, w_box), dtype=np.uint8)
        # Adjust points relative to ROI
        adjusted_points = mouth_points - [x, y]
        cv2.fillPoly(mask, [adjusted_points], 255)
        
        # Apply mask
        masked_roi = cv2.bitwise_and(mouth_roi, mouth_roi, mask=mask)
        
        # Convert to different color spaces for better detection
        gray = cv2.cvtColor(masked_roi, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(masked_roi, cv2.COLOR_BGR2HSV)
        
        # More lenient threshold - tongue can be various shades
        # Use adaptive thresholding or lower fixed threshold
        # Try multiple thresholds and use the one that gives most valid pixels
        thresholds = [15, 20, 25, 30]
        best_roi = None
        max_valid_pixels = 0
        
        for thresh in thresholds:
            _, mask_tongue = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
            # Also check for skin-like colors in HSV (tongue is usually pink/red)
            # Hue range for pink/red: 0-20 and 160-180
            hsv_mask1 = (hsv[:, :, 0] <= 20) | (hsv[:, :, 0] >= 160)
            hsv_mask2 = hsv[:, :, 1] > 30  # Some saturation
            hsv_mask3 = hsv[:, :, 2] > 30  # Some brightness
            hsv_mask = hsv_mask1 & hsv_mask2 & hsv_mask3
            
            # Combine both masks
            combined_mask = cv2.bitwise_and(mask_tongue, mask.astype(np.uint8))
            combined_mask = cv2.bitwise_and(combined_mask, hsv_mask.astype(np.uint8) * 255)
            
            valid_pixels = np.sum(combined_mask > 0)
            if valid_pixels > max_valid_pixels:
                max_valid_pixels = valid_pixels
                best_roi = cv2.bitwise_and(masked_roi, masked_roi, mask=combined_mask)
        
        # If no good mask found, use the original masked ROI with very lenient threshold
        if best_roi is None or max_valid_pixels < 50:
            # Last resort: use the full masked region with minimal filtering
            _, mask_tongue = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            combined_mask = cv2.bitwise_and(mask_tongue, mask.astype(np.uint8))
            best_roi = cv2.bitwise_and(masked_roi, masked_roi, mask=combined_mask)
        
        # Final validation - check if we have enough valid pixels
        if best_roi is not None:
            valid_pixels = np.sum(np.any(best_roi > [10, 10, 10], axis=2))
            total_pixels = best_roi.shape[0] * best_roi.shape[1]
            if valid_pixels > max(50, total_pixels * 0.05):  # At least 5% valid pixels
                return best_roi
        
        return None
    
    def analyze_tongue_color(self, tongue_region: np.ndarray) -> Dict:
        """Analyze tongue color: pale, red, or purple"""
        if tongue_region is None or tongue_region.size == 0:
            return {
                'color_detected': False,
                'color_type': None,
                'confidence': 0.0
            }
        
        # Convert to RGB
        rgb_region = cv2.cvtColor(tongue_region, cv2.COLOR_BGR2RGB)
        
        # Mask out black/zero pixels
        mask = np.all(rgb_region > [10, 10, 10], axis=2)
        if not np.any(mask):
            return {
                'color_detected': False,
                'color_type': None,
                'confidence': 0.0
            }
        
        valid_pixels = rgb_region[mask]
        
        # Calculate average color
        avg_color = np.mean(valid_pixels, axis=0)
        r, g, b = avg_color
        
        # Calculate color ratios
        total = r + g + b + 1e-6
        r_ratio = r / total
        g_ratio = g / total
        b_ratio = b / total
        
        # Color classification
        # Pale: low saturation, high lightness, blue component > red
        # Red: high red component relative to others
        # Purple: high blue component, moderate red
        
        lightness = np.mean(valid_pixels) / 255.0
        saturation = (np.max(valid_pixels) - np.min(valid_pixels)) / (np.max(valid_pixels) + 1e-6)
        
        color_type = None
        confidence = 0.0
        
        # Pale tongue: low saturation, high lightness, or blue > red
        if (saturation < 0.3 and lightness > 0.6) or (b_ratio > r_ratio * 1.2 and b > 100):
            color_type = 'pale'
            confidence = min(1.0 - saturation, (b_ratio - r_ratio) * 2)
        
        # Red tongue: high red component
        elif r_ratio > 0.4 and r > 150 and r > g * 1.2:
            color_type = 'red'
            confidence = min((r_ratio - 0.35) * 3, (r - 120) / 100)
        
        # Purple tongue: high blue, moderate red
        elif b_ratio > 0.35 and b > 120 and r > 80 and b > r * 1.1:
            color_type = 'purple'
            confidence = min((b_ratio - 0.3) * 2, (b - r) / 80)
        
        # Normal (pinkish)
        else:
            color_type = 'normal'
            confidence = 0.3
        
        return {
            'color_detected': color_type != 'normal',
            'color_type': color_type,
            'rgb': {
                'r': int(r),
                'g': int(g),
                'b': int(b)
            },
            'lightness': float(lightness),
            'saturation': float(saturation),
            'confidence': float(confidence)
        }
    
    def analyze_coating_thickness(self, tongue_region: np.ndarray) -> Dict:
        """Analyze tongue coating thickness"""
        if tongue_region is None or tongue_region.size == 0:
            return {
                'coating_detected': False,
                'thickness': None,
                'confidence': 0.0
            }
        
        # Convert to grayscale
        gray = cv2.cvtColor(tongue_region, cv2.COLOR_BGR2GRAY)
        
        # Mask out black pixels
        mask = gray > 30
        if not np.any(mask):
            return {
                'coating_detected': False,
                'thickness': None,
                'confidence': 0.0
            }
        
        valid_gray = gray[mask]
        
        # Coating appears as white/yellowish layer
        # Analyze texture and color variation
        
        # Calculate standard deviation (coating creates texture variation)
        std_dev = np.std(valid_gray)
        
        # Calculate mean brightness (coating is usually lighter)
        mean_brightness = np.mean(valid_gray)
        
        # Use edge detection to identify coating boundaries
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.size + 1e-6)
        
        # Analyze color channels for white/yellow coating
        rgb_region = cv2.cvtColor(tongue_region, cv2.COLOR_BGR2RGB)
        valid_rgb = rgb_region[mask]
        
        # Coating typically has:
        # - High brightness
        # - Low saturation (white/light yellow)
        # - Texture variation
        
        avg_rgb = np.mean(valid_rgb, axis=0)
        max_channel = np.max(avg_rgb)
        min_channel = np.min(avg_rgb)
        saturation = (max_channel - min_channel) / (max_channel + 1e-6)
        
        # Thick coating indicators:
        # - High brightness (> 180)
        # - Low saturation (< 0.3) - white/light
        # - High texture variation (std_dev > 25)
        # - Moderate edge density
        
        coating_score = 0.0
        if mean_brightness > 180:
            coating_score += 0.3
        if saturation < 0.3:
            coating_score += 0.3
        if std_dev > 25:
            coating_score += 0.2
        if 0.05 < edge_density < 0.15:
            coating_score += 0.2
        
        coating_detected = coating_score > 0.5
        thickness = 'thick' if coating_score > 0.7 else 'moderate' if coating_score > 0.5 else 'thin'
        
        return {
            'coating_detected': coating_detected,
            'thickness': thickness if coating_detected else 'none',
            'coating_score': float(coating_score),
            'mean_brightness': float(mean_brightness),
            'saturation': float(saturation),
            'texture_variation': float(std_dev),
            'confidence': float(coating_score)
        }
    
    def analyze_moisture(self, tongue_region: np.ndarray) -> Dict:
        """Analyze tongue moisture vs dryness via glossiness"""
        if tongue_region is None or tongue_region.size == 0:
            return {
                'moisture_level': None,
                'dryness_detected': False,
                'confidence': 0.0
            }
        
        # Moisture is indicated by glossiness (specular reflections)
        # Dry tongue: matte, low gloss
        # Moist tongue: glossy, high specular highlights
        
        # Convert to grayscale
        gray = cv2.cvtColor(tongue_region, cv2.COLOR_BGR2GRAY)
        
        # Mask out black pixels
        mask = gray > 30
        if not np.any(mask):
            return {
                'moisture_level': None,
                'dryness_detected': False,
                'confidence': 0.0
            }
        
        valid_gray = gray[mask]
        
        # Glossiness indicators:
        # 1. High brightness pixels (specular highlights)
        # 2. High contrast (bright highlights vs darker areas)
        # 3. Smooth texture (less variation in glossy areas)
        
        # Calculate brightness distribution
        brightness_hist = np.histogram(valid_gray, bins=50, range=(0, 255))[0]
        high_brightness_ratio = np.sum(valid_gray > 200) / len(valid_gray)
        
        # Calculate contrast (standard deviation)
        contrast = np.std(valid_gray)
        
        # Calculate smoothness (inverse of local variation)
        # Use Laplacian to detect smooth vs textured regions
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = np.var(laplacian[mask])
        smoothness = 1.0 / (laplacian_var + 1e-6)
        
        # Moisture score
        # High brightness ratio = glossy = moist
        # High contrast = glossy highlights = moist
        # High smoothness = glossy surface = moist
        
        moisture_score = 0.0
        if high_brightness_ratio > 0.15:  # Many bright pixels
            moisture_score += 0.4
        if contrast > 40:  # High contrast (highlights)
            moisture_score += 0.3
        if smoothness > 0.01:  # Smooth surface
            moisture_score += 0.3
        
        # Classify moisture level
        if moisture_score > 0.6:
            moisture_level = 'moist'
            dryness_detected = False
        elif moisture_score < 0.3:
            moisture_level = 'dry'
            dryness_detected = True
        else:
            moisture_level = 'moderate'
            dryness_detected = False
        
        return {
            'moisture_level': moisture_level,
            'dryness_detected': dryness_detected,
            'moisture_score': float(moisture_score),
            'high_brightness_ratio': float(high_brightness_ratio),
            'contrast': float(contrast),
            'smoothness': float(smoothness),
            'confidence': float(abs(moisture_score - 0.5) * 2)  # Higher confidence for extreme values
        }
    
    def analyze_tongue(self, image: np.ndarray) -> Dict:
        """Main function to analyze all tongue indicators"""
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return {
                'tongue_detected': False,
                'error': 'No face detected in image'
            }
        
        landmarks = results.multi_face_landmarks[0]
        
        # Extract tongue region
        tongue_region = self.extract_tongue_region(image, landmarks)
        
        if tongue_region is None:
            return {
                'tongue_detected': False,
                'error': 'Could not extract tongue region'
            }
        
        # Perform all analyses
        color_analysis = self.analyze_tongue_color(tongue_region)
        coating_analysis = self.analyze_coating_thickness(tongue_region)
        moisture_analysis = self.analyze_moisture(tongue_region)
        
        return {
            'tongue_detected': True,
            'color': color_analysis,
            'coating': coating_analysis,
            'moisture': moisture_analysis,
            'tongue_region_size': tongue_region.shape[:2]
        }

