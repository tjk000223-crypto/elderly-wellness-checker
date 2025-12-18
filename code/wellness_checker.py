"""
Remote Visual Wellness Checker for Elderly
Main application that combines face and tongue analysis
"""

import os
import cv2
import numpy as np
import argparse
import json
from datetime import datetime
from pathlib import Path
from face_analyzer import FaceAnalyzer
from tongue_analyzer import TongueAnalyzer


class WellnessChecker:
    def __init__(self):
        """Initialize the wellness checker with face and tongue analyzers"""
        self.face_analyzer = FaceAnalyzer()
        self.tongue_analyzer = TongueAnalyzer()
    
    def analyze_image(self, image_path: str, save_annotated: bool = True) -> dict:
        """Analyze a single image file"""
        # Read image
        absolute_path = os.path.join(os.getcwd(), 'test',  image_path)
        print(absolute_path)
        # Use imdecode to handle Unicode paths properly
        try:
            with open(absolute_path, 'rb') as f:
                img_bytes = f.read()
            image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            return {
                'error': f'Could not read image: {image_path} - {str(e)}',
                'success': False
            }
        if image is None:
            return {
                'error': f'Could not decode image: {image_path}',
                'success': False
            }

        # Perform analyses
        face_results = self.face_analyzer.analyze_face(image)
        tongue_results = self.tongue_analyzer.analyze_tongue(image)

        # Create annotated image
        annotated_image = self._draw_results(image.copy(), face_results, tongue_results)

        # Save annotated image if requested
        annotated_path = None
        if save_annotated:
            # Create output filename in the same folder as the original file
            original_path_obj = Path(absolute_path)
            annotated_path = original_path_obj.parent / f"{original_path_obj.stem}_annotated{original_path_obj.suffix}"
            # Overwrite if file exists
            cv2.imwrite(str(annotated_path), annotated_image)

        # Combine results
        results = {
            'timestamp': datetime.now().isoformat(),
            'image_path': image_path,
            'annotated_image_path': str(annotated_path) if annotated_path else None,
            'success': True,
            'face_analysis': face_results,
            'tongue_analysis': tongue_results,
            'summary': self._generate_summary(face_results, tongue_results)
        }

        return results
    
    def _adjust_brightness(self, image: np.ndarray, alpha: float = 1.2, beta: int = 20) -> np.ndarray:
        """Adjust image brightness and contrast"""
        # alpha: contrast control (1.0-3.0)
        # beta: brightness control (0-100)
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return adjusted
    
    def analyze_camera(self, duration_seconds: int = 30, output_dir: str = 'results'):
        """Analyze from camera feed for specified duration"""
        cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        '''
        # Set camera properties for better exposure and brightness
        # Try to configure camera settings (may not work on all cameras)
        try:
            # Enable auto-exposure (0.75 = auto mode, 0.25 = manual)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
            # Set exposure (negative values often mean auto)
            cap.set(cv2.CAP_PROP_EXPOSURE, -6)
            # Set brightness (0.0 to 1.0, higher = brighter)
            cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.6)
            # Set contrast
            cap.set(cv2.CAP_PROP_CONTRAST, 0.6)
            # Set gain (sensitivity, 0 = auto)
            cap.set(cv2.CAP_PROP_GAIN, 0)
        except Exception as e:
            print(f"Note: Some camera properties may not be supported: {e}")
        ''''''
        # Allow camera to adjust (warm-up frames)
        print("Adjusting camera settings (this may take a moment)...")
        for i in range(15):  # More warm-up frames for better adjustment
            ret, _ = cap.read()
            if not ret:
                break
            if i % 5 == 0:
                print(f"  Warming up... ({i+1}/15)")
        '''
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        print(f"Starting camera analysis for {duration_seconds} seconds...")
        print("Press 'q' to quit early, 's' to save current frame")
        
        frame_count = 0
        start_time = datetime.now()
        all_results = []
        
        # Reset blink counter for fresh analysis
        self.face_analyzer.reset_blink_counter()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            '''
            # Apply brightness adjustment if frame is too dark
            # Check average brightness
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)
            
            # Adaptive brightness adjustment based on current brightness
            if avg_brightness < 60:
                # Very dark - strong adjustment
                frame = self._adjust_brightness(frame, alpha=1.5, beta=40)
            elif avg_brightness < 80:
                # Moderately dark - moderate adjustment
                frame = self._adjust_brightness(frame, alpha=1.3, beta=25)
            elif avg_brightness < 100:
                # Slightly dark - light adjustment
                frame = self._adjust_brightness(frame, alpha=1.1, beta=15)
            '''
            frame_count += 1
            
            # Analyze every 10 frames (to reduce computation)
            face_results = self.face_analyzer.analyze_face(frame)
            tongue_results = self.tongue_analyzer.analyze_tongue(frame)
                
            # Draw results on frame
            annotated_frame = self._draw_results(frame.copy(), face_results, tongue_results)
                
            # Display
            cv2.imshow('Wellness Checker', annotated_frame)
                
            # Save results periodically
            if frame_count % 30 == 0:  # Every 30 frames (~1 second at 30 FPS)
                result = {
                    'timestamp': datetime.now().isoformat(),
                    'frame': frame_count,
                    'face_analysis': face_results,
                    'tongue_analysis': tongue_results,
                    'summary': self._generate_summary(face_results, tongue_results)
                }
                all_results.append(result)
            
            # Check for exit conditions
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed >= duration_seconds:
                break
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                frame_path = f'{output_dir}/frame_{timestamp}.jpg'
                cv2.imwrite(frame_path, frame)
                print(f"Frame saved to {frame_path}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Save all results
        if all_results:
            results_file = f'{output_dir}/camera_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\nAnalysis complete. Results saved to {results_file}")
            print(f"Total frames analyzed: {len(all_results)}")
        
        return all_results
    
    def _draw_text_with_outline(self, img, text, pos, font, font_scale, color, thickness, outline_color=(0, 0, 0), outline_thickness=3):
        """Draw text with outline for better visibility on any background"""
        x, y = pos
        # Draw outline (black background)
        for dx in range(-outline_thickness, outline_thickness + 1):
            for dy in range(-outline_thickness, outline_thickness + 1):
                if dx != 0 or dy != 0:
                    cv2.putText(img, text, (x + dx, y + dy), font, font_scale, outline_color, outline_thickness + 1)
        # Draw main text
        cv2.putText(img, text, pos, font, font_scale, color, thickness)
    
    def _draw_results(self, frame, face_results, tongue_results):
        """Draw analysis results on the frame with transparent background"""
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Scale font size proportionally to image size (base on width, assuming 640px width for scale 0.6)
        base_width = 640
        font_scale = 0.6 * (w / base_width)
        thickness = max(1, int(2 * (w / base_width)))
        y_offset = int(30 * (h / 480))  # Assuming base height 480
        line_height = int(25 * (h / 480))
        
        y = y_offset
        
        # Face analysis results
        if face_results.get('face_detected'):
            # Fatigue
            fatigue = face_results.get('fatigue', {})
            if fatigue.get('fatigue_detected'):
                text = f"Fatigue: DETECTED (EAR: {fatigue.get('eye_openness', 0):.3f})"
                self._draw_text_with_outline(frame, text, (20, y), font, font_scale, (0, 0, 255), thickness)
            else:
                text = f"Fatigue: Normal (EAR: {fatigue.get('eye_openness', 0):.3f})"
                self._draw_text_with_outline(frame, text, (20, y), font, font_scale, (0, 255, 0), thickness)
            y += line_height
            
            # Skin color
            skin = face_results.get('skin_color', {})
            if skin.get('color_change_detected'):
                changes = []
                if skin.get('pale_cheeks'):
                    changes.append('Pale cheeks')
                if skin.get('red_cheeks'):
                    changes.append('Red cheeks')
                if skin.get('pale_lips'):
                    changes.append('Pale lips')
                if skin.get('purple_lips'):
                    changes.append('Purple lips')
                text = f"Skin: {'; '.join(changes)}"
                self._draw_text_with_outline(frame, text, (20, y), font, font_scale, (0, 0, 255), thickness)
            else:
                text = "Skin: Normal"
                self._draw_text_with_outline(frame, text, (20, y), font, font_scale, (0, 255, 0), thickness)
            y += line_height
            
            # Tension
            tension = face_results.get('tension', {})
            if tension.get('tension_detected'):
                text = "Discomfort: DETECTED"
                self._draw_text_with_outline(frame, text, (20, y), font, font_scale, (0, 0, 255), thickness)
            else:
                text = "Discomfort: Normal"
                self._draw_text_with_outline(frame, text, (20, y), font, font_scale, (0, 255, 0), thickness)
            y += line_height
            
            # Symmetry
            symmetry = face_results.get('symmetry', {})
            if symmetry.get('asymmetry_detected'):
                text = f"Symmetry: ASYMMETRIC (Score: {symmetry.get('symmetry_score', 0):.2f})"
                self._draw_text_with_outline(frame, text, (20, y), font, font_scale, (0, 0, 255), thickness)
            else:
                text = f"Symmetry: Normal (Score: {symmetry.get('symmetry_score', 0):.2f})"
                self._draw_text_with_outline(frame, text, (20, y), font, font_scale, (0, 255, 0), thickness)
            y += line_height
        else:
            text = "Face: Not detected"
            self._draw_text_with_outline(frame, text, (20, y), font, font_scale, (0, 0, 255), thickness)
            y += line_height
        
        y += 10
        
        # Tongue analysis results
        if tongue_results.get('tongue_detected'):
            # Color
            color = tongue_results.get('color', {})
            color_type = color.get('color_type', 'unknown')
            if color.get('color_detected'):
                text = f"Tongue Color: {color_type.upper()}"
                self._draw_text_with_outline(frame, text, (20, y), font, font_scale, (0, 0, 255), thickness)
            else:
                text = f"Tongue Color: {color_type}"
                self._draw_text_with_outline(frame, text, (20, y), font, font_scale, (0, 255, 0), thickness)
            y += line_height
            
            # Coating
            coating = tongue_results.get('coating', {})
            if coating.get('coating_detected'):
                text = f"Coating: {coating.get('thickness', 'unknown').upper()}"
                self._draw_text_with_outline(frame, text, (20, y), font, font_scale, (0, 0, 255), thickness)
            else:
                text = "Coating: None"
                self._draw_text_with_outline(frame, text, (20, y), font, font_scale, (0, 255, 0), thickness)
            y += line_height
            
            # Moisture
            moisture = tongue_results.get('moisture', {})
            moisture_level = moisture.get('moisture_level', 'unknown')
            if moisture.get('dryness_detected'):
                text = f"Moisture: {moisture_level.upper()} (DRY)"
                self._draw_text_with_outline(frame, text, (20, y), font, font_scale, (0, 0, 255), thickness)
            else:
                text = f"Moisture: {moisture_level}"
                self._draw_text_with_outline(frame, text, (20, y), font, font_scale, (0, 255, 0), thickness)
        else:
            text = "Tongue: Not detected"
            self._draw_text_with_outline(frame, text, (20, y), font, font_scale, (0, 0, 255), thickness)
        
        return frame
    
    def _generate_summary(self, face_results, tongue_results) -> dict:
        """Generate a summary of all findings"""
        alerts = []
        warnings = []
        
        # Face analysis alerts
        if face_results.get('face_detected'):
            if face_results.get('fatigue', {}).get('fatigue_detected'):
                alerts.append('Fatigue detected - low eye openness or abnormal blink rate')
            
            if face_results.get('skin_color', {}).get('color_change_detected'):
                skin = face_results.get('skin_color', {})
                changes = []
                if skin.get('pale_cheeks'):
                    changes.append('pale cheeks')
                if skin.get('red_cheeks'):
                    changes.append('red cheeks')
                if skin.get('pale_lips'):
                    changes.append('pale lips')
                if skin.get('purple_lips'):
                    changes.append('purple/blue lips')
                warnings.append(f'Skin color changes: {", ".join(changes)}')
            
            if face_results.get('tension', {}).get('tension_detected'):
                warnings.append('Possible facial tension/discomfort detected')
            
            if face_results.get('symmetry', {}).get('asymmetry_detected'):
                alerts.append('Facial asymmetry detected - possible stroke symptom')
        
        # Tongue analysis alerts
        if tongue_results.get('tongue_detected'):
            color = tongue_results.get('color', {})
            if color.get('color_detected'):
                color_type = color.get('color_type', 'unknown')
                if color_type in ['pale', 'purple']:
                    alerts.append(f'Abnormal tongue color: {color_type}')
                else:
                    warnings.append(f'Tongue color: {color_type}')
            
            coating = tongue_results.get('coating', {})
            if coating.get('coating_detected') and coating.get('thickness') == 'thick':
                warnings.append('Thick tongue coating detected')
            
            moisture = tongue_results.get('moisture', {})
            if moisture.get('dryness_detected'):
                alerts.append('Tongue dryness detected')
        
        overall_status = 'NORMAL' if not alerts else 'ATTENTION NEEDED'
        if alerts:
            overall_status = 'ALERT'
        
        return {
            'overall_status': overall_status,
            'alerts': alerts,
            'warnings': warnings,
            'alert_count': len(alerts),
            'warning_count': len(warnings)
        }


def main():
    parser = argparse.ArgumentParser(
        description='Remote Visual Wellness Checker for Elderly',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single image
  python wellness_checker.py --image photo.jpg
  
  # Analyze from camera for 60 seconds
  python wellness_checker.py --camera --duration 60
  
  # Analyze image and save results
  python wellness_checker.py --image photo.jpg --output results.json
        """
    )
    
    parser.add_argument('--image', type=str, help='Path to image file to analyze')
    parser.add_argument('--camera', action='store_true', help='Use camera for live analysis')
    parser.add_argument('--duration', type=int, default=30, help='Duration for camera analysis in seconds (default: 30)')
    parser.add_argument('--output', type=str, help='Output file path for JSON results')
    parser.add_argument('--no-annotated-image', action='store_true', help='Do not save annotated image (only for --image mode)')
    
    args = parser.parse_args()
    
    checker = WellnessChecker()
    
    if args.image:
        # Analyze single image
        print(f"Analyzing image: {args.image}")
        save_annotated = not args.no_annotated_image
        results = checker.analyze_image(args.image, save_annotated=save_annotated)
        
        if results.get('success'):
            # Print summary
            summary = results.get('summary', {})
            print("\n" + "="*50)
            print("WELLNESS CHECK SUMMARY")
            print("="*50)
            print(f"Overall Status: {summary.get('overall_status', 'UNKNOWN')}")
            print(f"\nAlerts ({summary.get('alert_count', 0)}):")
            for alert in summary.get('alerts', []):
                print(f"  ALERT: {alert}")
            print(f"\nWarnings ({summary.get('warning_count', 0)}):")
            for warning in summary.get('warnings', []):
                print(f"  WARNING: {warning}")
            print("="*50)
            
            # Print annotated image path if saved
            if results.get('annotated_image_path'):
                print(f"\nAnnotated image saved to: {results.get('annotated_image_path')}")
                
                # Display the annotated image
                print("Displaying annotated image (press any key to close)...")
                annotated_image = cv2.imread(results.get('annotated_image_path'))
                if annotated_image is not None:
                    # Resize if too large for display
                    h, w = annotated_image.shape[:2]
                    max_display_size = 1200
                    if w > max_display_size or h > max_display_size:
                        scale = max_display_size / max(w, h)
                        new_w = int(w * scale)
                        new_h = int(h * scale)
                        annotated_image = cv2.resize(annotated_image, (new_w, new_h))
                    
                    cv2.imshow('Wellness Check - Annotated Image', annotated_image)
                    cv2.waitKey(0)  # Wait for any key press
                    cv2.destroyAllWindows()
            
            # Save results if output specified
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"JSON results saved to {args.output}")
        else:
            print(f"Error: {results.get('error', 'Unknown error')}")
    
    elif args.camera:
        # Analyze from camera
        checker.analyze_camera(duration_seconds=args.duration)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

