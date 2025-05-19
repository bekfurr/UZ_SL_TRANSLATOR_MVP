import json
import base64
import cv2
import numpy as np
import pickle
import mediapipe as mp
import logging
import threading
import time
import os
from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import sync_to_async
from django.contrib.auth.models import User
from .models import TrainedModel, TranslationSession
from django.conf import settings

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class TranslatorConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.session_id = self.scope['url_route']['kwargs']['session_id']
        self.session_group_name = f'translator_{self.session_id}'
        
        # Join session group
        await self.channel_layer.group_add(
            self.session_group_name,
            self.channel_name
        )
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.3)
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize model variables
        self.model = None
        self.label_mapping = {}
        self.inverse_label_mapping = {}
        self.last_prediction_time = time.time()
        self.prediction_interval = 3.0  # seconds
        self.landmarks_history = []
        self.text_output = ""
        
        logging.debug(f"WebSocket connection established for session {self.session_id}")
        await self.accept()
    
    async def disconnect(self, close_code):
        logging.debug(f"WebSocket disconnected with code {close_code}")
        # Leave session group
        await self.channel_layer.group_discard(
            self.session_group_name,
            self.channel_name
        )
        
        # Clean up resources
        if hasattr(self, 'hands'):
            self.hands.close()
        if hasattr(self, 'pose'):
            self.pose.close()
    
    async def receive(self, text_data=None, bytes_data=None):
        if text_data:
            logging.debug(f"Received text data: {text_data[:100]}...")
            text_data_json = json.loads(text_data)
            message_type = text_data_json.get('type')
            
            if message_type == 'load_model':
                model_id = text_data_json.get('model_id')
                success = await self.load_model(model_id)
                await self.send(text_data=json.dumps({
                    'type': 'model_loaded',
                    'success': success
                }))
                logging.debug(f"Model {model_id} loaded: {success}")
            
            elif message_type == 'set_interval':
                self.prediction_interval = float(text_data_json.get('interval', 3.0))
                await self.send(text_data=json.dumps({
                    'type': 'interval_set',
                    'interval': self.prediction_interval
                }))
                logging.debug(f"Interval set to {self.prediction_interval}")
            
            elif message_type == 'clear_output':
                self.text_output = ""
                await self.send(text_data=json.dumps({
                    'type': 'output_cleared'
                }))
                logging.debug("Output cleared")
        
        elif bytes_data:
            # Process frame data
            if self.model is None:
                await self.send(text_data=json.dumps({
                    'type': 'error',
                    'message': 'Model not loaded'
                }))
                return
            
            # Convert bytes to numpy array
            frame = await self.bytes_to_frame(bytes_data)
            if frame is None:
                return
            
            # Extract landmarks
            landmarks, frame_with_skeleton, _, hands_detected = await self.extract_landmarks(frame)
            
            # Send processed frame back
            processed_frame_bytes = await self.frame_to_bytes(frame_with_skeleton)
            await self.send(bytes_data=processed_frame_bytes)
            
            # Check if it's time to make a prediction
            current_time = time.time()
            if hands_detected and landmarks and current_time - self.last_prediction_time >= self.prediction_interval:
                # Make prediction
                prediction = await self.predict(landmarks)
                if prediction:
                    self.text_output += prediction + " "
                    self.last_prediction_time = current_time
                    
                    # Send prediction
                    await self.send(text_data=json.dumps({
                        'type': 'prediction',
                        'word': prediction,
                        'full_text': self.text_output
                    }))
                    logging.debug(f"Prediction: {prediction}")
    
    @sync_to_async
    def load_model(self, model_id):
        try:
            model_obj = TrainedModel.objects.get(id=model_id)
            model_path = os.path.join(settings.MEDIA_ROOT, model_obj.file.name)
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.model = model_data['model']
                self.label_mapping = model_data.get('label_mapping', {})
                self.inverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
            
            return True
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            return False
    
    @sync_to_async
    def bytes_to_frame(self, bytes_data):
        try:
            # Decode image
            img_array = np.frombuffer(bytes_data, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            logging.error(f"Error converting bytes to frame: {str(e)}")
            return None
    
    @sync_to_async
    def frame_to_bytes(self, frame):
        try:
            # Encode frame to bytes
            _, buffer = cv2.imencode('.jpg', frame)
            return buffer.tobytes()
        except Exception as e:
            logging.error(f"Error converting frame to bytes: {str(e)}")
            return b''
    
    @sync_to_async
    def extract_landmarks(self, frame, draw_skeleton=True):
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape
            frame_rgb = cv2.convertScaleAbs(frame_rgb, alpha=1.5, beta=15)

            hand_results = self.hands.process(frame_rgb)
            pose_results = self.pose.process(frame_rgb)
            landmarks_list = []
            frame_with_skeleton = frame.copy()
            bounding_boxes = []
            hands_detected = False

            data_aux = []
            if hand_results.multi_hand_landmarks:
                hands_detected = True
                for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                    if draw_skeleton:
                        self.mp_drawing.draw_landmarks(
                            frame_with_skeleton,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS
                        )
                    x_coords = [landmark.x for landmark in hand_landmarks.landmark]
                    y_coords = [landmark.y for landmark in hand_landmarks.landmark]
                    x_min = int(min(x_coords) * w) - 20
                    x_max = int(max(x_coords) * w) + 20
                    y_min = int(min(y_coords) * h) - 20
                    y_max = int(max(y_coords) * h) + 20
                    bounding_boxes.append((x_min, y_min, x_max, y_max))
                    if draw_skeleton:
                        cv2.rectangle(frame_with_skeleton, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    x_ = [landmark.x for landmark in hand_landmarks.landmark]
                    y_ = [landmark.y for landmark in hand_landmarks.landmark]
                    if x_ and y_:
                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y
                            data_aux.append(x - min(x_))
                            data_aux.append(y - min(y_))

            if pose_results.pose_landmarks:
                pose_landmarks = pose_results.pose_landmarks.landmark
                left_elbow = pose_landmarks[13]
                data_aux.append(left_elbow.x * w)
                data_aux.append(left_elbow.y * h)
                right_elbow = pose_landmarks[14]
                data_aux.append(right_elbow.x * w)
                data_aux.append(right_elbow.y * h)
                if draw_skeleton:
                    self.mp_drawing.draw_landmarks(
                        frame_with_skeleton,
                        pose_results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS
                    )

            if len(data_aux) < 88:
                data_aux.extend([0.0] * (88 - len(data_aux)))

            landmarks_list.append(data_aux)
            return landmarks_list, frame_with_skeleton, bounding_boxes, hands_detected
        except Exception as e:
            logging.error(f"Error extracting landmarks: {e}")
            return [], frame, [], False
    
    @sync_to_async
    def predict(self, landmarks):
        try:
            if not landmarks or not self.model:
                return None
            
            # Store landmarks for motion detection
            self.landmarks_history.append(landmarks[0])
            if len(self.landmarks_history) > 30:  # Keep only last 30 frames
                self.landmarks_history.pop(0)
            
            # Use average of recent landmarks for prediction
            avg_features = np.mean(self.landmarks_history, axis=0)
            if len(avg_features) == 88:  # Expected feature length
                prediction = self.model.predict([avg_features])
                predicted_idx = prediction[0]
                predicted_word = self.inverse_label_mapping.get(predicted_idx, "Unknown")
                return predicted_word
            
            return None
        except Exception as e:
            logging.error(f"Error making prediction: {str(e)}")
            return None
