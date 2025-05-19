import os
import cv2
import json
import pickle
import numpy as np
import mediapipe as mp
import logging
import uuid
import threading
import base64
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from .forms import UserRegistrationForm, VideoUploadForm, ModelUploadForm, DataProcessorForm, ModelTrainerForm
from .models import SignVideo, TrainedModel, TranslationSession

# Logging setup
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# MediaPipe initialization
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def home(request):
    return render(request, 'translator/home.html')

def register(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'Account created for {username}! You can now log in.')
            return redirect('login')
    else:
        form = UserRegistrationForm()
    return render(request, 'translator/register.html', {'form': form})

@login_required
def dashboard(request):
    user_models = TrainedModel.objects.filter(created_by=request.user)
    user_videos = SignVideo.objects.filter(uploaded_by=request.user)
    user_sessions = TranslationSession.objects.filter(user=request.user).order_by('-start_time')[:5]
    
    context = {
        'user_models': user_models,
        'user_videos': user_videos,
        'user_sessions': user_sessions,
    }
    return render(request, 'translator/dashboard.html', context)

@login_required
def data_processor(request):
    videos = SignVideo.objects.filter(uploaded_by=request.user)
    return render(request, 'translator/data_processor.html', {'videos': videos})

@login_required
def model_trainer(request):
    models = TrainedModel.objects.filter(created_by=request.user)
    return render(request, 'translator/model_trainer.html', {'models': models})

@login_required
def realtime_translator(request):
    models = TrainedModel.objects.filter(created_by=request.user)
    return render(request, 'translator/realtime_translator.html', {'models': models})

@login_required
def upload_video(request):
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video = form.save(commit=False)
            video.uploaded_by = request.user
            video.save()
            messages.success(request, 'Video uploaded successfully!')
            return redirect('data_processor')
    else:
        form = VideoUploadForm()
    return render(request, 'translator/upload_video.html', {'form': form})

@login_required
def upload_model(request):
    if request.method == 'POST':
        form = ModelUploadForm(request.POST, request.FILES)
        if form.is_valid():
            model = form.save(commit=False)
            model.created_by = request.user
            model.save()
            messages.success(request, 'Model uploaded successfully!')
            return redirect('model_trainer')
    else:
        form = ModelUploadForm()
    return render(request, 'translator/upload_model.html', {'form': form})

@login_required
def process_data(request):
    # Get all videos from the user
    videos = SignVideo.objects.filter(uploaded_by=request.user)
    
    if request.method == 'POST':
        # Create temporary directory for processing
        temp_dir = os.path.join(settings.MEDIA_ROOT, f'temp_{uuid.uuid4().hex}')
        os.makedirs(temp_dir, exist_ok=True)
        
        if not videos:
            messages.error(request, 'You need to upload videos first before processing data.')
            return redirect('upload_video')
        
        # Create words.json file
        words_data = []
        for video in videos:
            video_path = os.path.join(settings.MEDIA_ROOT, video.video.name)
            dest_path = os.path.join(temp_dir, os.path.basename(video.video.name))
            
            # Copy video file to temp directory
            with open(video_path, 'rb') as src, open(dest_path, 'wb') as dst:
                dst.write(src.read())
            
            words_data.append({
                "word_uz": video.word,
                "video": os.path.basename(video.video.name)
            })
        
        # Save words.json
        with open(os.path.join(temp_dir, 'words.json'), 'w', encoding='utf-8') as f:
            json.dump(words_data, f, ensure_ascii=False, indent=2)
        
        # Process data in background
        threading.Thread(target=process_data_background, args=(temp_dir, request.user.id)).start()
        
        messages.success(request, 'Data processing started. This may take a few minutes.')
        return redirect('data_processor')
    
    return render(request, 'translator/process_data.html', {'videos': videos})

def process_data_background(temp_dir, user_id):
    try:
        # Initialize MediaPipe
        hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, max_num_hands=2)
        pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.3)
        
        # Load words.json
        with open(os.path.join(temp_dir, 'words.json'), 'r', encoding='utf-8') as f:
            words_data = json.load(f)
        
        data = []
        labels = []
        class_names = []
        
        for item in words_data:
            word = item["word_uz"]
            video_path = os.path.join(temp_dir, item["video"])
            if not os.path.exists(video_path):
                logging.warning(f"Video file not found: {video_path}")
                continue
            
            logging.info(f"Processing video: {video_path} for class: {word}")
            start_frame, end_frame, landmarks_history = detect_hand_and_elbow_movement(video_path, hands, pose)
            
            if landmarks_history:
                frame_features = landmarks_history[start_frame:end_frame + 1] if start_frame is not None and end_frame is not None else landmarks_history
                if frame_features:
                    expected_length = len(frame_features[0])
                    frame_features = [f for f in frame_features if len(f) == expected_length]
                    if len(frame_features) == 0:
                        logging.warning(f"No consistent features extracted from video: {video_path}, using default zero features.")
                        frame_features = [np.zeros(88).tolist()]
                    avg_features = np.mean(frame_features, axis=0)
                    data.append(avg_features)
                    labels.append(word)
                    class_names.append(word)
                    logging.info(f"Successfully processed video: {video_path}, class: {word}")
                else:
                    logging.warning(f"No valid features extracted from video: {video_path}, using default zero features.")
                    data.append(np.zeros(88).tolist())
                    labels.append(word)
                    class_names.append(word)
                    logging.info(f"Added default features for video: {video_path}, class: {word}, label: {word}")
        
        # Save processed data
        pickle_path = os.path.join(settings.MEDIA_ROOT, 'data', f'data_mixed_{uuid.uuid4().hex}.pickle')
        os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
        
        with open(pickle_path, 'wb') as f:
            pickle.dump({'data': data, 'labels': labels, 'class_names': class_names}, f)
        
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)
        
        logging.info(f"Data processing completed. Saved to {pickle_path}")
    except Exception as e:
        logging.error(f"Error in data processing: {str(e)}")

def detect_hand_and_elbow_movement(video_path, hands, pose):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FPS, 60)
    landmarks_history = []
    motion_detected = False
    start_frame = None
    end_frame = None
    expected_length = None
    min_frames = 30
    prev_landmarks = None
    smoothing_factor = 0.7

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        frame_rgb = cv2.convertScaleAbs(frame_rgb, alpha=1.5, beta=15)

        hand_results = hands.process(frame_rgb)
        pose_results = pose.process(frame_rgb)

        data_aux = []
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
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

        if len(data_aux) < 88:
            data_aux.extend([0.0] * (88 - len(data_aux)))

        if expected_length is None and data_aux:
            expected_length = len(data_aux)
        if len(data_aux) == expected_length:
            if prev_landmarks is not None:
                smoothed_landmarks = smoothing_factor * prev_landmarks + (1 - smoothing_factor) * np.array(data_aux)
                data_aux = smoothed_landmarks.tolist()
            landmarks_history.append(data_aux)
            if len(landmarks_history) > 1 and len(landmarks_history) >= min_frames:
                prev_data = np.array(landmarks_history[-2])
                curr_data = np.array(data_aux)
                if len(curr_data) == len(prev_data):
                    diff = np.linalg.norm(curr_data - prev_data)
                    if diff > 0.01:
                        if not motion_detected:
                            start_frame = len(landmarks_history) - 1
                            motion_detected = True
                    elif motion_detected and diff < 0.002:
                        end_frame = len(landmarks_history) - 1
                        break
            prev_landmarks = np.array(data_aux)
        else:
            if motion_detected and end_frame is None and len(landmarks_history) >= min_frames:
                end_frame = len(landmarks_history) - 1
                break

    cap.release()
    if not motion_detected:
        if landmarks_history:
            start_frame = 0
            end_frame = len(landmarks_history) - 1
        else:
            landmarks_history = [np.zeros(88).tolist()]
            start_frame = 0
            end_frame = 0
            logging.warning(f"No landmarks detected in video: {video_path}, using default zero features.")

    return start_frame, end_frame, landmarks_history

@login_required
def train_model(request):
    if request.method == 'POST':
        form = ModelTrainerForm(request.POST, request.FILES)
        if form.is_valid():
            pickle_file = request.FILES['pickle_file']
            fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'data'))
            filename = fs.save(pickle_file.name, pickle_file)
            pickle_path = os.path.join(settings.MEDIA_ROOT, 'data', filename)
            
            # Train model in background
            threading.Thread(target=train_model_background, args=(pickle_path, request.user.id)).start()
            
            messages.success(request, 'Model training started. This may take a few minutes.')
            return redirect('model_trainer')
    else:
        form = ModelTrainerForm()
    
    return render(request, 'translator/train_model.html', {'form': form})

def train_model_background(pickle_path, user_id):
    try:
        from django.contrib.auth.models import User
        user = User.objects.get(id=user_id)
        
        # Load data
        data_dict = pickle.load(open(pickle_path, 'rb'))
        data = data_dict['data']
        labels = data_dict['labels']
        
        if not data:
            logging.error("No data found in the pickle file!")
            return
        
        unique_classes = len(set(labels))
        if unique_classes < 2:
            logging.warning(f"Only {unique_classes} class found! Training with limited data might not be effective.")
            if unique_classes == 0:
                logging.error("No classes found!")
                return
        
        feature_lengths = [len(d) for d in data]
        if not feature_lengths:
            logging.error("No valid feature lengths found in data!")
            return
        most_common_length = max(set(feature_lengths), key=feature_lengths.count)
        logging.info(f"Most common feature length: {most_common_length}")
        
        filtered_data = [d for d in data if len(d) == most_common_length]
        filtered_labels = [labels[i] for i, d in enumerate(data) if len(d) == most_common_length]
        
        if not filtered_data:
            logging.error("No valid data for training after filtering!")
            return
        
        data = np.asarray(filtered_data)
        labels = np.asarray(filtered_labels)
        
        le = LabelEncoder()
        labels_encoded = le.fit_transform(labels)
        label_mapping = dict(zip(le.classes_, range(len(le.classes_))))
        
        x_train, x_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.1, shuffle=True)
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)
        score = accuracy_score(y_predict, y_test)
        logging.info(f'Hand + Elbow: {score * 100:.2f}% of samples classified correctly!')
        
        # Save model
        model_path = os.path.join(settings.MEDIA_ROOT, 'models', f'model_mixed_{uuid.uuid4().hex}.p')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump({'model': model, 'label_mapping': label_mapping}, f)
        
        # Create model record
        model_name = f"Model {uuid.uuid4().hex[:8]}"
        model_file = os.path.relpath(model_path, settings.MEDIA_ROOT)
        
        TrainedModel.objects.create(
            name=model_name,
            description=f"Trained with {len(data)} samples, {unique_classes} classes",
            file=model_file,
            created_by=user,
            accuracy=score * 100
        )
        
        logging.info(f"Model training completed. Saved to {model_path}")
    except Exception as e:
        logging.error(f"Error in model training: {str(e)}")

@login_required
def translate_video(request):
    if request.method == 'POST':
        video_file = request.FILES.get('video')
        model_id = request.POST.get('model_id')
        
        if not video_file or not model_id:
            return JsonResponse({'error': 'Please provide both video and model.'}, status=400)
        
        # Save video temporarily
        fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'temp'))
        filename = fs.save(video_file.name, video_file)
        video_path = os.path.join(settings.MEDIA_ROOT, 'temp', filename)
        
        # Get model
        try:
            model_obj = TrainedModel.objects.get(id=model_id)
            model_path = os.path.join(settings.MEDIA_ROOT, model_obj.file.name)
            
            # Process video
            result = translate_video_background(video_path, model_path)
            
            # Clean up
            if os.path.exists(video_path):
                os.remove(video_path)
            
            return JsonResponse({'translation': result})
        except TrainedModel.DoesNotExist:
            return JsonResponse({'error': 'Model not found.'}, status=404)
        except Exception as e:
            logging.error(f"Error in translate_video: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)
    
    models = TrainedModel.objects.filter(created_by=request.user)
    return render(request, 'translator/translate_video.html', {'models': models})

def translate_video_background(video_path, model_path):
    try:
        # Load model
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            model = model_data['model']
            label_mapping = model_data.get('label_mapping', {})
            inverse_label_mapping = {v: k for k, v in label_mapping.items()}
        
        # Initialize MediaPipe
        hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, max_num_hands=2)
        pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.3)
        
        # Process video
        start_frame, end_frame, landmarks_history = detect_hand_and_elbow_movement(video_path, hands, pose)
        
        if start_frame < len(landmarks_history) and (end_frame is None or start_frame < end_frame):
            if end_frame is None:
                end_frame = len(landmarks_history) - 1
            frame_features = landmarks_history[start_frame:end_frame + 1]
            expected_length = len(frame_features[0])
            frame_features = [f for f in frame_features if len(f) == expected_length]
            if not frame_features:
                return "No consistent data detected"
            avg_features = np.mean(frame_features, axis=0)
            if model and len(avg_features) == 88:
                prediction = model.predict([avg_features])
                predicted_idx = prediction[0]
                predicted_word = inverse_label_mapping.get(predicted_idx, "Unknown")
                return predicted_word
            else:
                return "Model not loaded or incorrect feature length"
        else:
            return "No valid data detected"
    except Exception as e:
        logging.error(f"Error in video translation: {str(e)}")
        return f"Error: {str(e)}"

@csrf_exempt
def translate_frame(request):
    if request.method == 'POST':
        try:
            # Get frame data
            frame_data = request.FILES.get('frame')
            model_id = request.POST.get('model_id')
            
            if not frame_data or not model_id:
                return JsonResponse({'error': 'Missing frame data or model ID'}, status=400)
            
            # Read frame
            frame = cv2.imdecode(np.frombuffer(frame_data.read(), np.uint8), cv2.IMREAD_COLOR)
            
            # Get model
            model_obj = get_object_or_404(TrainedModel, id=model_id)
            model_path = os.path.join(settings.MEDIA_ROOT, model_obj.file.name)
            
            # Load model
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                model = model_data['model']
                label_mapping = model_data.get('label_mapping', {})
                inverse_label_mapping = {v: k for k, v in label_mapping.items()}
            
            # Initialize MediaPipe with lower detection confidence
            hands = mp_hands.Hands(
                static_image_mode=True,  # Set to True for better accuracy with still images
                max_num_hands=2,
                min_detection_confidence=0.2  # Lower threshold for better detection
            )
            pose = mp_pose.Pose(
                static_image_mode=True,  # Set to True for better accuracy with still images
                min_detection_confidence=0.2  # Lower threshold for better detection
            )
            
            # Extract landmarks
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape
            
            # Enhance image for better detection
            frame_rgb = cv2.convertScaleAbs(frame_rgb, alpha=1.5, beta=15)
            
            hand_results = hands.process(frame_rgb)
            pose_results = pose.process(frame_rgb)
            
            # Draw landmarks on frame
            frame_with_skeleton = frame.copy()
            
            # Add status text to frame
            cv2.putText(
                frame_with_skeleton,
                "Status: Processing",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )
            
            # Draw hand landmarks if detected
            hands_detected = False
            if hand_results.multi_hand_landmarks:
                hands_detected = True
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame_with_skeleton,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
            
            # Draw pose landmarks if detected
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame_with_skeleton,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )
            
            # Extract features
            data_aux = []
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
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
            
            if len(data_aux) < 88:
                data_aux.extend([0.0] * (88 - len(data_aux)))
            
            # Make prediction
            if model and len(data_aux) == 88 and hands_detected:
                prediction = model.predict([data_aux])
                predicted_idx = prediction[0]
                predicted_word = inverse_label_mapping.get(predicted_idx, "Unknown")
                
                # Add prediction text to frame
                cv2.putText(
                    frame_with_skeleton,
                    f"Detected: {predicted_word}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )
                
                # Update status
                cv2.putText(
                    frame_with_skeleton,
                    "Status: Hand Detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )
                
                # Convert frame to base64 for response
                _, buffer = cv2.imencode('.jpg', frame_with_skeleton)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                return JsonResponse({
                    'word': predicted_word,
                    'frame': f"data:image/jpeg;base64,{frame_base64}"
                })
            else:
                # No hands detected or other issue
                if not hands_detected:
                    cv2.putText(
                        frame_with_skeleton,
                        "Status: No Hand Detected",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA
                    )
                
                # Convert frame to base64 for response
                _, buffer = cv2.imencode('.jpg', frame_with_skeleton)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                return JsonResponse({
                    'word': None,
                    'frame': f"data:image/jpeg;base64,{frame_base64}"
                }) 
        except Exception as e:
            logging.error(f"Error in frame translation: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Only POST method is allowed'}, status=405)
