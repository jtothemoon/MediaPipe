from flask import Flask, request, jsonify, send_from_directory
import os
import cv2
import mediapipe as mp
import ffmpeg
import math
import pymysql.cursors
from datetime import date

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/'
PROCESSED_FOLDER = 'processed/'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

# MySQL 데이터베이스 연결 설정
connection = pymysql.connect(
    host='project-db-cgi.smhrd.com',
    port=3307,
    user='homewalk',
    password='homewalk',
    database='homewalk',
    cursorclass=pymysql.cursors.DictCursor
)

# Mediapipe 포즈 추적 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(point1, point2, point3):
    """Calculate the angle between three points."""
    x1, y1 = point1
    x2, y2 = point2
    x3, y3 = point3

    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    return abs(angle)

def save_score_to_db(video_id, user_id, score):
    """일별로 사용자 ID와 점수를 데이터베이스에 저장 또는 업데이트하는 함수"""
    with connection.cursor() as cursor:
        today = date.today()  # 오늘 날짜를 가져옴
        sql = """
        INSERT INTO user_video_scores (video_id, user_id, score, record_date)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            score = VALUES(score),
            updated_at = CURRENT_TIMESTAMP;
        """
        cursor.execute(sql, (video_id, user_id, score, today))
        connection.commit()

def process_video_with_pose_estimation(input_file, output_file, userId):
    cap = cv2.VideoCapture(input_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    total_frames = 0
    incorrect_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        frame_is_incorrect = False  # 현재 프레임이 잘못되었는지 추적

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # 1. 무릎 각도 계산 (왼쪽 무릎 예시)
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * width,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * height]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * width,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * height]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * width,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * height]

            left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            knee_color = (0, 255, 0) if 160 <= left_knee_angle <= 180 else (0, 0, 255)

            # 2. 허리와 다리의 정렬 확인 (왼쪽 다리)
            leg_straightness_color = (0, 255, 0)  # 초록색으로 초기화

            deviation = abs(left_hip[0] - left_knee[0]) + abs(left_knee[0] - left_ankle[0])
            if deviation > width * 0.05:  # 허용 오차를 화면 너비의 5%로 설정
                leg_straightness_color = (0, 0, 255)  # 빨간색으로 표시

            # 포즈 랜드마크 그리기
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # 색상 표시
            cv2.putText(frame, f'Knee Angle: {int(left_knee_angle)}', tuple(int(v) for v in left_knee),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, knee_color, 2, cv2.LINE_AA)
            cv2.putText(frame, 'Leg Straightness', tuple(int(v) for v in left_hip),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, leg_straightness_color, 2, cv2.LINE_AA)

            # 만약 빨간색으로 표시된다면, 프레임을 잘못된 것으로 간주
            if knee_color == (0, 0, 255) or leg_straightness_color == (0, 0, 255):
                frame_is_incorrect = True

        if frame_is_incorrect:
            incorrect_frames += 1

        # 사용자 ID 및 현재 점수를 비디오의 오른쪽 상단에 표시
        cv2.putText(frame, f'User: {userId}', (width - 200, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # 현재 점수를 표시
        incorrect_percentage = (incorrect_frames / total_frames) * 100 if total_frames > 0 else 0
        score = max(100 - int(incorrect_percentage), 0)  # 점수는 0 이상 100 이하로 제한
        cv2.putText(frame, f'Score: {score}', (width - 200, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        out.write(frame)

    cap.release()
    out.release()

    # 계산된 점수를 반환합니다.
    return score

def get_unique_filename(directory, filename):
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename

    while os.path.exists(os.path.join(directory, new_filename)):
        new_filename = f"{base}_{counter}{ext}"
        counter += 1

    return new_filename

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    userId = request.form.get('userid', 'Unknown')  # 사용자 ID를 폼에서 가져옴

    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    filename = get_unique_filename(UPLOAD_FOLDER, file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    temp_filename = get_unique_filename(PROCESSED_FOLDER, "temp_" + filename)
    temp_file_path = os.path.join(PROCESSED_FOLDER, temp_filename)

    # 포즈 추출 및 인식 처리, 점수를 반환합니다.
    score = process_video_with_pose_estimation(file_path, temp_file_path, userId)

    # 점수를 DB에 저장 (일별로)
    save_score_to_db(filename, userId, score)

    processed_filename = get_unique_filename(PROCESSED_FOLDER, filename)
    processed_file_path = os.path.join(PROCESSED_FOLDER, processed_filename)

    try:
        # 파일을 사용 중인 다른 프로세스가 완료될 때까지 대기
        (
            ffmpeg
            .input(temp_file_path)
            .output(processed_file_path, vcodec='libx264', acodec='aac', strict='experimental')
            .run()
        )
    except ffmpeg.Error as e:
        return jsonify({'error': str(e)}), 500

    # 임시 파일 삭제 시도
    try:
        os.remove(temp_file_path)
    except PermissionError as e:
        return jsonify({'error': str(e)}), 500

    processed_file_url = f'http://localhost:5000/processed/{processed_filename}'
    return jsonify({'url': processed_file_url}), 200

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
