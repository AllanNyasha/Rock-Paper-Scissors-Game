from flask import Flask,render_template, flash,Response
from keras.models import load_model
import tensorflow as tf
import keras
import operator
import cv2
import time
from random import choice
from tensorflow.python.keras.backend import set_session

app=Flask(__name__)


global graph
graph = tf.get_default_graph()

session = keras.backend.get_session()
init = tf.global_variables_initializer()
session.run(init)
model = load_model('first_try.h5')
#element was not found in the graph
model._make_predict_function()
print('model loaded')

prev_move = None

def calculate_winner(move1, move2):
    if move1 == move2:
        return "Tie"

    if move1 == "rock":
        if move2 == "scissors":
            return "User"
        if move2 == "paper":
            return "Computer"

    if move1 == "paper":
        if move2 == "rock":
            return "User"
        if move2 == "scissors":
            return "Computer"

    if move1 == "scissors":
        if move2 == "paper":
            return "User"
        if move2 == "rock":
            return "Computer"






@app.route('/')
def index():
    return render_template('index.html')

#
# cap = cv2.VideoCapture(0)
# _, frame = cap.read()
# frame = cv2.resize(frame, (150, 150))
def gen():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    prev_move = None

    # Read until video is completed
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # frame = cv2.resize(frame, (500, 500))

        ##########################################################################################
        x1 = int(0.2 * frame.shape[1])
        y1 = 1
        x2 = frame.shape[1] - 10
        y2 = int(0.2 * frame.shape[1])

        cv2.rectangle(frame, (x1 + 500, y1 - 1), (x2 - 20, y2 + 80), (255, 0, 0), 1)
        # extracting the roi
        roi = frame[y1:y2, x1:x2]

        # resizing the roi so that it can be fed into the model
        roi = cv2.resize(roi, (150, 150))
        # mine has all 3 channels
        #  roi=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        _, test_image = cv2.threshold(roi, 120, 265, cv2.THRESH_BINARY)
        # cv2.imshow('test', test_image)

        # Batch of 1
        with graph.as_default():
          set_session(session)
          result = model.predict(test_image.reshape(1, 150, 150, 3))
        prediction = {'none': result[0][0],
                      'paper': result[0][1],
                      'rock': result[0][2],
                      'scissors': result[0][3]}

        # sorting based on top prediction
        prediction = sorted(prediction.items(), key=operator.itemgetter(1),
                            reverse=True)

        ################################################

        # ROI
        x1 = int(0.0001 * frame.shape[1])
        y1 = 1
        x2 = frame.shape[1] - 10
        y2 = int(0.0001 * frame.shape[1])

        # cv2.rectangle(frame, (x1-1 ,y1-1),(x2+10,y2+5),(255,0,0),1)
        # extracting the roi
        roi = frame[y1:y2, x1:x2]

        # computer_move_name = choice(['rock', 'paper', 'scissors'])

        # Display the prediction
        # cv2.putText(frame,prediction[0][0],(x1+100,y2+30),cv2.FONT_HERSHEY_PLAIN,2,(0, 255, 0), 5)
        user_move_name = prediction[0][0]

        if prev_move != user_move_name:
            if user_move_name != "none":
                computer_move_name = choice(['rock', 'paper', 'scissors'])
                winner = calculate_winner(user_move_name, computer_move_name)
            else:
                computer_move_name = "none"
                winner = "Waiting..."
        prev_move = user_move_name

        if computer_move_name != "none":
            icon = cv2.imread(
                "images/{}.png".format(computer_move_name))
            icon = cv2.resize(icon, (330, 190))
            frame[10:200, 20:350] = icon

        font = cv2.FONT_HERSHEY_SIMPLEX
        # display computer move
        # cv2.putText(frame, computer_move_name, (x1 + 90, y2 + 15), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 5)

        cv2.putText(frame, "Put your hand in this box! ",
                    (730, 370), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Your Move: " + user_move_name,
                    (60, 300), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Computer's Move: " + computer_move_name,
                    (50, 230), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        try:
            cv2.putText(frame, "Winner: " + winner,
                        (100, 650), font, 2, (0, 0, 255), 4, cv2.LINE_AA)
        except (RuntimeError, TypeError, NameError):
            print(RuntimeError, TypeError, NameError)

        ##########################################################################################

        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
      ##  time.sleep(0.1)




@app.route('/video_feed')
def video_feed():
    prev_move = None
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__==('__main__'):
    app.run()