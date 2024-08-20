# !/bin/sh
cd ../classifier
nohup python classifier_app1.py > app1.log 2>&1 &
echo $! > app1_PID.txt
echo "Started Flask-App1."