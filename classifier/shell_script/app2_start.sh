# !/bin/sh
cd ../classifier
nohup python classifier_app2.py > app2.log 2>&1 &
echo $! > app2_PID.txt
echo "Started Flask-App2."