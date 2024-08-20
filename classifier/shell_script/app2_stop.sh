cd ../classifier
kill -9 `cat app2_PID.txt`
rm app2_PID.txt
echo "Stopped Flask-App2."