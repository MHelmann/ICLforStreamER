cd ../classifier
kill -9 `cat app1_PID.txt`
rm app1_PID.txt
echo "Stopped Flask-App1."