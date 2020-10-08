gnome-terminal -x bash -c "/usr/bin/python2.7 /brat-v1.3_Crunchy_Frog/standalone.py"
gnome-terminal -x bash -c "docker run -p 8501:8501 --mount type=bind,source=/mult_model/,target=/models/mult_model -t tensorflow/serving --model_config_file=/models/mult_model/models.config"
gnome-terminal -x bash -c "python3 ./manage.py runserver  0.0.0.0:8000"
