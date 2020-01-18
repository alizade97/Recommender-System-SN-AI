**** Interest Matching in Social Network based on AI ****

Recommnedation Engine configuration and installation


### PREREQUISITES ###
--- Python version bigger 3.8.0
--- Python Package Manager such as Ananconda or pip package manager
--- code editor can be used

### INSTALLATION ###

Step 1. Python installation: https://www.python.org/downloads/windows/
	It will be set as environment variable when installed
	If you want to check, by running python keyword you can see version

Step 2. Anaconda Package Manager Installation: https://www.anaconda.com/distribution/
	After installation for downloading packages you can use either Anaconda Navigator that is GUI for installing packages or anaconda propmt 
	PIP package manager installation: https://bootstrap.pypa.io/get-pip.py
	By entering this url save this and open command prompt go suitable folder and run python get-pip.py. By running pip -v you can check version

Step 3.After those installations you need to either create environment (in PIP you can install virtual env packages using pip install virtualenv command and create environment by virtualenv yourname and activate it by using yourname\Scripts\activateor) and install those packages in base directory:

Flask==1.1.1
Flask-MySQLdb==0.2.0
Flask-SocketIO==4.2.1
gensim==3.8.1
googletrans==2.4.0
gunicorn==19.9.0
h5py==2.9.0
HeapDict==1.0.1
html5lib==1.0.1
imageio==2.6.1
imagesize==1.1.0
jieba==0.39
json5==0.8.5
Keras==2.2.4
Keras-Applications==1.0.8
Keras-Preprocessing==1.1.0
mtcnn==0.0.9
mysqlclient==1.4.4
numpy==1.16.5
numpydoc==0.9.1
opencv-python==4.1.1.26
pandas==0.25.2
requests==2.22.0
scikit-image==0.15.0
scikit-learn==0.21.3
scipy==1.3.1
tensorboard==1.13.1
tensorflow==1.13.1
tensorflow-estimator==1.13.0
tensorflow-hub==0.6.0
urllib3==1.24.2
Werkzeug==0.16.0

Step 4. After import tritonre.sql file from db folder to your database and by opening tritonre.py file and give configure your database server details

Step 5. And configure back-end address to your file

Step6. Enter command prompt and run python tritonre.py
