Web Application for detecting Malaria

To execute the project in Local host:
clone the project in to your local machine using the command
 git clone https://github.com/master-projects-theses/kodavati-vamsi.git

1) python version 3.6.0
2) Pip package
3) Install all the dependencies in the requirements.txt using
    pip install requirements.txt
4) run the command
   python app.py
5) Open the local host end point in the chrome browser. 

To execute the project with Docker image:
The image for the web application is uploaded into Docker hub in a public repository
with the name vk1025/myapp/malaria_app_4  

1) install docker in to your Machine
2) For running the image in the container
  docker run -p 5000:5000 --name containername vk1025/myapp/malaria_app_4
3) Open the local host end point in the chrome browser.
   http://0.0.0.0:5000

To host the project in the Azure AKS Cluster please follow the instructions specified in the paper.
 
