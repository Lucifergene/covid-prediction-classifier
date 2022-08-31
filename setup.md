# Setting up Remote AWS-EC2 Environment


```
sudo yum update -y
sudo amazon-linux-extras install epel -y
sudo yum install git
git clone https://github.com/Lucifergene/covid-prediction-classifier.git
cd covid-prediction-classifier/
sudo yum install -y python3
pip3 install tensorflow==2.3.0
pip3 install -r requirements.txt
sudo yum install -y p7zip
cd models
7za e datty.7z.001

python3 app.py
```