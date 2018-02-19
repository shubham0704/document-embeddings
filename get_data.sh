sudo docker volume create --name data
sudo su
cd /var/lib/docker/volumes/data/_data
wget http://www.cs.cornell.edu/~schnabts/eval/wiki_2008.zip
unzip wiki_2008.zip
rm wiki_2008.zip
exit
# to run the container with the data do - sudo docker run -it -v data:/data 63ad552850f8 /bin/bash
