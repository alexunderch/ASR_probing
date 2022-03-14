#!/bin/bash
if ! [ $(id -u) = 0 ]; then
   echo "The script need to be run as root." >&2
   exit 1
fi

apt-get install default-jre unzip
wget http://nlp.stanford.edu/software/stanford-corenlp-latest.zip
unzip stanford-corenlp-latest.zip -d core-nlp-latest
rm stanford-corenlp-latest.zip
cp run_server.sh core-nlp-latest/stanford-corenlp-4.4.0/run_server.sh
cd core-nlp-latest/stanford-corenlp-4.4.0 
nohup sh ./run_server.sh
