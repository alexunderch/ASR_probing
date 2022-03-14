#!/bin/bash
for file in `find . -name "*.jar"`; do export
CLASSPATH="$CLASSPATH:`realpath $file`"; done
nohup java -cp "*" -mx4g edu.stanford.nlp.pipeline.StanfordCoreNLPServer -outputFormat json

