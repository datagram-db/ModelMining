#!/usr/bin/env bash
java -jar target/Calcutta-1.0-SNAPSHOT-jar-with-dependencies.jar $1 xray     parall
java -jar target/Calcutta-1.0-SNAPSHOT-jar-with-dependencies.jar $1 sepsis
java -jar target/Calcutta-1.0-SNAPSHOT-jar-with-dependencies.jar $1 bpi11