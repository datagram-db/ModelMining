#!/usr/bin/env bash
mvn clean compile assembly:single
cp target/Calcutta-1.0-SNAPSHOT-jar-with-dependencies.jar ../