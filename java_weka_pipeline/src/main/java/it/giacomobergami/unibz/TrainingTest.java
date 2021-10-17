package it.giacomobergami.unibz;

import com.fasterxml.jackson.annotation.JsonProperty;

public class TrainingTest {

    @JsonProperty("test")
    public String test;

    @JsonProperty("train")
    public String train;

    public TrainingTest() {
        test = null;
        train= null;
    }

    public TrainingTest(String test, String train) {
        this.test = test;
        this.train = train;
    }

    public String getTest() {
        return test;
    }

    public void setTest(String test) {
        this.test = test;
    }

    public String getTrain() {
        return train;
    }

    public void setTrain(String train) {
        this.train = train;
    }

    @Override
    public String toString() {
        return "{" +
                "test='" + test + '\'' +
                ", train='" + train + '\'' +
                '}';
    }
}
