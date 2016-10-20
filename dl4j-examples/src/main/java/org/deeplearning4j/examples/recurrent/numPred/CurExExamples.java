package org.deeplearning4j.examples.recurrent.numPred;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.deeplearning4j.examples.recurrent.basic.BasicRNNExample.LEARNSTRING_CHARS_LIST;

/**
 * Created by YoungH on 10/19/16.
 * This example trains a RNN. We take the history data of current exchange rate from CNY to One U.S. dollar as input,
 * , it will predict the rate of next time.
 */
public class CurExExamples {

    private static INDArray input;
    private static INDArray labels;
    private static DataSet trainingData;

    public static void main(String[] args) throws IOException, InterruptedException {
        loadData();
    }

    private static void loadData() throws IOException, InterruptedException {

        List<Integer> trainList = new ArrayList<Integer>();
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File("dl4j-examples/src/main/resources/curExc/training.csv")));
        while(rr.hasNext()){
            trainList.add((int) (Double.parseDouble(rr.next().get(2).toString())*10000));
        }
        input = Nd4j.zeros(1, trainList.size());
        labels = Nd4j.zeros(1, trainList.size());

        for (int i=0;i<trainList.size();i++) {
            if (i==0){
                input.putScalar(trainList.get(i),i);
            }else if (i==trainList.size()-1) {
                labels.putScalar(trainList.get(i),i);
            }else{
                input.putScalar(trainList.get(i),i);
                labels.putScalar(trainList.get(i),i);
            }
        }



//        for (int i : trainList) {
//            int d0 = i/10000;
//            int d1 = (i-d0*10000)/1000;
//            int d2 = (i-d0*10000-d1*1000)/100;
//            int d3 = (i-d0*10000-d1*1000-d2*100)/10;
//            int d4 = (i-d0*10000-d1*1000-d2*100-d3*10);
//        }


        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File("dl4j-examples/src/main/resources/curExc/test.csv")));


    }
}
