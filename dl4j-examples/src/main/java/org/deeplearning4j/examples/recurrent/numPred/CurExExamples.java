package org.deeplearning4j.examples.recurrent.numPred;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.examples.recurrent.stockPred.StockDataIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

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

    private static final int IN_NUM = 1;
    private static final int OUT_NUM = 1;
    private static final int Epochs = 200;

    private static final int lstmLayer1Size = 25;
    private static final int lstmLayer2Size = 50;
    private static final int lstmLayer3Size = 50;
    private static final int lstmLayer4Size = 25;


    private static List<Double> testList = new ArrayList<Double>();


    public static void main(String[] args) throws IOException, InterruptedException {
        String inputFile = "dl4j-examples/src/main/resources/curExc/training.csv";
        int batchSize = 1;
        int exampleLength = 40;

        CurExDataIterator iterator = new CurExDataIterator();
        iterator.loadData(inputFile,batchSize,exampleLength);

        loadTestData();

        MultiLayerNetwork net = getNetModel(IN_NUM,OUT_NUM);
        train(net, iterator);
    }


    public static MultiLayerNetwork getNetModel(int nIn,int nOut){
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(10)
            .learningRate(0.05)
            .rmsDecay(0.5)
            .seed(12345)
            .regularization(true)
            .l2(0.001)
            .weightInit(WeightInit.XAVIER)
            .updater(Updater.RMSPROP)
            .list()
            .layer(0, new GravesLSTM.Builder().nIn(nIn).nOut(lstmLayer1Size)
                .activation("tanh").build())
            .layer(1, new GravesLSTM.Builder().nIn(lstmLayer1Size).nOut(lstmLayer2Size)
                .activation("tanh").build())
            .layer(2, new GravesLSTM.Builder().nIn(lstmLayer2Size).nOut(lstmLayer3Size)
                .activation("tanh").build())
            .layer(3, new GravesLSTM.Builder().nIn(lstmLayer3Size).nOut(lstmLayer4Size)
                .activation("tanh").build())
            .layer(4, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE).activation("identity")
                .nIn(lstmLayer4Size).nOut(nOut).build())
            .pretrain(false).backprop(true)
            .backpropType(BackpropType.TruncatedBPTT).tBPTTBackwardLength(20).tBPTTForwardLength(20)
            .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(10));

        return net;
    }

    public static void train(MultiLayerNetwork net, CurExDataIterator iterator){
        //迭代训练
        for(int i=0;i<Epochs;i++) {
            DataSet dataSet = null;
            while (iterator.hasNext()) {
                dataSet = iterator.next();
                net.fit(dataSet);
            }
            iterator.reset();
            System.out.println();
            System.out.println("=================>完成第"+i+"次完整训练");
            INDArray initArray = getInitArray(iterator);

            System.out.println("预测结果：");
            INDArray tempOutput = initArray;
            for(int j=0;j<15;j++) {
                tempOutput = net.rnnTimeStep(tempOutput);
                System.out.print(tempOutput.getDouble(0)*iterator.getMaxArr()[1]+" ");
            }
            System.out.println();
            net.rnnClearPreviousState();
        }
    }


    private static void loadTestData() throws IOException, InterruptedException {

        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File("dl4j-examples/src/main/resources/curExc/test.csv")));

        while(rrTest.hasNext()){
            testList.add(Double.parseDouble(rrTest.next().get(2).toString()));
        }
    }

    private static INDArray getInitArray(CurExDataIterator iter){
        double[] maxNums = iter.getMaxArr();
        INDArray initArray = Nd4j.zeros(1, 1, 1);
        initArray.putScalar(new int[]{0,0,0}, testList.get(0)/maxNums[0]);
        return initArray;
    }
}
