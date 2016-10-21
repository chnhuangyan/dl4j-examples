package org.deeplearning4j.examples.recurrent.numPred;


import org.deeplearning4j.examples.recurrent.stockPred.DailyData;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.NoSuchElementException;

/**
 * Created by YoungH on 10/21/16.
 */
public class CurExDataIterator {

    private static final int VECTOR_SIZE = 1;
    //每批次的训练数据组数
    private int batchNum;

    //每组训练数据长度(DailyData的个数)
    private int exampleLength;

    //数据集
    private List<Double[]> dataList;

    //存放剩余数据组的index信息
    private List<Integer> dataRecord;

    private double[] maxNum={0,0};

    public CurExDataIterator() {
        dataRecord = new ArrayList<>();
    }

    public boolean loadData(String fileName, int batchNum, int exampleLength){
        this.batchNum = batchNum;
        this.exampleLength = exampleLength;
        //加载文件中的股票数据
        try {
            dataList = readDataFromFile(fileName);
        }catch (Exception e){
            e.printStackTrace();
            return false;
        }
        //重置训练批次列表
        resetDataRecord();
        return true;
    }

    public List<Double[]> readDataFromFile(String fileName) throws IOException {
        dataList = new ArrayList<>();
        FileInputStream fis = new FileInputStream(fileName);
        BufferedReader in = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
        String line = in.readLine();

        System.out.println("读取数据..");
        while(line!=null){
            String[] strArr = line.split(",");
            double tempTrainNum = Double.valueOf(strArr[2]);
            double tempLabelNum = Double.valueOf(strArr[3]);
            maxNum[0]=Math.max(maxNum[0],tempTrainNum);
            maxNum[1]=Math.max(maxNum[1],tempLabelNum);
            dataList.add(new Double[]{tempTrainNum,tempLabelNum});
            line = in.readLine();
        }
        in.close();
        fis.close();
//        System.out.println("反转list...");
//        Collections.reverse(dataList);
        return dataList;
    }

    private void resetDataRecord(){
        dataRecord.clear();
        int total = dataList.size()/exampleLength+1;
        for( int i=0; i<total; i++ ){
            dataRecord.add(i * exampleLength);
        }
    }

    public double[] getMaxArr(){
        return this.maxNum;
    }

    public void reset(){
        resetDataRecord();
    }

    public boolean hasNext(){
        return dataRecord.size() > 0;
    }

    public DataSet next(){
        return next(batchNum);
    }

    public DataSet next(int num){
        if( dataRecord.size() <= 0 ) {
            throw new NoSuchElementException();
        }
        int actualBatchSize = Math.min(num, dataRecord.size());
        int actualLength = Math.min(exampleLength,dataList.size()-dataRecord.get(0)-1);
        INDArray input = Nd4j.create(new int[]{actualBatchSize,VECTOR_SIZE,actualLength}, 'f');
        INDArray label = Nd4j.create(new int[]{actualBatchSize,1,actualLength}, 'f');
        double trainData,labelData;
        //获取每批次的训练数据和标签数据
        for(int i=0;i<actualBatchSize;i++){
            int index = dataRecord.remove(0);
            int endIndex = Math.min(index+exampleLength,dataList.size()-1);

            for(int j=index;j<endIndex;j++){
                trainData = dataList.get(j)[0];
                labelData = dataList.get(j)[1];
                //构造训练向量
                int c = endIndex-j-1;
                input.putScalar(new int[]{i, 0, c}, trainData/maxNum[0]);

                //构造label向量
                label.putScalar(new int[]{i, 0, c}, labelData/maxNum[1]);
            }
            if(dataRecord.size()<=0) {
                break;
            }
        }

        return new DataSet(input, label);
    }

    public int totalExamples() {
        return (dataList.size()) / exampleLength;
    }

}
