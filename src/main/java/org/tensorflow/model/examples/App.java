package org.tensorflow.model.examples;

import java.io.IOException;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Graph;
import org.tensorflow.op.Ops;

import org.tensorflow.model.examples.cnn.fastrcnn.DetectionResultParser;
import org.tensorflow.model.examples.cnn.fastrcnn.FasterRcnnInception;
import org.tensorflow.model.examples.cnn.lenet.CnnMnist;
import org.tensorflow.model.examples.cnn.vgg.VGG11OnFashionMnist;
import org.tensorflow.model.examples.dense.SimpleMnist;
import org.tensorflow.model.examples.regression.linear.LinearRegressionExample;
import org.tensorflow.model.examples.tensors.TensorCreation;

import com.google.common.collect.Table;

public class App {

    private static void printTable(Table<Integer, String, Float> table) throws IOException {

        // Create a parser for parsing the detection results
        DetectionResultParser parser = new DetectionResultParser();
        
        // Initialize the parser with the table
        parser.load(table);

        // Iterate the row mapping print the data of each row
        for (Integer row : parser.getKeySetPerRow()) {
            System.out.println("Detection class: " + parser.getLabelByRow(row));
            System.out.println("Detection score: " + parser.getScoreByRow(row));
            System.out.println("ymin: " + parser.getYminByRow(row));
            System.out.println("xmin: " + parser.getXminByRow(row));
            System.out.println("ymax: " + parser.getYmaxByRow(row));
            System.out.println("xmax: " + parser.getXmaxByRow(row));
            System.out.println("------------------");
        }
    }

    public static void main(String[] args) throws IOException {
        switch (args[0]) {
            case "fastrcnn" -> {
                // load saved model
                String modelPath = "models/faster_rcnn_inception_resnet_v2_1024x1024";
                SavedModelBundle model = SavedModelBundle.load(modelPath, "serve");

                // TF computing things
                Graph g = new Graph();
                Ops tf = Ops.create(g);

                // Run detection task on multiple images
                for (int i = 0; i < 2; i++) {
                    String imagePath = String.format("testimages/image%d.jpg", i);
                    String outputImagePath = String.format("outputs/image%drcnn.jpg", i);
                    Table<Integer, String, Float> resultTable = FasterRcnnInception.runDetectionTask(model, g, tf, imagePath, outputImagePath);
                    printTable(resultTable);
                }
            }
            case "lenet" -> {
                int epochs = 1;
                int minibatchSize = 64;
                String optimizerName = "adam";
                CnnMnist.main(epochs, minibatchSize, optimizerName);
            }
            case "vgg" ->
                VGG11OnFashionMnist.main();
            case "linear" ->
                LinearRegressionExample.main();
            case "logistic" ->
                SimpleMnist.main();
            case "tensors" ->
                TensorCreation.main();
            default ->
                System.out.println("Invalid mode name!");
        }
    }
}
