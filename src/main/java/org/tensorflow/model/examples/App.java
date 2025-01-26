package org.tensorflow.model.examples;

import com.google.common.collect.Table;
import org.tensorflow.model.examples.cnn.fastrcnn.FasterRcnnInception;
import org.tensorflow.model.examples.cnn.lenet.CnnMnist;
import org.tensorflow.model.examples.cnn.vgg.VGG11OnFashionMnist;
import org.tensorflow.model.examples.dense.SimpleMnist;
import org.tensorflow.model.examples.regression.linear.LinearRegressionExample;
import org.tensorflow.model.examples.tensors.TensorCreation;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class App {

    // private static final String COCO_LABELS_FILE = "coco-labels-2017.txt";

    private static String[] loadExternalLabels(String labelFileName) throws IOException {
    /**
     * Load COCO dataset labels from an extertal text file.
     */
        String strFilePath = ClassLoader.getSystemResource(labelFileName).getPath();
        Path filePath = Paths.get(strFilePath);
        List<String> lines = Files.lines(filePath).collect(Collectors.toList());
        String[] labelsArray = lines.toArray(new String[0]);
        return labelsArray;
    }

    private static void printTable(Table<Integer, String, Float> table) throws IOException {
        // Create a mapping of each row in the table
        Map<Integer, Map<String, Float>> map = table.rowMap();

        // Iterate the row mapping print the data of each row
        for (Integer row : map.keySet()) {
            System.out.println("Detection class: " + loadExternalLabels("coco-labels-2017.txt")[Math.round(map.get(row).get("detection_class")) - 1]);
            System.out.println("Detection score: " + map.get(row).get("detection_score"));
            System.out.println("ymin: " + map.get(row).get("ymin"));
            System.out.println("xmin: " + map.get(row).get("xmin"));
            System.out.println("ymax: " + map.get(row).get("ymax"));
            System.out.println("xmax: " + map.get(row).get("xmax"));
            System.out.println("------------------");
        }
    }

    public static void main(String[] args) throws IOException {
        switch (args[0]) {
            case "fastrcnn" -> {
                Table<Integer, String, Float> resultTable = FasterRcnnInception.main(Arrays.stream(args, 1, args.length).toArray(String[]::new));
                printTable(resultTable);
            }
            case "lenet" ->
                CnnMnist.main(Arrays.stream(args, 1, args.length).toArray(String[]::new));
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
