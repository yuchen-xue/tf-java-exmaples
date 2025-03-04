/*
 *  Copyright 2021, 2024 The TensorFlow Authors. All Rights Reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *  =======================================================================
 */

package org.tensorflow.model.examples.cnn.fastrcnn;
/*

From the web page this is the output dictionary

num_detections: a tf.int tensor with only one value, the number of detections [N].
detection_boxes: a tf.float32 tensor of shape [N, 4] containing bounding box coordinates in the following order: [ymin, xmin, ymax, xmax].
detection_classes: a tf.int tensor of shape [N] containing detection class index from the label file.
detection_scores: a tf.float32 tensor of shape [N] containing detection scores.
raw_detection_boxes: a tf.float32 tensor of shape [1, M, 4] containing decoded detection boxes without Non-Max suppression. M is the number of raw detections.
raw_detection_scores: a tf.float32 tensor of shape [1, M, 90] and contains class score logits for raw detection boxes. M is the number of raw detections.
detection_anchor_indices: a tf.float32 tensor of shape [N] and contains the anchor indices of the detections after NMS.
detection_multiclass_scores: a tf.float32 tensor of shape [1, N, 90] and contains class score distribution (including background) for detection boxes in the image including background class.

However using
venv\Scripts\python.exe venv\Lib\site-packages\tensorflow\python\tools\saved_model_cli.py show --dir models\faster_rcnn_inception_resnet_v2_1024x1024 --all
2021-03-19 12:25:37.000143: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cudart64_110.dll

MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['__saved_model_init_op']:
  The given SavedModel SignatureDef contains the following input(s):
  The given SavedModel SignatureDef contains the following output(s):
    outputs['__saved_model_init_op'] tensor_info:
        dtype: DT_INVALID
        shape: unknown_rank
        name: NoOp
  Method name is:

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['input_tensor'] tensor_info:
        dtype: DT_UINT8
        shape: (1, -1, -1, 3)
        name: serving_default_input_tensor:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['detection_anchor_indices'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 300)
        name: StatefulPartitionedCall:0
    outputs['detection_boxes'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 300, 4)
        name: StatefulPartitionedCall:1
    outputs['detection_classes'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 300)
        name: StatefulPartitionedCall:2
    outputs['detection_multiclass_scores'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 300, 91)
        name: StatefulPartitionedCall:3
    outputs['detection_scores'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 300)
        name: StatefulPartitionedCall:4
    outputs['num_detections'] tensor_info:
        dtype: DT_FLOAT
        shape: (1)
        name: StatefulPartitionedCall:5
    outputs['raw_detection_boxes'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 300, 4)
        name: StatefulPartitionedCall:6
    outputs['raw_detection_scores'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 300, 91)
        name: StatefulPartitionedCall:7
  Method name is: tensorflow/serving/predict

Defined Functions:
  Function Name: '__call__'
    Option #1
      Callable with:
        Argument #1
          input_tensor: TensorSpec(shape=(1, None, None, 3), dtype=tf.uint8, name='input_tensor')

So it appears there's a discrepancy between the web page and running saved_model_cli as
num_detections: a tf.int tensor with only one value, the number of detections [N].
but the actual tensor is DT_FLOAT according to saved_model_cli
also the web page states
detection_classes: a tf.int tensor of shape [N] containing detection class index from the label file.
but again the actual tensor is DT_FLOAT according to saved_model_cli.
*/

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Result;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.FloatNdArray;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.core.Reshape;
import org.tensorflow.op.image.DecodeJpeg;
import org.tensorflow.op.image.EncodeJpeg;
import org.tensorflow.op.io.ReadFile;
import org.tensorflow.op.io.WriteFile;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TString;
import org.tensorflow.types.TUint8;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;

/**
 * Loads an image using ReadFile and DecodeJpeg and then uses the saved model
 * faster_rcnn/inception_resnet_v2_1024x1024/1 to detect objects with a
 * detection score greater than 0.3
 * Uses the DrawBounding boxes
 */
public final class FasterRcnnInception {

    private static DecodeJpeg getDecodedImage(Ops tf, String imagePath) {
        Constant<TString> fileName = tf.constant(imagePath);
        ReadFile readFile = tf.io.readFile(fileName);
        DecodeJpeg.Options options = DecodeJpeg.channels(3L);
        return tf.image.decodeJpeg(readFile.contents(), options);
    }

    private static Table<Integer, String, Float> processDetectionResult(Session.Runner runner, Ops tf,
            Result outputTensorMap, Shape imageShape, TUint8 reshapeTensor, String outputImagePath) {

        TFloat32 numDetections = (TFloat32) outputTensorMap.get("num_detections").get();
        int numDetects = (int) numDetections.getFloat(0);

        // Create a table to store the results
        Table<Integer, String, Float> resultTable = HashBasedTable.create();

        if (numDetects > 0) {
            TFloat32 detectionBoxes = (TFloat32) outputTensorMap.get("detection_boxes").get();
            TFloat32 detectionScores = (TFloat32) outputTensorMap.get("detection_scores").get();
            TFloat32 detectionClasses = (TFloat32) outputTensorMap.get("detection_classes").get();
            ArrayList<FloatNdArray> boxArray = new ArrayList<>();

            // TODO tf.image.combinedNonMaxSuppression
            for (int n = 0; n < numDetects; n++) {
                // put probability and position in outputMap
                float detectionScore = detectionScores.getFloat(0, n);
                float detectionClass = detectionClasses.getFloat(0, n);

                // only include those classes with detection score greater than 0.3f
                if (detectionScore > 0.3f) {
                    FloatNdArray detectionBox = detectionBoxes.get(0, n);
                    boxArray.add(detectionBox);

                    // Collect detection results
                    resultTable.put(n, "detection_class", detectionClass);
                    resultTable.put(n, "detection_score", detectionScore);
                    resultTable.put(n, "ymin", detectionBox.getFloat(0));
                    resultTable.put(n, "xmin", detectionBox.getFloat(1));
                    resultTable.put(n, "ymax", detectionBox.getFloat(2));
                    resultTable.put(n, "xmax", detectionBox.getFloat(3));
                }
            }
            /*
             * These values are also returned by the FasterRCNN, but we don't use them in
             * this example.
             * TFloat32 rawDetectionBoxes = (TFloat32)
             * outputTensorMap.get("raw_detection_boxes").get();
             * TFloat32 rawDetectionScores = (TFloat32)
             * outputTensorMap.get("raw_detection_scores").get();
             * TFloat32 detectionAnchorIndices = (TFloat32)
             * outputTensorMap.get("detection_anchor_indices").get();
             * TFloat32 detectionMulticlassScores = (TFloat32)
             * outputTensorMap.get("detection_multiclass_scores").get();
             */

            // 2-D. A list of RGBA colors to cycle through for the boxes.
            Operand<TFloat32> colors = tf.constant(new float[][] {
                    { 0.9f, 0.3f, 0.3f, 0.0f },
                    { 0.3f, 0.3f, 0.9f, 0.0f },
                    { 0.3f, 0.9f, 0.3f, 0.0f }
            });

            Shape boxesShape = Shape.of(1, boxArray.size(), 4);
            int boxCount = 0;
            // 3-D with shape `[batch, num_bounding_boxes, 4]` containing bounding boxes
            try (TFloat32 boxes = TFloat32.tensorOf(boxesShape)) {
                // batch size of 1
                boxes.setFloat(1, 0, 0, 0);
                for (FloatNdArray floatNdArray : boxArray) {
                    boxes.set(floatNdArray, 0, boxCount);
                    boxCount++;
                }

                // Placeholders for boxes and path to outputimage
                Placeholder<TFloat32> boxesPlaceHolder = tf.placeholder(TFloat32.class, Placeholder.shape(boxesShape));
                Placeholder<TString> outImagePathPlaceholder = tf.placeholder(TString.class);
                // Create JPEG from the Tensor with quality of 100%
                EncodeJpeg.Options jpgOptions = EncodeJpeg.quality(100L);
                // convert the 4D input image to normalised 0.0f - 1.0f
                // Draw bounding boxes using boxes tensor and list of colors
                // multiply by 255 then reshape and recast to TUint8 3D tensor
                WriteFile writeFile = tf.io.writeFile(outImagePathPlaceholder,
                        tf.image.encodeJpeg(
                                tf.dtypes.cast(tf.reshape(
                                        tf.math.mul(
                                                tf.image.drawBoundingBoxes(tf.math.div(
                                                        tf.dtypes.cast(tf.constant(reshapeTensor),
                                                                TFloat32.class),
                                                        tf.constant(255.0f)),
                                                        boxesPlaceHolder, colors),
                                                tf.constant(255.0f)),
                                        tf.array(
                                                imageShape.asArray()[0],
                                                imageShape.asArray()[1],
                                                imageShape.asArray()[2])),
                                        TUint8.class),
                                jpgOptions));
                // output the JPEG to file
                runner.feed(outImagePathPlaceholder, TString.scalarOf(outputImagePath))
                        .feed(boxesPlaceHolder, boxes)
                        .addTarget(writeFile).run();
            }
        }
        return resultTable;
    }

    public static Table<Integer, String, Float> runDetectionTask(SavedModelBundle model, Graph g, Ops tf,
            String imagePath, String outputImagePath) {

        try (Session s = new Session(g)) {
            DecodeJpeg decodeImage = getDecodedImage(tf, imagePath);

            Shape imageShape;
            try (var shapeResult = s.runner().fetch(decodeImage).run()) {
                imageShape = shapeResult.get(0).shape();
            }

            // reshape the tensor to 4D in order to feed it into the model
            Reshape<TUint8> reshape = tf.reshape(decodeImage,
                    tf.array(1,
                            imageShape.asArray()[0],
                            imageShape.asArray()[1],
                            imageShape.asArray()[2]));

            // Run the model, collect the result and save the output image
            try (var reshapeResult = s.runner().fetch(reshape).run()) {
                TUint8 reshapeTensor = (TUint8) reshapeResult.get(0);
                Map<String, Tensor> feedDict = new HashMap<>();
                // The given SavedModel SignatureDef input
                feedDict.put("input_tensor", reshapeTensor);
                // The given SavedModel MetaGraphDef key
                // detection_classes, detectionBoxes etc. are model output names
                try (Result outputTensorMap = model.function("serving_default").call(feedDict)) {
                    return processDetectionResult(s.runner(), tf, outputTensorMap, imageShape, reshapeTensor,
                            outputImagePath);
                }
            }
        }
    }
}
