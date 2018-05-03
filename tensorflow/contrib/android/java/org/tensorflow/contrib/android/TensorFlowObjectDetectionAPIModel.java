package org.tensorflow.contrib.android;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.util.Log;

import org.tensorflow.Graph;
import org.tensorflow.Operation;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Vector;

/**
 * Created by vardanpetrosyan on 11/19/17.
 */
public class TensorFlowObjectDetectionAPIModel {

    public static class Recognition {
        /**
         * A unique identifier for what has been recognized. Specific to the class, not the instance of
         * the object.
         */
        private final String id;

        /**
         * Display name for the recognition.
         */
        private final String title;

        /**
         * A sortable score for how good the recognition is relative to others. Higher should be better.
         */
        private final Float confidence;

        /** Optional location within the source image for the location of the recognized object. */
        private RectF location;

        public Recognition(
                final String id, final String title, final Float confidence, final RectF location) {
            this.id = id;
            this.title = title;
            this.confidence = confidence;
            this.location = location;
        }

        public String getId() {
            return id;
        }

        public String getTitle() {
            return title;
        }

        public Float getConfidence() {
            return confidence;
        }

        public RectF getLocation() {
            return new RectF(location);
        }

        public void setLocation(RectF location) {
            this.location = location;
        }

        @Override
        public String toString() {
            String resultString = "";
            if (id != null) {
                resultString += "[" + id + "] ";
            }

            if (title != null) {
                resultString += title + " ";
            }

            if (confidence != null) {
                resultString += String.format("(%.1f%%) ", confidence * 100.0f);
            }

            if (location != null) {
                resultString += location + " ";
            }

            return resultString.trim();
        }
    }

    // Only return this many results.
    private static final int MAX_RESULTS = 10;
    private static final int MAX_PRE_RESULTS = 300;

    // Config values.
    private String inputName;
    private int inputWidth;
    private int inputHeight;

    // Pre-allocated buffers.
    private Vector<String> labels = new Vector<>();
    private int[] intValues;
    private byte[] byteValues;
    private float[] outputLocations;
    private float[] outputScores;
    private float[] outputClasses;
    private float[] outputNumDetections;
    private String[] outputNames;

    private boolean logStats = false;

    private TensorFlowInferenceInterface inferenceInterface;

    /**
     * Initializes a native TensorFlow session for classifying images.
     *
     * @param assetManager The asset manager to be used to load assets.
     * @param modelFilename The filepath of the model GraphDef protocol buffer.
     * @param labelFilename The filepath of label file for classes.
     */
    public static TensorFlowObjectDetectionAPIModel create(
            final AssetManager assetManager,
            final String modelFilename,
            final String labelFilename,
            final int inputWidth, final int inputHeight) throws IOException {
        final TensorFlowObjectDetectionAPIModel d = new TensorFlowObjectDetectionAPIModel();

        InputStream labelsInput = null;
        String actualFilename = labelFilename.split("file:///android_asset/")[1];
        labelsInput = assetManager.open(actualFilename);
        BufferedReader br = null;
        br = new BufferedReader(new InputStreamReader(labelsInput));
        String line;
        while ((line = br.readLine()) != null) {
            d.labels.add(line);
        }
        br.close();


        d.inferenceInterface = new TensorFlowInferenceInterface(assetManager, modelFilename);

        final Graph g = d.inferenceInterface.graph();

        d.inputName = "image_tensor";
        // The inputName node has a shape of [N, H, W, C], where
        // N is the batch size
        // H = W are the height and width
        // C is the number of channels (3 for our purposes - RGB)
        final Operation inputOp = g.operation(d.inputName);
        if (inputOp == null) {
            throw new RuntimeException("Failed to find input Node '" + d.inputName + "'");
        }
        d.inputWidth = inputWidth;
        d.inputHeight = inputHeight;
        // The outputScoresName node has a shape of [N, NumLocations], where N
        // is the batch size.
        final Operation outputOp1 = g.operation("detection_scores");
        if (outputOp1 == null) {
            throw new RuntimeException("Failed to find output Node 'detection_scores'");
        }
        final Operation outputOp2 = g.operation("detection_boxes");
        if (outputOp2 == null) {
            throw new RuntimeException("Failed to find output Node 'detection_boxes'");
        }
        final Operation outputOp3 = g.operation("detection_classes");
        if (outputOp3 == null) {
            throw new RuntimeException("Failed to find output Node 'detection_classes'");
        }

        // Pre-allocate buffers.
        d.outputNames = new String[] {"detection_boxes", "detection_scores",
                "detection_classes", "num_detections"};
        d.intValues = new int[d.inputHeight * d.inputWidth];
        d.byteValues = new byte[d.inputHeight * d.inputWidth * 3];
        d.outputScores = new float[MAX_PRE_RESULTS];
        d.outputLocations = new float[MAX_PRE_RESULTS * 4];
        d.outputClasses = new float[MAX_PRE_RESULTS];
        d.outputNumDetections = new float[1];
        return d;
    }

    private TensorFlowObjectDetectionAPIModel() {}

    public List<Recognition> recognizeImage(final Bitmap bitmap) {
        // Log this method so that it can be analyzed with systrace.
        // Preprocess the image data from 0-255 int to normalized float based
        // on the provided parameters.
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        for (int i = 0; i < intValues.length; ++i) {
            byteValues[i * 3 + 2] = (byte) (intValues[i] & 0xFF);
            byteValues[i * 3 + 1] = (byte) ((intValues[i] >> 8) & 0xFF);
            byteValues[i * 3 + 0] = (byte) ((intValues[i] >> 16) & 0xFF);
        }

        // Copy the input data into TensorFlow.
        inferenceInterface.feed(inputName, byteValues, 1, inputWidth, inputHeight, 3);

        // Run the inference call.
        long startTime = System.currentTimeMillis();
        inferenceInterface.run(outputNames, logStats);
        Log.d("Vardan", " Vardan running took " + (System.currentTimeMillis() - startTime));

        // Copy the output Tensor back into the output array.
        outputLocations = new float[MAX_PRE_RESULTS * 4];
        outputScores = new float[MAX_PRE_RESULTS];
        outputClasses = new float[MAX_PRE_RESULTS];
        outputNumDetections = new float[1];

        inferenceInterface.fetch(outputNames[0], outputLocations);
        inferenceInterface.fetch(outputNames[1], outputScores);
        inferenceInterface.fetch(outputNames[2], outputClasses);
        inferenceInterface.fetch(outputNames[3], outputNumDetections);

        // Find the best detections.
        final PriorityQueue<Recognition> pq =
                new PriorityQueue<>(
                        1,
                        new Comparator<Recognition>() {
                            @Override
                            public int compare(final Recognition lhs, final Recognition rhs) {
                                // Intentionally reversed to put high confidence at the head of the queue.
                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                            }
                        });

        // Scale them back to the input size.
        for (int i = 0; i < outputNumDetections[0]; ++i) {
            final RectF detection =
                    new RectF(
                            outputLocations[4 * i + 1] * inputWidth,
                            outputLocations[4 * i] * inputHeight,
                            outputLocations[4 * i + 3] * inputWidth,
                            outputLocations[4 * i + 2] * inputHeight);
            pq.add(
                    new Recognition("" + i, labels.get((int) outputClasses[i]), outputScores[i], detection));
        }

        final ArrayList<Recognition> recognitions = new ArrayList<Recognition>();
        for (int i = 0; i < Math.min(pq.size(), MAX_RESULTS); ++i) {
            recognitions.add(pq.poll());
        }
        return recognitions;
    }

    public void enableStatLogging(final boolean logStats) {
        this.logStats = logStats;
    }

    public String getStatString() {
        return inferenceInterface.getStatString();
    }

    public void close() {
        inferenceInterface.close();
    }
}
