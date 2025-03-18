package com.ontariotechu.sofe3980U;

import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import com.opencsv.CSVReader;
import com.opencsv.CSVReaderBuilder;

/**
 * Evaluate Multi-Class Classification Model
 */
public class App {
    public static void main(String[] args) {
        String filePath = "model.csv";
        List<Integer> actualClasses = new ArrayList<>();
        List<int[]> predictedClassDistributions = new ArrayList<>();
        List<double[]> predictedProbabilities = new ArrayList<>();

        // Read CSV File
        try (FileReader filereader = new FileReader(filePath);
             CSVReader csvReader = new CSVReaderBuilder(filereader).withSkipLines(1).build()) {

            List<String[]> allData = csvReader.readAll();

            for (String[] row : allData) {
                int y_true = Integer.parseInt(row[0]) - 1;  // Convert 1-based index to 0-based
                double[] y_pred = new double[5];

                for (int i = 0; i < 5; i++) {
                    y_pred[i] = Double.parseDouble(row[i + 1]);
                }

                actualClasses.add(y_true);
                predictedProbabilities.add(y_pred);
            }

        } catch (Exception e) {
            System.out.println("Error reading the CSV file");
            return;
        }

        // Determine predicted classes based on max probability
        List<Integer> predictedClasses = new ArrayList<>();
        for (double[] probs : predictedProbabilities) {
            predictedClasses.add(getPredictedClass(probs));
        }

        // Compute metrics
        double categoricalCE = calculateCategoricalCrossEntropy(actualClasses, predictedProbabilities);
        int[][] confusionMatrix = calculateConfusionMatrix(actualClasses, predictedClasses);

        // Display results
        System.out.println("Cross-Entropy (CE): " + categoricalCE);
        System.out.println("Confusion Matrix:");
        printConfusionMatrix(confusionMatrix);
    }

    // Get predicted class by selecting index with max probability
    public static int getPredictedClass(double[] probs) {
        int maxIndex = 0;
        double maxProb = probs[0];

        for (int i = 1; i < probs.length; i++) {
            if (probs[i] > maxProb) {
                maxProb = probs[i];
                maxIndex = i;
            }
        }
        return maxIndex;  // 0-based index
    }

    // Compute Categorical Cross-Entropy (CE)
    public static double calculateCategoricalCrossEntropy(List<Integer> actual, List<double[]> predicted) {
        double sum = 0.0;
        int n = actual.size();

        for (int i = 0; i < n; i++) {
            int y_true = actual.get(i);
            double p_true = predicted.get(i)[y_true];

            // Prevent log(0) issue by clamping probabilities
            p_true = Math.max(Math.min(p_true, 1 - 1e-9), 1e-9);
            sum += Math.log(p_true);
        }

        return -sum / n;
    }

    // Compute Confusion Matrix
    public static int[][] calculateConfusionMatrix(List<Integer> actual, List<Integer> predicted) {
        int[][] matrix = new int[5][5]; // 5x5 confusion matrix for classes 1-5

        for (int i = 0; i < actual.size(); i++) {
            int y_true = actual.get(i);
            int y_pred = predicted.get(i);
            matrix[y_true][y_pred]++;
        }

        return matrix;
    }

    // Print Confusion Matrix
    public static void printConfusionMatrix(int[][] matrix) {
        System.out.println("     Predicted");
        System.out.println("     1   2   3   4   5");
        for (int i = 0; i < 5; i++) {
            System.out.print((i + 1) + "  | ");
            for (int j = 0; j < 5; j++) {
                System.out.print(matrix[i][j] + "   ");
            }
            System.out.println();
        }
    }
}
