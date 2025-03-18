package com.ontariotechu.sofe3980U;

import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import com.opencsv.CSVReader;
import com.opencsv.CSVReaderBuilder;

/**
 * Evaluate Binary Classification Model
 */
public class App {
    public static void main(String[] args) {
        String filePath = "model_3.csv";
        List<Integer> actualValues = new ArrayList<>();
        List<Double> predictedProbs = new ArrayList<>();

        // Read CSV File
        try (FileReader filereader = new FileReader(filePath);
             CSVReader csvReader = new CSVReaderBuilder(filereader).withSkipLines(1).build()) {

            List<String[]> allData = csvReader.readAll();

            for (String[] row : allData) {
                int y_true = Integer.parseInt(row[0]);  // Actual labels (0 or 1)
                double y_predicted = Double.parseDouble(row[1]);  // Predicted probability (0.0 - 1.0)

                actualValues.add(y_true);
                predictedProbs.add(y_predicted);
            }

        } catch (Exception e) {
            System.out.println("Error reading the CSV file");
            return;
        }

        // Convert predicted probabilities to binary labels (threshold = 0.5)
        List<Integer> predictedLabels = new ArrayList<>();
        for (double prob : predictedProbs) {
            predictedLabels.add(prob >= 0.5 ? 1 : 0);
        }

        // Calculate Metrics
        double bce = calculateBCE(actualValues, predictedProbs);
        int[][] confusionMatrix = calculateConfusionMatrix(actualValues, predictedLabels);
        double accuracy = calculateAccuracy(confusionMatrix);
        double precision = calculatePrecision(confusionMatrix);
        double recall = calculateRecall(confusionMatrix);
        double f1Score = calculateF1Score(precision, recall);
        double aucRoc = calculateAUC(actualValues, predictedProbs);

        // Display Results
		System.out.println("Evaluation Results For model_3.csv:");
        System.out.println("Binary Cross-Entropy (BCE): " + bce);
        System.out.println("Confusion Matrix:");
        System.out.println("TP: " + confusionMatrix[1][1] + " | FP: " + confusionMatrix[0][1]);
        System.out.println("FN: " + confusionMatrix[1][0] + " | TN: " + confusionMatrix[0][0]);
        System.out.println("Accuracy: " + accuracy);
        System.out.println("Precision: " + precision);
        System.out.println("Recall: " + recall);
        System.out.println("F1 Score: " + f1Score);
        System.out.println("AUC-ROC: " + aucRoc);
    }

    // Binary Cross-Entropy (BCE) Loss
    public static double calculateBCE(List<Integer> actual, List<Double> predicted) {
        double sum = 0.0;
        int n = actual.size();
        for (int i = 0; i < n; i++) {
            double y = actual.get(i);
            double p = predicted.get(i);
            p = Math.max(Math.min(p, 1 - 1e-9), 1e-9);  // Prevent log(0)
            sum += y * Math.log(p) + (1 - y) * Math.log(1 - p);
        }
        return -sum / n;
    }

    // Confusion Matrix Calculation
    public static int[][] calculateConfusionMatrix(List<Integer> actual, List<Integer> predicted) {
        int[][] matrix = new int[2][2]; // [TN, FP] [FN, TP]
        for (int i = 0; i < actual.size(); i++) {
            int y_true = actual.get(i);
            int y_pred = predicted.get(i);
            matrix[y_true][y_pred]++;
        }
        return matrix;
    }

    // Accuracy Calculation
    public static double calculateAccuracy(int[][] matrix) {
        int TP = matrix[1][1], TN = matrix[0][0], FP = matrix[0][1], FN = matrix[1][0];
        return (double) (TP + TN) / (TP + TN + FP + FN);
    }

    // Precision Calculation
    public static double calculatePrecision(int[][] matrix) {
        int TP = matrix[1][1], FP = matrix[0][1];
        return TP + FP == 0 ? 0 : (double) TP / (TP + FP);
    }

    // Recall Calculation
    public static double calculateRecall(int[][] matrix) {
        int TP = matrix[1][1], FN = matrix[1][0];
        return TP + FN == 0 ? 0 : (double) TP / (TP + FN);
    }

    // F1 Score Calculation
    public static double calculateF1Score(double precision, double recall) {
        return (precision + recall) == 0 ? 0 : 2 * (precision * recall) / (precision + recall);
    }

    // AUC-ROC Calculation (Using Trapezoidal Rule)
    public static double calculateAUC(List<Integer> actual, List<Double> predicted) {
        List<double[]> pairs = new ArrayList<>();
        for (int i = 0; i < actual.size(); i++) {
            pairs.add(new double[]{predicted.get(i), actual.get(i)});
        }
        pairs.sort((a, b) -> Double.compare(b[0], a[0]));  // Sort by predicted probability (descending)

        int TP = 0, FP = 0, P = 0, N = 0;
        for (int label : actual) {
            if (label == 1) P++;
            else N++;
        }

        double prevFPR = 0.0, prevTPR = 0.0, auc = 0.0;
        for (double[] pair : pairs) {
            if (pair[1] == 1) TP++;
            else FP++;

            double TPR = (double) TP / P;  // True Positive Rate (Sensitivity)
            double FPR = (double) FP / N;  // False Positive Rate (1 - Specificity)

            auc += (FPR - prevFPR) * (TPR + prevTPR) / 2;  // Trapezoidal Rule
            prevFPR = FPR;
            prevTPR = TPR;
        }
        return auc;
    }
}