package com.ontariotechu.sofe3980U;

import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import com.opencsv.CSVReader;
import com.opencsv.CSVReaderBuilder;

/**
 * Evaluate Single Variable Continuous Regression
 */
public class App {
	public static void main(String[] args) {
		String filePath = "model_3.csv";
		FileReader filereader;
		List<String[]> allData;

		List<Double> actualValues = new ArrayList<>();
		List<Double> predictedValues = new ArrayList<>();

		try {
			filereader = new FileReader(filePath);
			CSVReader csvReader = new CSVReaderBuilder(filereader).withSkipLines(1).build();
			allData = csvReader.readAll();
		} catch (Exception e) {
			System.out.println("Error reading the CSV file");
			return;
		}

		for (String[] row : allData) {
			double y_true = Double.parseDouble(row[0]);
			double y_predicted = Double.parseDouble(row[1]);

			actualValues.add(y_true);
			predictedValues.add(y_predicted);
		}

		// Calculate error metrics
		double mse = calculateMSE(actualValues, predictedValues);
		double mae = calculateMAE(actualValues, predictedValues);
		double mare = calculateMARE(actualValues, predictedValues);

		// Display results
		System.out.println("Evaluation Results For model_3.csv:");
		System.out.println("MSE: " + mse);
		System.out.println("MAE: " + mae);
		System.out.println("MARE: " + mare + " %");
	}

	// Method to calculate Mean Squared Error (MSE)
	public static double calculateMSE(List<Double> actual, List<Double> predicted) {
		double sum = 0.0;
		int n = actual.size();
		for (int i = 0; i < n; i++) {
			double error = actual.get(i) - predicted.get(i);
			sum += error * error;
		}
		return sum / n;
	}

	// Method to calculate Mean Absolute Error (MAE)
	public static double calculateMAE(List<Double> actual, List<Double> predicted) {
		double sum = 0.0;
		int n = actual.size();
		for (int i = 0; i < n; i++) {
			sum += Math.abs(actual.get(i) - predicted.get(i));
		}
		return sum / n;
	}

	// Method to calculate Mean Absolute Relative Error (MARE)
	public static double calculateMARE(List<Double> actual, List<Double> predicted) {
		double sum = 0.0;
		int n = actual.size();
		for (int i = 0; i < n; i++) {
			if (actual.get(i) == 0) {
				continue; // Avoid division by zero
			}
			sum += Math.abs((actual.get(i) - predicted.get(i)) / actual.get(i));
		}
		return (sum / n) * 100;
	}
}
