---

# CART Classifier with Streamlit Integration

This project implements a **CART (Classification and Regression Tree)** Classifier in Python, designed to handle classification tasks. It provides a robust framework for training, testing, and evaluating decision trees using **gini** or **entropy** criteria. A **Streamlit** web interface is integrated to provide a user-friendly platform for interacting with the classifier.

---

## Features

### Core Functionality
- **Custom CART Implementation**:
  - Supports both **Gini Index** and **Entropy** as splitting criteria.
  - Handles dataset partitioning, threshold calculation, and weighted impurity computations.
- **Visualization**:
  - Displays the confusion matrix and other metrics after evaluation.
- **Interactive Tree Traversal**:
  - Allows users to visualize the decision tree traversal logic.

### Streamlit Integration
- **User Interface**:
  - Upload datasets via an intuitive Streamlit web app.
  - Configure parameters such as `max_depth`, `min_samples`, and `criterion` through sliders and dropdowns.
- **Real-Time Evaluation**:
  - Displays metrics like **Precision**, **Recall**, and **F1-Score** instantly after running the classifier.
  - Visualizes the decision tree structure and confusion matrix.

---

## Installation

### Prerequisites
- Python 3.8 or above

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cart-classifier.git
   cd cart-classifier
   ```

2. Install required libraries:

   The key libraries include:
   - `Streamlit`
   - `numpy`
   - `pandas`
   - `scikit-learn`
   - `matplotlib`

---

## Usage

### Running the CART Classifier
1. Modify the script `CART_Classifier.py` to load and train your dataset.
2. Train and evaluate the classifier by calling the methods:
   ```python
   from CART_Classifier import CartClassifier
   from DataHandler import DataHandler

   data = DataHandler("path_to_dataset.csv", label_column="target_column")
   classifier = CartClassifier(max_depth=5, min_samples=2, criterion="gini")
   classifier.fit(data)
   precision, recall, f1 = classifier.evaluate(data, "conf_matrix.png")
   print(f"Precision: {precision}, Recall: {recall}, F1-Score: {f1}")
   ```

### Running the Streamlit App
1. Start the Streamlit server:
   ```bash
   streamlit run app/main.py
   ```
2. Open your browser and navigate to [http://localhost:8501](http://localhost:8501).
3. Use the app to:
   - Upload a dataset.
   - Configure training parameters.
   - Train, evaluate, and visualize the decision tree.

---

## Output

The Output generated is a code and an image representing the decision tree and the confusion matrix of the model

**Example Code**

 ```python
  import sys
  
  def predict(x):
  
      if x[5] <= 9.795053004:
          if x[1] <= 9.0:
              if x[3] <= 21.0:
                  if x[0] <= 65.0:
                      return 0
                  else:
                      return 0
              else:
                  if x[0] <= 48.0:
                      return 1
                  else:
                      return 0
          else:
              if x[3] <= 13.0:
                  return 0
              else:
                  if x[3] <= 24.0:
                      return 1
                  else:
                      return 0
      else:
          if x[1] <= 7.0:
              if x[4] <= 18.0:
                  if x[4] <= 2.0:
                      return 1
                  else:
                      return 0
              else:
                  if x[0] <= 68.0:
                      return 1
                  else:
                      return 0
          else:
              if x[1] <= 11.0:
                  if x[4] <= 11.0:
                      return 1
                  else:
                      return 1
              else:
                  if x[3] <= 53.0:
                      return 1
                  else:
                      return 0
                  
  
  x = eval(sys.argv[1])
  result = predict(x)
  print(result)
 ```

---

## Key Features in the Streamlit App
- **Dataset Upload**: Allows users to upload a CSV dataset.
- **Model Configuration**:
  - Adjust `max_depth`, `min_samples`, and `criterion`.
- **Results Visualization**:
  - View metrics (Precision, Recall, F1-Score).
  - Display and save the confusion matrix.
  - Visualize the decision tree traversal logic.

---

## Example Workflow
1. **Upload Dataset**: Select a dataset via the Streamlit app.
2. **Configure Parameters**: Choose training parameters (e.g., `max_depth` = 5).
3. **Train the Model**: Run the CART classifier.
4. **View Results**: Check metrics, visualize the tree, and download outputs.

---

## Future Enhancements
- **Tree Visualization**: Implement graphical tree visualization using libraries like Graphviz.
- **Multi-Classifier Support**: Add support for other decision tree algorithms.
- **Model Export**: Enable saving and loading of trained models.

## Contact
For queries or contributions, reach out at **[aswajith707@gmail.com](mailto:aswajith707@gmail.com)**.

---
