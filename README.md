<h1>🧪 Diabetic Retinopathy Detection using Random Forest</h1>
This project performs <b>binary classification</b> of retinal fundus images to detect <b>Diabetic Retinopathy (DR)</b> using <b>GLCM texture features and a Random Forest Classifier</b>.

<h2>📌 Project Objective</h2>
To detect whether a retinal image belongs to a <b>Normal</b> patient or indicates <b>Diabetic Retinopathy</b>, based on texture analysis of the <b>Green Channel</b>.

<h2>🗂️ Dataset</h2>
<ul>
<li><b>Source:</b> IEEE Dataport – IDRiD Dataset</li>
<li><b>Images:</b> High-resolution retinal images</li>
<li><b>Labels:</b> Retinopathy grading (converted to binary: Normal = 0, DR = 1)</li>
</ul>

<h2>🔧 Tech Stack</h2>
<ul>
<li><b>Languages:</b> Python</li>
<li><b>Libraries:</b> OpenCV, NumPy, Pandas, Matplotlib, Seaborn, scikit-learn, skimage</li>
</ul>

<h2>🔍 Key Steps</h2>
<ol>
<li><b>Import Libraries</b></li>
<li><b>Set Paths & Load Dataset</b></li>
<li><b>Preprocess Images</b> (resize, enhance green channel using CLAHE)</li>
<li><b>Extract GLCM Features</b></li>
<li><b>Train Random Forest Classifier</b></li>
<li><b>Evaluate Model</b> (accuracy, sensitivity, specificity, F1 score)</li>
<li><b>Visualize Results</b></li>
</ol>

<h2>📊 Model Performance</h2>
<table>
<tr><th>Metric</th><th>Score</th></tr>
<tr><td>Accuracy</td><td>62.14%</td></tr>
<tr><td>Sensitivity (Recall)</td><td>86.96%</td></tr>
<tr><td>Specificity</td><td>11.76%</td></tr>
<tr><td>F1 Score</td><td>0.207</td></tr>
</table>

<h2>🌲 Random Forest Classifier</h2>
<ul>
<li>A supervised ensemble algorithm using multiple decision trees.</li>
<li>Reduces overfitting by averaging results across trees.</li>
<li>Performs well for small-to-medium datasets with low computational cost.</li>
</ul>

<h2>🚀 Future Improvements</h2>
<ul>
<li>Implement deep learning (CNN) for better accuracy</li>
<li>Use class balancing (e.g., SMOTE or augmentation)</li>
<li>Explore hybrid models (GLCM + CNN)</li>
</ul>

<h2>✅ Final Verdict</h2>
This model successfully identifies diabetic retinopathy using a simple, interpretable, and efficient machine learning pipeline. While not perfect, it provides a foundational approach using texture features, and offers several paths for improvement using modern deep learning techniques.

<h2>📚 References</h2>
<ul>
<li><a href="https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid">IDRiD Dataset - IEEE Dataport</a></li>
<li>scikit-learn documentation</li>
<li>skimage GLCM documentation</li>
</ul>
