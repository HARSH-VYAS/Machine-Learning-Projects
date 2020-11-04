# OCR + Regression on Image Data Set

1) Apply OCR using PyTesseract on images to find the total value spent and the date on which the amount was spent per receipt.
2) Apply Different Classification Algorithms to get the insights out of the data gathered from Task 1 

# Business Use Case (One Of Many):
Many employees upload their receipts to get an approval for the expense made by them either by travelling for company or team meeting in a restaurant. Everyone needs to fill up a long form where they need to explain every detail about how much money they spent, why they spent it, when they spent it, etc. We are trying to solve this lengthy process by apply different ML techniques. 
In addition Our Insights will tell Managers to whether approve the expense or not? 
# ML Use Case :
By Combining Image Processing and Machine Learning we are trying to gather as much accurate information as we can get from an Image. Here we tried to use LSTM algo using pyTesseract.
After the Total amount plus the date information is gathered. We can process the data and cluster them based on multiple factors like , (Very High Expense, Very Low Expense, Medium Category expense) . We are using K-means to cluster these data.
